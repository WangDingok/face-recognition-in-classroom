import time
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

import cv2
import torch

from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.recognizer import FaceRecognizer
from core.attendance_db import AttendanceDatabase
from utils.display import FPSDisplay

from utils.file_utils import load_id_to_name
from config.settings import RecognitionConfig, parse_args


class FaceRecognitionProcessor:
    def __init__(self,
                 input_video_path: str,
                 output_video_path: str,
                 face_detector_model_path: str,
                 face_index_path: str,
                 face_label_path: str,
                 id_to_name_path: str,
                 config: RecognitionConfig,
                 device: str,
                 attendance_db_path: str = "output/attendance/attendance.db"):

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.face_detector_model_path = face_detector_model_path
        self.face_index_path = face_index_path
        self.face_label_path = face_label_path
        self.id_to_name_path = id_to_name_path
        self.config = config
        self.device = device

        self.detector = FaceDetector(self.face_detector_model_path, self.device)
        self.embedder = FaceEmbedder(config.norm_threshold, self.device)

        self.id_to_name = load_id_to_name(self.id_to_name_path)

        self.recognizer = FaceRecognizer(
            self.face_index_path,
            self.face_label_path,
            self.id_to_name,
            config.frame_skip_recognition,
            config.sim_threshold,
            config.label_strategy,
            config.majority_ratio,
            config.vote_sim_threshold,
            config.min_valid,
            config.score_strategy,
            config.top_k,
            config.sim_margin_update
        )

        self.fps_display = FPSDisplay(buffer_size=10)
        
        # Attendance tracking
        self.attendance_db = AttendanceDatabase(attendance_db_path)
        self.attendance_db.set_session_id()
        self.face_crops = {}

    def _save_face_image(self, face_crop, student_id: str, track_id: int) -> str:
        """
        Save face crop image ke folder attendance_images
        
        Args:
            face_crop: Face image (numpy array in BGR format)
            student_id: ID của học sinh
            track_id: Track ID từ detector
        
        Returns:
            Path đến image file yang disave
        """
        try:
            # Create folder structure: output/attendance_images/YYYY_MM_DD/
            date_folder = datetime.now().strftime("%Y_%m_%d")
            image_dir = Path(f"output/attendance_images/{date_folder}")
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # Filename: student_id_track_id_timestamp.jpg
            timestamp = datetime.now().strftime("%H_%M_%S_%f")[:-3]
            image_filename = f"{student_id}_{track_id}_{timestamp}.jpg"
            image_path = image_dir / image_filename
            
            # Save image với quality 90%
            cv2.imwrite(str(image_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            return str(image_path)
        
        except Exception as e:
            print(f"❌ Error saving face image: {e}")
            return None
    
    def _save_attendance_record(self, label: str, student_name: str, 
                               face_crop, track_id: int, confidence: float):
        """
        Save attendance record ke database
        
        Args:
            label: Student ID (từ recognizer label)
            student_name: Student name (từ id_to_name mapping)
            face_crop: Face crop image
            track_id: Track ID
            confidence: Confidence score
        """
        try:
            # Save image
            image_path = self._save_face_image(face_crop, label, track_id)
            
            # Insert record ke database
            record_id = self.attendance_db.insert_record(
                student_id=label,
                student_name=student_name,
                image_path=image_path,
                confidence=confidence
            )
            
            if record_id > 0:
                print(f"✅ Recorded: {student_name} (ID: {label}) at {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Error saving attendance record: {e}")

    def _process_attendance_records(self, prev_recorded_labels: set):
        """
        Check untuk new recognized students dan save attendance records
        Dijalankan setelah recognize() untuk capture students yang terdeteksi pertama kali
        
        Args:
            prev_recorded_labels: Set dari labels yang sudah dicatat sebelumnya
        """
        try:
            # Loop through best_face_memory (track_id -> (sim, label, emb))
            for track_id, (confidence, label, emb) in self.recognizer.best_face_memory.items():
                # Skip "unknown" labels
                if label == "unknown" or label not in self.recognizer.label_to_global_id:
                    continue
                
                # Check jika label ini sudah dicatat sebelumnya
                if label in prev_recorded_labels:
                    continue
                
                # Check jika confidence >= threshold
                if confidence < self.recognizer.sim_threshold:
                    continue
                
                # Get face crop dari cache
                if track_id not in self.face_crops:
                    continue
                
                face_crop = self.face_crops[track_id]
                
                # Get student name
                student_name = self.id_to_name.get(label, label) if self.id_to_name else label
                
                # Save attendance record
                self._save_attendance_record(label, student_name, face_crop, track_id, confidence)
                
                # Mark label as recorded
                prev_recorded_labels.add(label)
        
        except Exception as e:
            print(f"❌ Error processing attendance records: {e}")

    def _update_last_seen_times(self, track_ids):
        """
        Update last_seen_time untuk students yang masih terlihat di frame ini
        
        Args:
            track_ids: List track IDs dari detector
        """
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            session_id = self.attendance_db.session_id
            
            # Loop through track IDs dan update last_seen_time
            for track_id in track_ids:
                # Get label dari recognizer
                if track_id in self.recognizer.best_face_memory:
                    confidence, label, emb = self.recognizer.best_face_memory[track_id]
                    
                    # Chỉ update nếu recognized (confidence >= threshold)
                    if label != "unknown" and confidence >= self.recognizer.sim_threshold:
                        # Update last_seen_time trong database
                        self.attendance_db.update_last_seen_time(session_id, label, current_time)
        
        except Exception as e:
            print(f"❌ Error updating last_seen_times: {e}")

    def _export_attendance_csv(self):
        """Export attendance records ke CSV file sau khi video selesai"""
        try:
            session_id = self.attendance_db.session_id
            csv_path = f"output/attendance/attendance_{session_id}.csv"
            
            # Finalize duration calculations
            updated = self.attendance_db.finalize_duration(session_id)
            print(f"✅ Duration calculated for {updated} records")
            
            success = self.attendance_db.export_csv(
                output_path=csv_path,
                session_id=session_id
            )
            
            if success:
                print(f"\n📊 Attendance Summary:")
                print(f"   Session: {session_id}")
                print(f"   CSV File: {csv_path}")
                
                # Show summary stats
                records = self.attendance_db.get_session_records(session_id)
                print(f"   Total Records: {len(records)}")
                print(f"   Unique Students: {len(set(r[2] for r in records))}")  # Unique student_id
        
        except Exception as e:
            print(f"❌ Error exporting attendance CSV: {e}")

    def __repr__(self):
        return (
            f"<FaceRecognitionProcessor(\n"
            f"  input_video_path = {self.input_video_path},\n"
            f"  output_video_path = {self.output_video_path},\n"
            f"  face_detector_model_path = {self.face_detector_model_path},\n"
            f"  face_index_path = {self.face_index_path},\n"
            f"  face_label_path = {self.face_label_path},\n"
            f"  id_to_name_path = {self.id_to_name_path},\n"
            f"  device = {self.device},\n"
            f"  config = {json.dumps(asdict(self.config), indent=4)}\n"
            f")>"
        )

    def draw_all(self, frame, boxes):
        frame = self.detector.draw_detected_boxes(frame, boxes)
        frame = self.recognizer.annotate_faces(frame, boxes)
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        self.fps_display.fps_video = fps

        frame_count = 0
        fps_detect_last, fps_embed_last, fps_query_last = 0.0, 0.0, 0.0
        prev_recorded_labels = set()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_total = time.time()

                boxes, fps_detect, fps_embed, fps_query = [], 0.0, 0.0, 0.0

                if frame_count % self.config.frame_skip_tracking == 0:
                    boxes, fps_detect = self.detector.detect(frame)
                    fps_detect_last = fps_detect

                if boxes:
                    faces_rgb, track_ids = self.detector.crop(frame, boxes)

                    self.face_crops.clear()
                    for track_id, face_rgb in zip(track_ids, faces_rgb):
                        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                        self.face_crops[track_id] = face_bgr

                    if frame_count % self.config.frame_skip_recognition == 0:
                        embeddings, fps_embed = self.embedder.extract_embeddings(faces_rgb)
                        fps_embed_last = fps_embed

                        if embeddings:
                            fps_query = self.recognizer.recognize(embeddings, track_ids, frame_count)
                            fps_query_last = fps_query
                            self._process_attendance_records(prev_recorded_labels)

                    self._update_last_seen_times(track_ids)
                    self.draw_all(frame, boxes)

                fps_process = 1.0 / (time.time() - start_total + 1e-6)

                self.fps_display.update(
                    detect=fps_detect_last,
                    embed=fps_embed_last,
                    query=fps_query_last,
                    total=fps_process
                )

                frame = self.fps_display.draw(frame)
                out.write(frame)

                small_frame = cv2.resize(frame, (1280, 720))
                cv2.imshow("Face Recognition", small_frame)

                elapsed_total = time.time() - start_total
                fps_total = 1.0 / (elapsed_total + 1e-6)

                wait_time = int(1000 / fps - elapsed_total * 1000)
                if wait_time > 0:
                    key = cv2.waitKey(wait_time) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                print(f'[Process FPS] {fps_process:.2f}, [Total FPS] {fps_total:.2f} for frame {frame_count}')

                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nVideo processing interrupted. Exporting attendance CSV...")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            self._export_attendance_csv()
            print(f"Video saved to: {self.output_video_path}")


if __name__ == '__main__':

    args = parse_args()

    if args.input_video_path is None:
        raise SystemExit("Missing --input_video_path")
    if args.output_video_path is None:
        raise SystemExit("Missing --output_video_path")
    if args.face_detector_model_path is None:
        raise SystemExit("Missing --face_detector_model_path")
    if args.face_index_path is None:
        raise SystemExit("Missing --face_index_path")
    if args.face_label_path is None:
        raise SystemExit("Missing --face_label_path")
    if args.id_to_name_path is None:
        raise SystemExit("Missing --id_to_name_path")

    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up recognition config
    recognition_config = RecognitionConfig(
        frame_skip_tracking=1,
        frame_skip_recognition=12,
        norm_threshold=0,
        sim_threshold=0.6,
        vote_sim_threshold=0.6,
        top_k=100,
        label_strategy='soft',
        majority_ratio=0.7,
        min_valid=10,
        score_strategy='max',
        sim_margin_update=0.1,
    )

    processor = FaceRecognitionProcessor(
        input_video_path=args.input_video_path,
        output_video_path=args.output_video_path,
        face_detector_model_path=args.face_detector_model_path,
        face_index_path=args.face_index_path,
        face_label_path=args.face_label_path,
        id_to_name_path=args.id_to_name_path,
        config=recognition_config,
        device=args.device
    )

    print(processor)

    processor.run()

