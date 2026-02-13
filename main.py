import time
import json
from dataclasses import asdict

import cv2
import torch

from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.recognizer import FaceRecognizer
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
                 device: str):

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

                if frame_count % self.config.frame_skip_recognition == 0:
                    embeddings, fps_embed = self.embedder.extract_embeddings(faces_rgb)
                    fps_embed_last = fps_embed

                    if embeddings:
                        fps_query = self.recognizer.recognize(embeddings, track_ids, frame_count)
                        fps_query_last = fps_query

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

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved to: {self.output_video_path}")


if __name__ == '__main__':

    args = parse_args()

    args.input_video_path = r"C:\Users\LENOVO\Downloads\07-20250717T092757Z-1-001\data 16-07\class 1\20250715184906152_FY3407255_CAM_07_GiuaPhong_1_video.mov"
    args.output_video_path = 'face_recognition_tracking_insight_class_1_ver4.mp4'
    args.face_detector_model_path = 'yolo_face/yolov8m-face.pt'
    args.face_index_path = 'face db 1607/face_index_class_1_fusion_filtered.index'
    args.face_label_path = 'face db 1607/face_labels_class_1_fusion_filtered.pkl'
    args.id_to_name_path = 'face db 1607/id2label/class_1.json'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up recognition config
    recognition_config = RecognitionConfig(
        frame_skip_tracking=1,
        frame_skip_recognition=5,
        norm_threshold=0,
        sim_threshold=0.6,
        vote_sim_threshold=0.6,
        top_k=200,
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

