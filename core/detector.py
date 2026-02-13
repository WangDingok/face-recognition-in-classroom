import time
import cv2
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model = YOLO(model_path).to(device)

    def detect(self, frame):
        start_time = time.time()
        results = self.model.track(
            frame,
            tracker='bytetrack.yaml',
            persist=True,
            verbose=False,
            conf=0.25  # Default in YOLO
        )
        elapsed = time.time() - start_time
        fps_detect = 1.0 / (elapsed + 1e-6)

        boxes = results[0].boxes if results and len(results) > 0 else []
        return boxes, fps_detect

    def crop(self, frame, boxes):
        faces_rgb, track_ids = [], []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id) if box.id is not None else -1
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            faces_rgb.append(face_rgb)
            track_ids.append(track_id)
        return faces_rgb, track_ids

    def detect_and_crop(self, frame):
        boxes, fps_detect = self.detect(frame)
        faces_rgb, track_ids = self.crop(frame, boxes)
        return faces_rgb, track_ids, boxes, fps_detect

    def draw_detected_boxes(self, frame, boxes):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        return frame


