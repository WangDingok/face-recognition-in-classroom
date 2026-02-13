import os
import json
from typing import Callable
import numpy as np
from functools import wraps
import time

# FPS Calculator
def return_with_fps(func):
    @wraps(func)    
    def wrapper(*args, **kwargs):
        start_time = time.time()
    
        result = func(*args, **kwargs)
        
        elapsed = time.time() - start_time
        fps = 1.0 / (elapsed + 1e-6)
        
        return result, fps
        
    return wrapper

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

# Load từ điển ánh xạ ID → Tên
def load_id_to_name(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_embed_func(arcface_model, face_detector=None) -> Callable[[np.ndarray, bool], np.ndarray]:
    """
    Trả về embed_func(img, is_aligned) => 512-dim float32 embedding
    Nếu ảnh chưa align, cần có face_detector để detect + align tự động
    """
    def embed_func(img: np.ndarray, is_aligned: bool = False) -> np.ndarray:
        if is_aligned:
            feat = arcface_model.get_feat(img).flatten()
        else:
            if face_detector is None:
                raise ValueError("Cần face_detector nếu ảnh chưa align.")
            faces, _ = face_detector.crop(img)
            if not faces:
                raise ValueError("Không phát hiện khuôn mặt.")
            face = faces[0]
            feat = arcface_model.get(img, face)
        return feat.astype("float32")

    return embed_func
