import time
from typing import List, Literal
import cv2
import numpy as np

from core.database import FaceIndexDatabase
from core.resolver import LabelResolver


class FaceRecognizer:
    def __init__(self,
                 index_path: str,
                 label_path: str,
                 dict_id_to_name: dict[str | int, str],
                 frame_skip_recognition: int = 5,
                 sim_threshold: float = 0.6,
                 label_strategy: Literal['soft', 'hard'] = 'soft',
                 majority_ratio: float = 0.5,
                 vote_sim_threshold: float = 0.5,
                 min_valid: int = 10,
                 score_strategy: Literal['mean', 'max'] = 'max',
                 top_k: int = 200,
                 sim_margin_update: float = 0.1):

        self.face_index = FaceIndexDatabase(dim=512, use_cosine=True)
        self.face_index.load(index_path, label_path)
        self.id_to_name = dict_id_to_name
        self.frame_skip_recognition = frame_skip_recognition
        self.sim_threshold = sim_threshold
        self.label_resolver = LabelResolver(label_strategy,
                                            majority_ratio,
                                            vote_sim_threshold,
                                            min_valid,
                                            score_strategy
                                            )
        self.top_k = top_k
        self.sim_margin_update = sim_margin_update

        self.best_face_memory = {}  # track_id -> (sim, label, emb)
        self.face_id_map = {}  # track_id -> label

        self.label_to_global_id = {}  # Lưu map: {'user_123': 1, 'user_456': 2}
        self.next_global_id = 1  # Bộ đếm ID bắt đầu từ 1

    def query_embeddings(self, embeddings: List[np.ndarray]):
        fps_query = 0.0
        try:
            batch = np.stack(embeddings).astype(np.float32)
            start_query = time.time()
            results_batch = self.face_index.query_batch(batch, k=self.top_k)
            elapsed = time.time() - start_query
            fps_query = 1.0 / (elapsed + 1e-6)

            return results_batch, fps_query

        except Exception as e:
            print(f"Failed to query FAISS: {e}")
            return None, fps_query

    def resolve_labels(self, embeddings: List[np.ndarray], track_ids, results_batch, frame_count: int):
        for idx, (track_id, emb) in enumerate(zip(track_ids, embeddings)):
            results = results_batch[idx]
            best_label, best_sim = self.label_resolver(results)
            prev_sim, prev_label, _ = self.best_face_memory.get(track_id, (-1, "unknown", None))

            if prev_label != "unknown":
                if best_label != prev_label and best_sim >= prev_sim + self.sim_margin_update:
                    self.best_face_memory[track_id] = (best_sim, best_label, emb)
                elif best_label == prev_label and best_sim > prev_sim:
                    self.best_face_memory[track_id] = (best_sim, best_label, emb)
            else:
                self.best_face_memory[track_id] = (best_sim, best_label, emb)

            if frame_count % self.frame_skip_recognition == 0:
                self.face_id_map[track_id] = best_label if best_sim >= self.sim_threshold else "unknown"

            _, current_mem_label, _ = self.best_face_memory[track_id]
            if current_mem_label != "unknown" and current_mem_label not in self.label_to_global_id:
                self.label_to_global_id[current_mem_label] = self.next_global_id
                self.next_global_id += 1

    def recognize(self, embeddings: List[np.ndarray], track_ids, frame_count: int):
        results_batch, fps_query = self.query_embeddings(embeddings)
        if results_batch is not None:
            self.resolve_labels(embeddings, track_ids, results_batch, frame_count)
        return fps_query

    def annotate_faces(self, frame, boxes):
        for box in boxes:
            track_id = int(box.id) if box.id is not None else -1
            x1, y1, x2, _ = map(int, box.xyxy[0])

            sim_score, display_label, _ = self.best_face_memory.get(
                track_id, (-1, "unknown", None)
            )
            shown_label = display_label if sim_score >= self.sim_threshold else None
            shown_name = self.id_to_name.get(shown_label, shown_label) if self.id_to_name else shown_label

            if shown_label is not None:
                global_id = self.label_to_global_id.get(shown_label, f"T{track_id}")
                text_display = f"{shown_name} (ID:{global_id}) Sim:{sim_score:.2f}"
                cv2.putText(frame, text_display,
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                pass

        return frame