import os
import time
from glob import glob
from typing import List, Tuple
import numpy as np
import cv2
import insightface


class FaceEmbedder:
    def __init__(self, norm_threshold: float = 21.0, device: str = 'cuda'):
        self.model = insightface.model_zoo.get_model('buffalo_l', download=True)
        ctx_id = 0 if device == 'cuda' else -1
        self.model.prepare(ctx_id=ctx_id)
        print(self.model.__dict__)
        self.norm_threshold = norm_threshold

    def extract_embeddings(self, faces_rgb: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        start_time = time.time()
        batch_embeddings = self.model.get_feat(faces_rgb)
        elapsed = time.time() - start_time
        fps_embed = 1.0 / (elapsed + 1e-6)

        embeddings = []
        for emb in batch_embeddings:
            norm = np.linalg.norm(emb)
            if norm >= self.norm_threshold:
                embeddings.append(emb)
        return embeddings, fps_embed

    def process_folder(self,
                       folder_path: str,
                       batch_size: int = 32,
                       is_aligned: bool = True,
                       use_norm: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:

        all_embeddings = []
        all_labels = []

        imgs_rgb = []
        labels_batch = []

        for label in os.listdir(folder_path):
            label_dir = os.path.join(folder_path, label)
            if not os.path.isdir(label_dir):
                continue

            for img_path in glob(os.path.join(label_dir, "*.*")):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[!] Không đọc được ảnh: {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_rgb.append(img_rgb)
                labels_batch.append(label)

                if len(imgs_rgb) == batch_size and is_aligned:
                    embs, _ = self.extract_embeddings(imgs_rgb)
                    if use_norm:
                        norms = np.linalg.norm(embs, axis=1)
                        valid_idx = norms >= self.norm_threshold
                        embs = embs[valid_idx]
                        labels = [l for i, l in enumerate(labels_batch) if valid_idx[i]]
                    else:
                        labels = labels_batch[:len(embs)]

                    all_embeddings.append(embs)
                    all_labels.extend(labels)
                    imgs_rgb, labels_batch = [], []

                else:
                    print(f"Chưa đủ batch ({len(imgs_rgb)}), hoặc chưa bật aligned.")

        # batch cuối
        if imgs_rgb:
            embs, _ = self.extract_embeddings(imgs_rgb)
            if use_norm:
                norms = np.linalg.norm(embs, axis=1)
                valid_idx = norms >= self.norm_threshold
                embs = embs[valid_idx]
                labels = [l for i, l in enumerate(labels_batch) if valid_idx[i]]
            else:
                labels = labels_batch[:len(embs)]

            all_embeddings.append(embs)
            all_labels.extend(labels)

        # Stack tất cả thành 1 np.ndarray
        if all_embeddings:
            stacked_embs = np.vstack(all_embeddings).astype("float32")
        else:
            stacked_embs = np.empty((0, self.model.embedding_size), dtype="float32")

        return stacked_embs, all_labels

if __name__ == '__main__':

    from core.database import FaceIndexDatabase

    # Init
    db = FaceIndexDatabase()
    db.load("face_index.index", "face_labels.pkl")

    embedder = FaceEmbedder()

    # Add new person
    embs, labels = embedder.process_folder("all_person", is_aligned=True)
    db.add_batch(embs, labels)

    # Save
    db.save("face_index.index", "face_labels.pkl", overwrite=False)

    # Query
    import numpy as np

    query_vec = np.random.rand(512)
    results = db.query(query_vec, k=5)
    for i, (label, score) in enumerate(results):
        print(f"{i + 1}. {label} - score: {score:.2f}")

