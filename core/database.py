import os
import pickle
from typing import List
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize


class FaceIndexDatabase:
    def __init__(self, dim: int = 512, use_cosine: bool = True):
        self.dim = dim
        self.use_cosine = use_cosine
        self.index = faiss.IndexFlatIP(dim) if use_cosine else faiss.IndexFlatL2(dim)
        self.labels = []

    def load(self, index_path="face_index.index", label_path="face_labels.pkl"):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            print(f"Không tìm thấy index tại {index_path}, tạo index mới.")

        if os.path.exists(label_path):
            with open(label_path, "rb") as f:
                self.labels = pickle.load(f)
        else:
            print(f"Không tìm thấy label tại {label_path}, khởi tạo nhãn rỗng.")

    def save(self, index_path="face_index.index", label_path="face_labels.pkl", overwrite=True):
        if not overwrite and os.path.exists(index_path) and os.path.exists(label_path):
            # Nếu không ghi đè, ta sẽ tự động load và append
            print(f"[i] Đang load index cũ từ: {index_path}")
            old_index = faiss.read_index(index_path)
            with open(label_path, "rb") as f:
                old_labels = pickle.load(f)

            # Append embedding + labels mới
            old_index.add(self.index.xb)
            all_labels = old_labels + self.labels

            # Save lại
            faiss.write_index(old_index, index_path)
            with open(label_path, "wb") as f:
                pickle.dump(all_labels, f)

            print(f"[+] Đã ghi thêm vào index cũ (append mode). Tổng: {old_index.ntotal} embeddings.")
            return

        # Nếu overwrite=True hoặc chưa có file → Ghi đè bình thường
        faiss.write_index(self.index, index_path)
        with open(label_path, "wb") as f:
            pickle.dump(self.labels, f)
        print(f"[✓] Đã lưu index mới ({self.index.ntotal} embeddings).")

    def load_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, -1].astype(str).tolist()
        embeddings = df.iloc[:, :-1].values.astype("float32")
        if self.use_cosine:
            embeddings = normalize(embeddings, norm='l2')
        self.index.add(embeddings)

    def add_batch(self, embeddings: List[np.ndarray], labels: List[str]):
        feats = np.stack(embeddings).astype("float32")
        if self.use_cosine:
            feats = normalize(feats, norm='l2')
        self.index.add(feats)
        self.labels.extend(labels)

    def add_embedding(self, embedding: np.ndarray, label: str):
        vec = np.array(embedding, dtype="float32").reshape(1, -1)
        if self.use_cosine:
            vec = normalize(vec, norm='l2')
        self.index.add(vec)
        self.labels.append(label)

    def query(self, embedding: np.ndarray, k: int = 1):
        vec = np.array(embedding, dtype="float32").reshape(1, -1)
        if self.use_cosine:
            vec = normalize(vec, norm='l2')
        D, I = self.index.search(vec, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.labels):
                results.append((self.labels[idx], score))
            else:
                results.append(("unknown", -1))
        return results

    def query_batch(self, embeddings: np.ndarray, k: int = 1):
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)
        embeddings = embeddings.astype("float32")
        if self.use_cosine:
            embeddings = normalize(embeddings, norm='l2')

        D, I = self.index.search(embeddings, k)
        results_batch = []
        for d_row, i_row in zip(D, I):
            result = []
            for score, idx in zip(d_row, i_row):
                if 0 <= idx < len(self.labels):
                    result.append((self.labels[idx], score))
                else:
                    result.append(("unknown", -1))
            results_batch.append(result)
        return results_batch

    def get_all_labels(self):
        return list(set(self.labels))

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def remove_by_label(self, label: str):
        indices_to_keep = [i for i, l in enumerate(self.labels) if l != label]
        if not indices_to_keep:
            self.index.reset()
            self.labels = []
            return
        kept_embs = self.index.reconstruct_n(0, self.index.ntotal)
        kept_embs = [kept_embs[i] for i in indices_to_keep]
        self.index.reset()
        self.index.add(np.array(kept_embs).astype('float32'))
        self.labels = [self.labels[i] for i in indices_to_keep]



if __name__ == '__main__':
    # Init
    face_index = FaceIndexDatabase(dim=512, use_cosine=True)

    # Load in csv
    face_index.load_from_csv('face_embedding_fusion_class_1_filtered.csv')

    # Query
    query_vec = np.random.rand(512)
    # top_result = face_index.query(query_vec, k=10)[0]
    # print(f"Label gần nhất: {top_result} với độ tương đồng: {top_result:.2f}")

    results = face_index.query(query_vec, k=10)
    for rank, (label, score) in enumerate(results, 1):
        print(f"{rank}. {label} - similarity: {score:.2f}")

    # Save
    face_index.save("face_index_class_1_fusion_filtered.index", "face_labels_class_1_fusion_filtered.pkl")

    # Load
    # face_index.load("face_index.index", "face_labels.pkl")


