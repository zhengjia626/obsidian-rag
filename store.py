"""极简持久化向量库（余弦相似度检索）。

- 向量用 numpy 存储，元数据用 pickle 一并落盘
- 检索返回 (相似度, 元数据) 列表
"""
from __future__ import annotations

import os
import pickle

import numpy as np


class VectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.meta: list[dict] = []

    def add(self, vectors: list[list[float]], meta: list[dict]) -> None:
        self.vectors = np.array(vectors, dtype=np.float32)
        self.meta = meta

    def search(self, query_vec: list[float], top_k: int = 5) -> list[tuple[float, dict]]:
        if self.vectors is None or len(self.meta) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1.0
        sims = (self.vectors @ q) / (norms * qn)
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "meta": self.meta}, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store.vectors = data["vectors"]
        store.meta = data["meta"]
        return store
