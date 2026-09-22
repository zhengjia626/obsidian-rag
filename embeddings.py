"""嵌入后端。

默认 TfidfBackend：纯标准库实现，零下载即可运行（语义较弱，适合做骨架基线）。
可一键切换为 sentence-transformers（本地语义向量）或 openai（API 嵌入）。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

from obsidian_parser import tokenize


class EmbeddingBackend(ABC):
    dim: int = 0

    def fit(self, corpus: list[str]) -> None:  # noqa: D401 - 可选的语料拟合
        """部分后端（如 TF-IDF）需要先见语料来构建词表。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class TfidfBackend(EmbeddingBackend):
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.dim = 0

    def fit(self, corpus: list[str]) -> None:
        df = Counter()
        for doc in corpus:
            for tok in set(tokenize(doc)):
                df[tok] += 1
        n = max(1, len(corpus))
        self.idf = {
            tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()
        }
        self.vocab = {tok: i for i, tok in enumerate(sorted(self.idf))}
        self.dim = len(self.vocab)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.dim == 0:
            raise RuntimeError("TfidfBackend 需要先 fit() 再 embed()")
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            tf = Counter(tokenize(text))
            norm = (
                math.sqrt(
                    sum((self.idf.get(t, 0.0) * c) ** 2 for t, c in tf.items())
                )
                or 1.0
            )
            for tok, c in tf.items():
                if tok in self.vocab:
                    vec[self.vocab[tok]] = (self.idf[tok] * c) / norm
            out.append(vec)
        return out


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class OpenAIBackend(EmbeddingBackend):
    def __init__(
        self, model: str = "text-embedding-3-small", base_url: str | None = None
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url)
        self.model = model
        probe = self.client.embeddings.create(model=model, input="probe")
        self.dim = len(probe.data[0].embedding)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def build_backend(
    name: str, model: str | None = None, base_url: str | None = None
) -> EmbeddingBackend:
    if name == "tfidf":
        return TfidfBackend()
    if name == "sentence-transformers":
        return SentenceTransformerBackend(model or "all-MiniLM-L6-v2")
    if name == "openai":
        return OpenAIBackend(model or "text-embedding-3-small", base_url)
    raise ValueError(f"未知嵌入后端: {name}")
