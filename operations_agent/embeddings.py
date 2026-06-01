"""Embedding clients behind a small Protocol.

Duplicate detection needs to turn ticket text into vectors. The real client uses
a local sentence-transformers model (free, offline, recruiter-reproducible); the
fake client is deterministic and dependency-free so tests never download a model
or hit a network.

The Protocol is intentionally tiny — ``embed(texts) -> list[vector]`` — so the
index and duplicate logic never care which implementation they got.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one vector per input text."""
        ...


class LocalEmbeddingClient:
    """sentence-transformers embeddings, model loaded lazily on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # loaded on first embed() — keeps import cheap

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[Vector]:
        model = self._ensure_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


class FakeEmbeddingClient:
    """Deterministic, dependency-free embeddings for tests and offline runs.

    Builds a small bag-of-words vector over a fixed feature space derived from
    each token's hash. Paraphrases that share words land near each other, which
    is enough to exercise the retrieve-then-adjudicate flow without a real model.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> Vector:
        vec = [0.0] * self._dim
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]


def _tokenize(text: str) -> list[str]:
    return [w for w in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if len(w) > 2]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity. Vectors need not be pre-normalized."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
