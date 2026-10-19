"""Embedding models for the photonic RAG history store.

Two interchangeable embedders implement the same ``embed(text) -> Vector``
contract:

* :class:`HashingEmbeddingModel` — a dependency-free bag-of-tokens hashing
  embedder used for offline experiments, CI, and anywhere the
  ``sentence-transformers`` package is not installed.
* :class:`SentenceTransformerEmbeddingModel` — a thin adapter over a pretrained
  ``sentence-transformers`` checkpoint, for when stronger semantic retrieval is
  worth the extra dependency.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, List

Vector = List[float]

_IDENTIFIER_OR_NUMBER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")


def stable_hash(token: str) -> int:
    """Return a deterministic 64-bit unsigned slot id for ``token``."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class HashingEmbeddingModel:
    """Bag-of-tokens hashing embedder producing an L2-normalised sparse vector.

    Every token maps to one fixed dimension and contributes ``+1`` or ``-1``;
    the sign comes from a second hash of the same token so the ``+``/``-`` split
    is stable across runs.  The final vector is normalised to unit length.  This
    is deterministic and cheap — exactly what the offline mock experiments need,
    where retrieval only has to be *stable*, not semantically rich.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"embedding dimension must be positive, got {dimension}")
        self.dimension = dimension

    def embed(self, text: str) -> Vector:
        accum = [0.0] * self.dimension
        for token in _IDENTIFIER_OR_NUMBER.findall(text.lower()):
            slot = stable_hash(token) % self.dimension
            sign = 1.0 if stable_hash(token + "#sign") % 2 == 0 else -1.0
            accum[slot] += sign
        norm = math.sqrt(sum(value * value for value in accum))
        if norm == 0.0:
            return accum
        return [value / norm for value in accum]


class SentenceTransformerEmbeddingModel:
    """Adapter that delegates ``embed`` to a pretrained sentence-transformer."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or os.environ.get("PHOTON_TUNER_EMBEDDING_DEVICE", "cpu")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Pretrained embeddings require the 'sentence-transformers' package. "
                "Install it, or use --embedding-model hashing for the offline fallback."
            ) from exc
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, text: str) -> Vector:
        encoded = self.model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in encoded]


def create_embedding_model(model_name: str) -> Any:
    """Factory: ``"hashing"`` builds the offline embedder, otherwise a pretrained one."""
    if model_name == "hashing":
        return HashingEmbeddingModel()
    return SentenceTransformerEmbeddingModel(model_name)
