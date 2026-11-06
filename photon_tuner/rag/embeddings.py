"""光子 RAG 历史存储的嵌入模型。

两个可互换的嵌入器实现相同的 ``embed(text) -> Vector`` 契约：

* :class:`HashingEmbeddingModel` —— 无依赖的词袋哈希嵌入器，用于离线实验、
  CI 以及未安装 ``langchain-huggingface`` 包的任何场景。
* :class:`SentenceTransformerEmbeddingModel` —— 对 LangChain
  :class:`~langchain_huggingface.HuggingFaceEmbeddings` 的适配封装，用于
  更强语义检索值得额外依赖的场景。它同时暴露框架的
  ``embed_query``/``embed_documents`` 接口和项目的 ``embed(text)`` 别名。
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
    """返回 ``token`` 的确定性 64 位无符号槽位 id。"""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class HashingEmbeddingModel:
    """词袋哈希嵌入器，生成 L2 归一化的稀疏向量。

    每个 token 映射到一个固定维度并贡献 ``+1`` 或 ``-1``；符号来自同一 token
    的第二次哈希，使 ``+``/``-`` 的分配在多次运行间保持稳定。最终向量归一化为
    单位长度。该方法确定性强且开销低——正是离线模拟实验所需，这类场景中检索
    只需*稳定*，无需语义丰富。
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
    """对 LangChain ``HuggingFaceEmbeddings`` 的适配封装。

    保持项目的 ``embed(text)`` 契约，同时暴露框架的
    ``embed_query``/``embed_documents`` 方法，使同一对象可直接传入 LangChain
    的 ``VectorStore.from_texts``。
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or os.environ.get("PHOTON_TUNER_EMBEDDING_DEVICE", "cpu")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise ImportError(
                "Pretrained embeddings require the 'langchain-huggingface' package. "
                "Install it, or use --embedding-model hashing for the offline fallback."
            ) from exc
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed(self, text: str) -> Vector:
        return [float(value) for value in self.model.embed_query(text)]

    def embed_query(self, text: str) -> Vector:
        """LangChain ``Embeddings`` 接口：嵌入单个查询。"""
        return self.embed(text)

    def embed_documents(self, texts: List[str]) -> List[Vector]:
        """LangChain ``Embeddings`` 接口：批量嵌入文档。"""
        return [
            [float(value) for value in vector]
            for vector in self.model.embed_documents(list(texts))
        ]


def create_embedding_model(model_name: str) -> Any:
    """工厂函数：``"hashing"`` 构建离线嵌入器，否则构建预训练嵌入器。"""
    if model_name == "hashing":
        return HashingEmbeddingModel()
    return SentenceTransformerEmbeddingModel(model_name)
