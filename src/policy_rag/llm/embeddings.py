# sentence-Transformer：负责“把文本变向量“
# Chroma：负责“存向量并做相似度检索“
# metadata：负责“引用可追溯“

# 该脚本把一批文本统一编码成向量，并做缓存与格式统一，供向量库（chroma）入库/检索使用

from __future__ import annotations

# lru_cache 装饰器的作用是：把函数的返回结果缓存起来，同样的参数再次调用时，不再重复计算，而是直接从缓存里拿结果
from functools import lru_cache
from typing import List

import numpy as np
# 用于把文本变成向量，Embedding 模型
from sentence_transformers import SentenceTransformer

# maxsize=1 表示只保留一组输入、输出的缓存
@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> list[list[float]]:
    model = _get_model(model_name)
    # vecs默认输出为np.ndarray
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    if isinstance(vecs, np.ndarray):
        vecs.astype("float32").tolist()
    return [v.astype("float32").tolist() for v in vecs]