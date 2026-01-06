from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # 把Chroma想成一个数据库
        # 创建一个Chroma的“持久化客户端“，连接到一个数据库实例（数据存在硬盘上）
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # 去找（或创建）数据库中的一张表（存向量+文本+元数据）
        self.collection = self.client.get_or_create_collection(name=collection_name)

    # Chroma 天然只有 4 类字段：id、embedding、document、metadata
    def upsert(
        self,
        ids: list[str], # 每条记录的唯一主键
        documents: list[str], # 每条向量对于的原文文本，向量只用于找 top-k 证据，真正给用户看的引用片段来自 documents
        embeddings: list[list[float]], # 每条文本的向量表示
        metadatas: list[dict[str, Any]]  # 一个dict：自己塞什么键值都行
    ):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()
    
    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: Optional[dict[str, Any]] = None # 用于限定查询范围，再做向量相似度检索
    ) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"] # 默认会返回ids
        )
    
    def get(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        return self.collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
    
    def delete(
        self,
        where = None,
        ids = None,
    ) -> None:
        self.collection.delete(where=where, ids=ids)