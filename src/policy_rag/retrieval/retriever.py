from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from policy_rag.index.chroma_store import ChromaStore
from policy_rag.llm.embeddings import embed_texts

@dataclass
class RetrievedChunk:
    rank: int
    chunk_id: str
    distance: float
    text: str
    metadata: dict[str, Any]

def retrieve_top_k(
    store: ChromaStore,
    query: str,
    model_name: str,
    top_k: int = 8,
    where: Optional[dict[str, Any]] = None,
) -> list[RetrievedChunk]:
    q_emb = embed_texts([query], model_name=model_name, batch_size=1)
    res = store.query(q_emb, top_k, where)

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[RetrievedChunk] = []
    for i in range(len(ids)):
        out.append(
            RetrievedChunk(
                rank=i + 1,
                chunk_id=ids[i],
                distance=float(dists[i]) if dists[i] and dists[i] is not None else float("nan"),
                text=str(docs[i] or ""),
                metadata=metas[i],
            )
        )
    return out

def make_snippet(text: str, max_chars: int = 140) -> str:
    s = (text or "").replace("\n", "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "..."
