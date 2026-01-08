from __future__ import annotations

from fastapi import FastAPI

from policy_rag.api.routes_chat import router as chat_router
from policy_rag.api.routes_ingest import router as ingest_router
from policy_rag.api.routes_summary import router as summary_router

app = FastAPI(
    title="Policy RAG Assistant",
    version="0.2.1",
    description="Campus policy RAG asssistant (Phase 2 API)"
)

# 将子路由集合接到主APP上，让它们真正生效
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(summary_router)

@app.get("/health")
def health():
    return {"ok": True}