from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ChatMessage:
    role: str # “system" | "user" | "assistant"
    content: str

class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.2, num_predict: int = 800):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict

    def chat(self, messages: list[ChatMessage]) -> str:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            }
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=240) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(
                f"Ollama request failed. Is Ollama running at {self.base_url}? "
                f"Error: {e}"
            ) from e
        
        obj = json.loads(raw)
        msg = obj.get("message", {})
        return str(msg.get("content", ""))