from __future__ import annotations

import json

def extract_first_json(text: str) -> str:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    start = t.find("{")
    end = t.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    
    candidate = t[start: end + 1]
    return json.loads(candidate)