from __future__ import annotations

import json
from typing import Any


def _escape_control_chars_inside_json_strings(s: str) -> str:
    """
    Escape raw control characters inside JSON strings, e.g. actual newline in
    "...028-61830<LF>511..." -> "...028-61830\\n511..."
    """
    out: list[str] = []
    in_str = False
    escaped = False

    for ch in s:
        if in_str:
            if escaped:
                out.append(ch)
                escaped = False
                continue

            if ch == "\\":  # start escape sequence
                out.append(ch)
                escaped = True
            elif ch == '"':  # end of string
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_str = True
            else:
                out.append(ch)

    return "".join(out)


def extract_first_json(text: str) -> Any:
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty LLM output, no JSON found.")

    obj_i = t.find("{")
    arr_i = t.find("[")
    starts = [i for i in [obj_i, arr_i] if i != -1]
    if not starts:
        raise ValueError("No JSON object/array start found in LLM output.")

    start = min(starts)
    s = t[start:]

    decoder = json.JSONDecoder()

    def _try_parse(payload: str) -> Any:
        obj, _end = decoder.raw_decode(payload)
        return obj

    try:
        return _try_parse(s)
    except json.JSONDecodeError:
        # Fallback: fix common Ollama/LLM issue — raw newlines inside JSON strings
        fixed = _escape_control_chars_inside_json_strings(s)
        try:
            return _try_parse(fixed)
        except json.JSONDecodeError as e:
            lo = max(0, e.pos - 120)
            hi = min(len(fixed), e.pos + 120)
            context = fixed[lo:hi].replace("\n", "\\n")
            raise ValueError(
                f"Invalid JSON from LLM even after sanitization: {e}. "
                f"Context around pos {e.pos}: {context}"
            ) from e
