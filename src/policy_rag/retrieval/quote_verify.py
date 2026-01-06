from __future__ import annotations

import re


def _norm(s: str) -> str:
    """
    温和归一化：把所有空白（换行/制表/多空格）压成单空格，首尾去空。
    不做激进去标点，避免误判。
    """
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def quote_in_text(quote: str, source_text: str) -> bool:
    """
    返回 quote 是否“可在 source_text 中找到”。
    处理一些常见模型输出习惯：末尾省略号、引号、换行差异。
    """
    q = _norm(quote)
    t = _norm(source_text)

    if not q:
        return False

    # 去掉两端常见引号
    q = q.strip("“”\"'")

    # 如果模型在 quote 末尾加了省略号，做容错
    q2 = q.rstrip("…").rstrip("...").strip()

    if q in t:
        return True
    if q2 and q2 in t:
        return True

    return False
