from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Optional

# 证据门控的判断结果，调用方只负责“怎么展示/怎么处理“
@dataclass
class EvidenceDecision:
    ok: bool # 证据是否足够支撑“生成确定性结论“
    reasons: list[str] # 若 ok=False，这里列出“不足“的具体原因
    suggestions: list[str] # 当 ok=False 时，给用户的“下一步怎么做“建议
    stats: dict[str, Any] # 门控时计算出来的一些关键指标

def assess_evidence(
    hits: list[Any],
    top1_max_dist: float,
    good_hit_max_dist: float,
    min_good_hits: int,
    min_gap: float,
) -> EvidenceDecision:
    """
    hits: list of RetrievedChunk (needs .distance and .metadata)
    distance: smaller is better
    """
    reasons: list[str] = []
    suggestions: list[str] = []

    if not hits:
        return EvidenceDecision(
            ok=False,
            reasons=["没有检索到任何候选证据（top-k 为空）"],
            suggestions=[
                "确认已完成 index-chunks 并且 Chroma collection 非空",
                "换一种问法（更接近条款原文关键词）",
                "限定 doc_id/category 再检索",
            ],
            stats={},
        )
    
    dists = [float(h.distance) for h in hits if h.distance is not None]
    if not dists:
        return EvidenceDecision(
            ok=False,
            reasons=["检索结果缺少 distance，无法判断证据强度"],
            suggestions=[
                "检查 embedding 生成是否正常",
                "检查 collection.query(include=['distances']) 是否返回距离",
            ],
            stats={},
        )
    
    top1 = dists[0]
    med = median(dists)
    gap = med - top1

    good_hits = sum(1 for d in dists if d <= good_hit_max_dist)

    # 统计检索到的文档id和页码，用于判断引用链是否可靠、以及在证据不足时给出更合理的提示
    uniq_docs = set()
    uniq_pages = set()

    for h in hits:
        md = getattr(h, "metadata", {}) or {}
        did = str(md.get("doc_id", "")).strip()
        pg = md.get("page_number", None)
        if did:
            uniq_docs.add(did)
        if pg is not None:
            try:
                uniq_pages.add(pg)
            except Exception:
                pass

    # 规则1：top1 必须足够近
    if top1 > top1_max_dist:
        reasons.append(f"最相关证据的距离偏大（top1 distance={top1:.4f} > {top1_max_dist:.4f}），相关性可能不足")
        suggestions.append("尝试把问题改得更具体：加入制度名称/奖项名称/身份/年份/关键条件等")

    # 规则2：好命中数量要足够
    if good_hits < min_good_hits:
        reasons.append(f"高相关证据数量不足（good_hits={good_hits} < {min_good_hits}，阈值={good_hit_max_dist:.4f}）")
        suggestions.append("可以限定范围：--doc-id 指定具体制度，或 --category 先缩小到奖学金/学籍/毕业等")

    # 规则3：区分度（top1 是否明显优于整体）
    if gap < min_gap:
        reasons.append(f"检索区分度不够（median-top1 gap={gap:.4f} < {min_gap:.4f}），结果可能是“泛相关”而非命中条款")
        suggestions.append("把问题改为条款式问法：例如“申请条件包含哪些硬性要求/材料清单/时间节点/例外条款？”")

    # 规则4：引用可用性（页码缺失的情况直接警告）
    if len(uniq_pages) == 0:
        reasons.append("候选证据缺少页码信息，无法形成可靠引用（page_number 缺失）")
        suggestions.append("确认 Step 0.2/0.3 的 pages.jsonl/chunks.jsonl 都包含 page_number，并重新 index-chunks")

    if len(uniq_docs) == 0:
        reasons.append("候选证据缺少 doc_id，无法追溯来源")
        suggestions.append("检查 chunks 入库时 metadata 是否写入 doc_id")

    ok = len(reasons) == 0

    # suggesions 去重
    seen = set()
    uniq_sugs = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            uniq_sugs.append(s)

    return EvidenceDecision(
        ok=ok,
        reasons=reasons,
        suggestions=uniq_sugs,
        stats={
            "top1": top1,
            "median": med,
            "gap": gap,
            "good_hits": good_hits,
            "uniq_docs": len(uniq_docs),
            "uniq_pages": len(uniq_pages),
        },
    )