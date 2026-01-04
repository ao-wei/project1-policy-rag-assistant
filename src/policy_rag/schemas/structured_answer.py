from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class Citation(BaseModel):
    source_id: int = Field(..., ge=1, description="Reference to provided sources (1-based).")
    quote: str = Field(..., min_length=1, description="Short verbatim quote from the source chunk.")

# 最小证据单元：用来表示一条“可以直接给用户执行/理解的政策要点“
# 一条“要点 + 证据“的原子单元
class Item(BaseModel):
    text: str = Field(..., min_length=1, description="A single actionable policy point.")
    citations: list[Citation] = Field(..., min_length=1)
    confidence: Literal["high", "medium", "low"] = "medium"

class StructuredAnswer(BaseModel):
    question: str

    # 固定模板字段（全是“要点列表“，便于 UI/Checklist 化）
    applicable_to: list[Item] = Field(default_factory=list, description="适用对象/范围")
    key_conclusions: list[Item] = Field(default_factory=list, description="核心结论/最重要结论")
    conditions: list[Item] = Field(default_factory=list, description="条件/资格/门槛（硬性要求）")
    materials: list[Item] = Field(default_factory=list, description="材料清单/证明材料")
    procedure: list[Item] = Field(default_factory=list, description="流程步骤/提交路径/审批链路")
    time_nodes: list[Item] = Field(default_factory=list, description="时间节点/截止日期/公示期等")
    exceptions_pitfalls: list[Item] = Field(default_factory=list, description="例外条款/坑点/不满足的常见原因")
    contact_channel: list[Item] = Field(default_factory=list, description="咨询渠道/联系部门/官方入口")

    # 不确定性与追问（允许无引用，但必须明确“缺证据“）
    uncertainties: list[str] = Field(default_factory=list, description="证据不足但用户常关心的信息点（明确写不确定）")
    follow_up_questions: list[str] = Field(default_factory=list, description="为了得到确定答案，需要用户补充的问题")
    warnings: list[str] = Field(default_factory=list, description="免责声明/核对最新版提醒")