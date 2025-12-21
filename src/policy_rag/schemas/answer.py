from __future__ import annotations

from typing import Optional, Literal

# BaseModel：“带强校验的dataclass + 自动 JSON 解析/导出工具“
# Field：给字段加“约束 + 默认值 + 文档描述“的工具
from pydantic import BaseModel, Field

"""
AskResult
├─ AskAnswer
│  ├─ Claim
│  │  ├─ Citation
│  │  └─ Citation
│  └─ Claim
└─ Refusal
"""

# Citation：引用（证据指针），代表某条结论的证据来自哪里
class Citation(BaseModel):
    # 指向我们给模型的第几条证据
    source_id: int = Field(..., ge=1, description="Reference to provided sources (1-based).")
    # 从那条证据里摘出的原文片段（短引用）
    quote: str = Field(..., min_length=1, description="Short verbatim quote from the source chunk.")

# 一条可核对的结论（带证据）
class Claim(BaseModel):
    claim: str = Field(..., min_length=1) # 结论本身
    citations: list[Citation] = Field(..., min_length=1) # 支撑这条结论的证据列表
    confidence: Literal["high", "medium", "low"] = "medium"

# 正常回答的顶层结构
class AskAnswer(BaseModel):
    question: str
    # default_factory的意思是：当该字段没有被传入时，用一个“工厂函数“动态生成默认值
    # 不用claims: list[Claim] = []是因为在 Python 中，[]这种可变对象如果作为默认值写在类定义中，容易出现“多个实例共享一个列表“的坑
    claims: list[Claim] = Field(default_factory=list) 
    follow_up_questions: list[str] = Field(default_factory=list())# 当用户问题不够具体时，引导追问
    warnings: list[str] = Field(default_factory=list())# 风险提示

# 拒答结构（证据不足时的标准输出）
class Refusal(BaseModel):
    question: str
    refusal: bool = True
    reason: str
    follow_up_questions: list[str] = Field(default_factory=list())
    warnings: list[str] = Field(default_factory=list())

