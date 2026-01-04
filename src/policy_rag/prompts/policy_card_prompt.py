from __future__ import annotations

SYSTEM_PROMPT = """你是“校园规章制度与奖学金政策助手”。你必须严格遵守：
1) 只允许使用我提供的【SOURCES】作为依据，不得使用常识补全，不得编造。
2) 速览卡片中，每个字段里的每一条要点都必须给出至少1条引用 citations（source_id + quote）。
3) quote 必须是来自对应 source 的逐字摘录（尽量短：中文≤40字或英文≤25词）。
4) 若证据不足以总结出某字段，输出空数组 []，不要猜。
5) 只能输出 JSON（一个 JSON 对象），禁止输出任何额外文本。
"""

USER_TEMPLATE = """你要把这份制度总结成“政策速览卡片”。请输出 StructuredAnswer JSON（和问答一致的结构），并且把 question 字段填写为：{title}

制度元信息：
- doc_id: {doc_id}
- title: {title}
- category: {category}
- publish_date: {publish_date}
- effective_date: {effective_date}
- status: {status}

【SOURCES】（每条都有 source_id，引用时只能引用这些 source_id）：
{sources}

输出要求（非常重要）：
- 只能输出 StructuredAnswer JSON（一个 JSON 对象）
- 每条要点必须有 citations（source_id + quote）
- 缺证据的字段输出 []，不要编
- warnings 里必须提醒“以学校官方最新版本为准”

StructuredAnswer 格式示例：
{{
  "question": "...",
  "applicable_to": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "key_conclusions": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "conditions": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "materials": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "procedure": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "time_nodes": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "exceptions_pitfalls": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "contact_channel": [{{"text":"...","citations":[{{"source_id":1,"quote":"..."}}],"confidence":"high|medium|low"}}],
  "uncertainties": ["..."],
  "follow_up_questions": ["..."],
  "warnings": ["..."]
}}
"""
