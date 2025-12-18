
# Campus Policy RAG Assistant（校园规章制度与奖学金政策 RAG 助手）

一个面向大学生的“政策/规章制度助手”MVP：把冗长、正式、条款繁杂的制度文件（优先 PDF）变成**可检索、可引用、可结构化总结**的知识库，帮助快速抓重点、避免漏掉关键条件/材料/截止日期/例外条款。

> 核心原则：**证据优先（Evidence-first）**  
> - 任何结论都必须能追溯到原文 chunk（文件名 + 页码/段落 + 原文片段）。  
> - 证据不足时必须明确提示“不确定/需要补充资料/建议核对官方最新版本”，禁止编造。

---

## 功能进度（当前已完成：Phase 0 → Step 0.1–0.5）

### ✅ 已完成
- **Step 0.1**：`docs.csv` 元数据校验（字段齐全 / doc_id 唯一 / 路径存在 / 日期格式合法）
- **Step 0.2**：PDF 解析为**按页文本**（保留页码），落盘 `pages.jsonl`
- **Step 0.3**：按页切块（chunking，保留页码与字符位置），落盘 `chunks.jsonl`
- **Step 0.4**：对 chunks 做 embedding 并写入本地 **Chroma**（持久化）
- **Step 0.5**：最小检索（query → top-k chunks），终端输出页码与原文片段（snippet）

### 🚧 进行中（下一步）
- **Step 0.6**：Evidence Gate（阈值拒答/追问） + 进入 Phase 1 的 `ask`（LLM 结构化答案 + 强制引用）

---

## 技术栈（当前）
- Python 3.10+
- CLI：Typer + Rich
- PDF 文本提取：pypdf
- Embedding：sentence-transformers（默认 `BAAI/bge-small-zh-v1.5`）
- 向量库：Chroma（本地持久化）

---

## 仓库结构（当前）

```bash
policy-rag-assistant/
├─ data/
│  ├─ raw/              # 原始 PDF（建议不提交到 Git）
│  ├─ metadata/
│  │  ├─ docs.csv
│  │  └─ docs.schema.json
│  ├─ parsed/           # 解析与切块产物（pages/chunks）
│  └─ index/            # 向量库持久化（chroma）
├─ src/policy_rag/
│  ├─ cli/              # CLI 命令入口
│  ├─ config/           # Settings
│  ├─ ingestion/        # parse/chunk/index
│  ├─ index/            # chroma store
│  ├─ llm/              # embeddings
│  └─ retrieval/        # retriever + snippet
└─ tests/
````

> 注意：`data/raw/`、`data/parsed/`、`data/index/` 默认应在 `.gitignore` 中忽略（体积大/可再生/可能涉隐私与版权）。

---

## 快速开始

### 1) 安装

在项目根目录：

```bash
pip install -e .
```

### 2) 准备一份示例 PDF 与 docs.csv

把 PDF 放到（示例）：

```bash
data/raw/scholarship/policy_sample.pdf
```

编辑 `data/metadata/docs.csv`（至少 1 行）：

```csv
doc_id,title,category,publish_date,effective_date,status,source_type,file_path,checksum
scholarship_2024_sample,研究生奖学金评定办法（示例）,scholarship,2024-09-01,2024-09-01,active,school_official,data/raw/scholarship/policy_sample.pdf,
```

---

## CLI 使用说明（Step 0.1–0.5）

### Step 0.1：校验元数据

```bash
policy-rag validate-metadata
```

### Step 0.2：解析 PDF → 按页落盘

```bash
policy-rag parse-pdf --doc-id scholarship_2024_sample
# 输出：data/parsed/<doc_id>/pages.jsonl
```

### Step 0.3：按页切块 → chunks.jsonl（保留页码与 char span）

```bash
policy-rag chunk-pages --doc-id scholarship_2024_sample
# 输出：data/parsed/<doc_id>/chunks.jsonl
```

可选参数（v0）：

```bash
policy-rag chunk-pages --doc-id scholarship_2024_sample --chunk-size 1000 --overlap 150 --min-chunk-chars 80
```

### Step 0.4：Embedding + 写入 Chroma（本地持久化）

```bash
policy-rag index-chunks --doc-id scholarship_2024_sample
# 输出：data/index/chroma/（Chroma 持久化目录）
```

### Step 0.5：最小检索（只验证证据检索与页码引用链）

```bash
policy-rag search --query "研究生奖学金的申请条件是什么？" --top-k 8
```

限定 doc/category：

```bash
policy-rag search --query "需要提交哪些材料？" --top-k 8 --doc-id scholarship_2024_sample
policy-rag search --query "截止时间是什么？" --top-k 8 --category scholarship
```

---

## 配置（环境变量）

可用环境变量覆盖默认配置：

```bash
export EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
export CHROMA_COLLECTION="policy_chunks"
```

> Windows PowerShell：

```powershell
$env:EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
$env:CHROMA_COLLECTION="policy_chunks"
```

---

## 数据与隐私说明

* 本项目默认不提交任何真实政策 PDF 到 Git 仓库（避免版权/隐私与仓库膨胀）。
* 解析产物（`data/parsed`）与索引（`data/index`）均可再生，建议忽略。
* 若需要公开 Demo，建议使用可公开分发的制度文件或自行脱敏处理。

---

## 未来开发计划（Roadmap）

### Phase 0：需求与数据准备（已完成到 Step 0.5）

* [x] 元数据字段与校验（docs.csv）
* [x] PDF 按页解析与落盘（保留页码）
* [x] v0 切块策略（页内字符切分 + overlap）
* [x] embedding + Chroma 入库
* [x] 最小检索与证据展示（页码 + snippet）
* [ ] Step 0.6：Evidence Gate（相似度阈值 + 证据不足拒答/追问策略）

### Phase 1：CLI MVP（跑通闭环）

* [ ] `ingest`：PDF → 解析 → 切块 → embedding → 入库（整合 Step0.2–0.4）
* [ ] `ask`：检索 top-k → **结构化答案**（逐条结论 + 引用）
* [ ] `summarize`：对指定 doc 输出**政策速览卡片**（固定字段 + 引用）
* [ ] `todo`：从证据中抽取流程/材料/时间节点 → checklist（建议也附引用）

### Phase 2：API 化（FastAPI）

* [ ] `POST /ingest`：上传并入库
* [ ] `POST /chat`：问答（可选流式）
* [ ] `GET /doc/{doc_id}/summary`：速览卡片
* [ ] OpenAPI 文档 + curl/Postman 示例

### Phase 3：前端 Demo（Streamlit）

* [ ] 政策列表与筛选（按 category/status）
* [ ] 聊天 UI + 引用可展开（页码/原文片段）
* [ ] 一键生成待办清单

### Phase 4：v1 工程化（质量/版本/评测/观测）

* [ ] 制度多版本共存：默认引用“最新现行版本”，显示版本依据（publish/effective/status）
* [ ] 混合检索：向量 + 关键词（BM25）+ 可选 rerank
* [ ] 回归评测集（30–80 常见问题）+ 自动对比报告
* [ ] Langfuse（可选）记录检索证据、prompt、耗时、token

### Phase 5：v2 扩展（可选）

* [ ] 扩展到政府公开政策：jurisdiction/document_type/validity 等字段
* [ ] 回答中展示“信息截至日期与来源”
* [ ] 更强文档解析：表格/扫描件 OCR、结构化条款抽取

---

## License

TBD（建议 MIT / Apache-2.0）

## Contributing

欢迎提 issue / PR（建议先从：切块优化、证据门控、评测集构建开始）。
