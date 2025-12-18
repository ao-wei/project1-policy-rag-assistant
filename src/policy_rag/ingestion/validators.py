# 该脚本为入库前的质量闸门，专门用来检查维护的data/metadata/docs.csv 是否合格，避免将错误数据/错误路径/重复文档 ID 导入到知识库中
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = [
    "doc_id",
    "title",
    "category",
    "publish_date",
    "effective_date",
    "status",
    "source_type",
    "file_path",
]

OPTIONAL_COLUMNS = ["checksum"]

ALLOWED_STATUS = {"active", "repealed", "draft", "unknown"}

@dataclass
class ValidationIssue:
    level: str # "ERROR" or "WARN"
    row: int | None # 1-based row number in CSV (excluding header), or None for global issues
    field: str | None
    message: str

def _parse_iso_date(value: str) -> bool:
    if value.strip() == "":
        return True
    try:
        date.fromisoformat(value.strip())
        return True
    except ValueError:
        return False
    
def validate_docs_csv(csv_path: Path, repo_root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not csv_path.exists():
        issues.append(ValidationIssue("ERROR", None, None, f"docs.csv not found: {csv_path}"))
        return issues
    
    # utf-8-sig表示按UTF-8 读文件，如果开头有 BOM（部分软件保存 UTF-8 的 CSV 时，会在文件开头加 BOM 标记），就自动把 BOM 吃掉
    # newline=""是为了避免双重换行
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        # 用DictReader读取CSV：每一行会变成一个 dict
        reader = csv.DictReader(f)
        # 获取表头
        headers = reader.fieldnames or []
        # 检查“必需列“是否齐全
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]

        if missing:
            issues.append(ValidationIssue("ERROR", None, None, f"Missing required columns: {missing}"))
            return issues
        
        for c in OPTIONAL_COLUMNS:
            if c not in headers:
                issues.append(ValidationIssue("WARN", None, None, f"Optional colums missing(ok for now): {c}"))

        seen_doc_ids: set[str] = set()
        seen_paths: set[str] = set()

        for idx, row in enumerate(reader, start=1):
            # 注意这里不能写作 row.get("doc_id", "")，因为这样解决的是“ doc_id 键缺失“，不是“值为 None“
            doc_id = (row.get("doc_id") or "").strip()
            title = (row.get("title") or "").strip()
            file_path = (row.get("file_path") or "").strip()
            status = (row.get("status") or "").strip()
            publish_date = (row.get("publish_date") or "").strip()
            effective_date = (row.get("effective_date") or "").strip()

            # 必需字段空值检验
            for field in REQUIRED_COLUMNS:
                v = (row.get(field) or "").strip()
                if v == "":
                    issues.append(ValidationIssue("ERROR", idx, field, "Required field is empty"))

            # doc_id 唯一性检验
            if doc_id:
                if doc_id in seen_doc_ids:
                    issues.append(ValidationIssue("ERROR", idx, "doc_id", f"Duplicate doc_id: {doc_id}"))
                seen_doc_ids.add(doc_id)

            # file_path 存在性检验与唯一性检验
            if file_path:
                abs_path = (repo_root / file_path).resolve()
                if str(abs_path) in seen_paths:
                    issues.append(ValidationIssue("WARN", idx, "file_path", f"Duplicate file_path: {file_path}"))
                seen_paths.add(file_path)

                if not abs_path.exists():
                    issues.append(
                        ValidationIssue("ERROR", idx, "file_path", f"File not found: {file_path} (resolved: {abs_path})")
                    )
                elif abs_path.is_dir():
                    issues.append(ValidationIssue("ERROR", idx, "file_path", f"file_path points to a directory: {file_path}"))

            # status 限制
            if status and status not in ALLOWED_STATUS:
                issues.append(
                    ValidationIssue("WARN", idx, "status", f"Unknown status '{status}'. Allowed: {sorted(ALLOWED_STATUS)}")
                )

            # ISO date 格式检验
            if publish_date and not _parse_iso_date(publish_date):
                issues.append(ValidationIssue("ERROR", idx, "publish_date", "Invalid date format (expected YYYY-MM-DD)"))

            if effective_date and not _parse_iso_date(effective_date):
                issues.append(ValidationIssue("ERROR", idx, "effective_date", "Invalid date format (expected YYYY-MM-DD)"))

            if title == "":
                issues.append(ValidationIssue("WARN", idx, "title", "Empty title reduces UX/search quality"))

    return issues