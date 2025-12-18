# 让类型注解变成“延迟解析“，从而更省心、更少前向引用/循环导入坑
from __future__ import annotations

from dataclasses import dataclass
# 用一种更安全、更跨平台、更“面向对象“的方式来处理文件路径和文件系统操作
from pathlib import Path
import os

# dataclass 是 Python 供的一个“自动生成样板代码“的装饰器，适合主要用于装数据的类
# 会自动生成__init__, __repr__, __eq__等函数
# frozen=True 的意思是：这个 dataclass 创建出来后，字段不允许被修改
@dataclass(frozen=True)
class Settings:
    repo_root: Path
    docs_csv_path: Path
    parsed_dir: Path
    index_dir: Path

    # Embedding
    embedding_model: str
    chroma_collection: str

    @staticmethod
    def from_repo_root(repo_root: Path | None = None) -> Settings:
        # Python常见写法：若 repo_root 不是 None 且为真值，用它；否则，用 Path.cwd()
        root = repo_root or Path.cwd()
        return Settings(
            repo_root=root,
            docs_csv_path=root / "data" / "metadata" / "docs.csv",
            parsed_dir=root / "data" / "parsed",
            index_dir= root / "data" / "index",
            # 优先从环境变量中读取配置；如果没配环境变量，就用默认值
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "policy-chunks"),
        )