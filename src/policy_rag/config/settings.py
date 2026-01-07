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

    # Evidence Gate
    evidence_top1_max_dist: float # top-1（排名第一的chunk）的距离不能超过这个上限
    evidence_good_hit_max_dist: float # 将检索结果中 distance ≤ 该阈值的 chunk 视为“高相关/好证据“
    evidence_min_good_hits: int # 至少要有多少条“好证据“
    evidence_min_gap: float # 用median(distance) - top1_distance 衡量区分度，top1 必须“明显优于整体“
    
    # llm
    llm_provider: str
    ollama_base_url: str # Ollama服务的地址
    ollama_model: str # 调用的具体模型名
    # 采样温度，控制输出的“随机性/发散程度“
    # 温度越低：越保守、越稳定，越像“按规矩填表
    # 温度越高：更有创造性，但更容易跑偏
    ollama_temperature: float 
    ollama_num_predict: int # 本次最多生成多少token

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

            # Evidence Gate Defaults(distance 越小越相关)
            evidence_top1_max_dist=float(os.getenv("EVIDENCE_TOP1_MAX_DIST", "0.95")),
            evidence_good_hit_max_dist=float(os.getenv("EVIDENCE_GOOD_HIT_MAX_DIST", "1.05")),
            evidence_min_good_hits=int(os.getenv("EVIDENCE_MIN_GOOD_HITS", "2")),
            evidence_min_gap=float(os.getenv("EVIDENCE_MIN_GAP", "0.03")),

            # LLM(Ollama)
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://0.0.0.0:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
            ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
            ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "4800")),
        )