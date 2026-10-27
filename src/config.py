"""配置层：把 .env 解析成不可变的 Settings 单例。

设计原则：
- 所有路径都从项目根目录推导，换机器不用改代码；
- 服务商按 Key 自动选择，缺 Key 时给出可运行的降级方案；
- 配置只读，运行期通过 get_settings() 取同一个实例。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=False)

# ---------------------------------------------------------------------------
# 服务商预设：都走 OpenAI 兼容协议，换服务商只需要换 base_url + 模型名
# ---------------------------------------------------------------------------
LLM_PRESETS: dict[str, dict[str, str]] = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
}

EMBEDDING_PRESETS: dict[str, dict[str, str]] = {
    "dashscope": {"model": "text-embedding-v2", "key_env": "DASHSCOPE_API_KEY"},
    "openai": {"model": "text-embedding-3-small", "key_env": "OPENAI_API_KEY"},
    "local": {"model": "BAAI/bge-small-zh-v1.5", "key_env": ""},
    "hash": {"model": "hash-ngram-v1", "key_env": ""},
}


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else str(value).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _pick_llm_provider() -> str:
    explicit = _env("LLM_PROVIDER", "auto").lower()
    if explicit in LLM_PRESETS:
        return explicit
    if _env("DASHSCOPE_API_KEY"):
        return "dashscope"
    if _env("DEEPSEEK_API_KEY"):
        return "deepseek"
    if _env("OPENAI_API_KEY"):
        return "openai"
    return "dashscope"


def _pick_embedding_provider() -> str:
    explicit = _env("EMBEDDING_PROVIDER", "auto").lower()
    if explicit in EMBEDDING_PRESETS:
        return explicit
    if _env("DASHSCOPE_API_KEY"):
        return "dashscope"
    if _env("OPENAI_API_KEY"):
        return "openai"
    # 没有云端 Key 时优先用本地模型（fastembed），装不上再退哈希
    try:
        import fastembed  # noqa: F401

        return "local"
    except Exception:
        return "hash"


@dataclass(frozen=True)
class Settings:
    """不可变配置对象，进程内单例。"""

    base_dir: Path
    data_dir: Path
    manuals_dir: Path
    docs_cache_dir: Path
    vectorstore_dir: Path
    faiss_dir: Path
    urls_file: Path
    hash_file: Path
    work_orders_file: Path
    fonts_dir: Path

    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float
    llm_max_tokens: int

    embedding_provider: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str

    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    request_timeout: int
    crawl_delay: float
    crawl_retries: int
    user_agent: str
    pdf_font_path: str
    app_title: str

    # ---------------- 派生属性 ----------------
    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key and self.llm_model and self.llm_base_url)

    @property
    def embedding_signature(self) -> str:
        return f"{self.embedding_provider}:{self.embedding_model}"

    @property
    def embedding_dim_hint(self) -> int | None:
        hints = {
            "text-embedding-v2": 1536,
            "text-embedding-v3": 1024,
            "text-embedding-v4": 1024,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "BAAI/bge-small-zh-v1.5": 512,
            "hash-ngram-v1": 512,
        }
        return hints.get(self.embedding_model)

    @staticmethod
    def masked(value: str) -> str:
        if not value:
            return "（未配置）"
        if len(value) <= 8:
            return "****"
        return f"{value[:4]}...{value[-4:]}"

    def describe(self) -> dict[str, str]:
        state = "已配置" if self.llm_ready else "未配置（仅检索模式）"
        return {
            "大模型服务": self.llm_provider,
            "大模型": self.llm_model,
            "大模型 Key": self.masked(self.llm_api_key),
            "大模型状态": state,
            "Embedding 服务": self.embedding_provider,
            "Embedding 模型": self.embedding_model,
            "Embedding Key": self.masked(self.embedding_api_key),
            "向量库目录": str(self.faiss_dir),
            "切分参数": f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}",
            "检索条数": str(self.retriever_k),
            "抓取间隔": f"{self.crawl_delay} 秒",
            "监控来源": str(self.urls_file),
        }

    def warnings(self) -> list[str]:
        """启动自检：只提示，不阻塞。"""
        issues: list[str] = []
        if not self.llm_ready:
            issues.append("未配置大模型 Key：诊断会返回「仅检索」结果（手册原文 + 出处）。")
        if self.embedding_provider == "hash":
            issues.append(
                "未配置 Embedding Key：当前使用离线哈希向量，只能按字面匹配，语义检索效果有限。"
            )
        if not self.work_orders_file.exists():
            issues.append(f"工单文件不存在：{self.work_orders_file}（可运行 init_mock_data.py 生成）")
        if not self.urls_file.exists():
            issues.append(f"监控配置不存在：{self.urls_file}")
        return issues

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.manuals_dir,
            self.docs_cache_dir,
            self.vectorstore_dir,
            self.faiss_dir,
            self.fonts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    llm_provider = _pick_llm_provider()
    llm_preset = LLM_PRESETS[llm_provider]
    embedding_provider = _pick_embedding_provider()
    embedding_preset = EMBEDDING_PRESETS[embedding_provider]

    data_dir = BASE_DIR / "data"
    vectorstore_dir = BASE_DIR / "vectorstore"

    settings = Settings(
        base_dir=BASE_DIR,
        data_dir=data_dir,
        manuals_dir=data_dir / "manuals",
        docs_cache_dir=data_dir / "docs_cache",
        vectorstore_dir=vectorstore_dir,
        faiss_dir=vectorstore_dir / "faiss_index",
        urls_file=data_dir / "urls.json",
        hash_file=data_dir / "hash_records.json",
        work_orders_file=data_dir / "work_orders.xlsx",
        fonts_dir=data_dir / "fonts",
        llm_provider=llm_provider,
        llm_model=_env("LLM_MODEL", llm_preset["model"]),
        llm_base_url=_env("LLM_BASE_URL", llm_preset["base_url"]).rstrip("/"),
        llm_api_key=_env("LLM_API_KEY") or _env(llm_preset["key_env"]),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 3000),
        embedding_provider=embedding_provider,
        embedding_model=_env("EMBEDDING_MODEL", embedding_preset["model"]),
        embedding_base_url=_env("EMBEDDING_BASE_URL").rstrip("/"),
        embedding_api_key=_env("EMBEDDING_API_KEY") or _env(embedding_preset["key_env"]),
        chunk_size=_env_int("CHUNK_SIZE", 600),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 80),
        retriever_k=_env_int("RETRIEVER_K", 4),
        request_timeout=_env_int("REQUEST_TIMEOUT", 30),
        crawl_delay=_env_float("CRAWL_DELAY", 2.0),
        crawl_retries=_env_int("CRAWL_RETRIES", 3),
        user_agent=_env(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        ),
        pdf_font_path=_env("PDF_FONT_PATH"),
        app_title=_env("APP_TITLE", "宇树机器人售后 AI 助手"),
    )
    settings.ensure_dirs()
    return settings
