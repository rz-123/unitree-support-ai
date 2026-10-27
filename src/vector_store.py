"""向量库：FAISS + 可选 Embedding 服务，带指纹校验与按来源替换。

没有任何 Key 时自动退到 512 维字符 n-gram 哈希向量，保证项目能零配置跑起来。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorStoreNotReady(RuntimeError):
    """索引不存在或需要重建。"""


# ---------------------------------------------------------------------------
# 离线兜底：字符 n-gram 哈希向量
# ---------------------------------------------------------------------------
class HashEmbeddings(Embeddings):
    """确定性哈希向量：不联网、维度固定，适合演示与回归测试。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        cleaned = re.sub(r"\s+", " ", (text or "").lower())
        for n in (1, 2, 3):
            for i in range(len(cleaned) - n + 1):
                gram = cleaned[i : i + n]
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                vector[idx] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class EmbeddingBundle:
    signature: str
    dim: int | None
    embeddings: Embeddings


class DashScopeCurlEmbeddings(Embeddings):
    """走 curl_cffi 的 DashScope Embedding。

    公司网络/代理会把普通 requests 的 TLS 指纹拦掉（握手直接 EOF），
    但浏览器指纹能过，所以这里直接调 OpenAI 兼容接口。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v2",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        batch_size: int = 10,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import time as _time

        from curl_cffi import requests as curl_requests

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"model": self.model, "input": batch, "encoding_format": "float"}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = curl_requests.post(
                        f"{self.base_url}/embeddings",
                        json=payload,
                        headers=headers,
                        impersonate="chrome",
                        timeout=60,
                    )
                    if response.status_code == 200:
                        for item in response.json().get("data", []):
                            vectors.append(list(item["embedding"]))
                        last_error = None
                        break
                    last_error = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:160]}"
                    )
                except Exception as exc:  # 超时/连接抖动，退避重试
                    last_error = exc
                _time.sleep(2 * (attempt + 1))
            if last_error is not None:
                raise RuntimeError(f"DashScope embedding 失败：{last_error}")
        if len(vectors) != len(texts):
            raise RuntimeError(f"Embedding 数量不匹配：期望 {len(texts)}，实际 {len(vectors)}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def build_embedding_bundle(settings: Settings | None = None) -> EmbeddingBundle:
    settings = settings or get_settings()
    provider = settings.embedding_provider
    if provider == "dashscope":
        embeddings = DashScopeCurlEmbeddings(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "model": settings.embedding_model,
            "api_key": settings.embedding_api_key,
        }
        if settings.embedding_base_url:
            kwargs["base_url"] = settings.embedding_base_url
        embeddings = OpenAIEmbeddings(**kwargs)
    elif provider == "local":
        try:
            import os

            # 沙箱/受限环境下系统临时目录不可写，把 TMP 与模型缓存都放到项目内
            tmp_dir = settings.data_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TMPDIR"] = str(tmp_dir)
            os.environ["TEMP"] = str(tmp_dir)
            os.environ["TMP"] = str(tmp_dir)

            from langchain_community.embeddings import FastEmbedEmbeddings

            embeddings = FastEmbedEmbeddings(
                model_name=settings.embedding_model,
                cache_dir=str(settings.data_dir / "models"),
            )
        except Exception as exc:  # 没装 fastembed / 模型下载失败时兜底
            logger.warning("本地 Embedding 不可用，退回哈希向量：%s", exc)
            embeddings = HashEmbeddings(512)
    else:
        embeddings = HashEmbeddings(512)

    return EmbeddingBundle(
        signature=settings.embedding_signature,
        dim=settings.embedding_dim_hint,
        embeddings=embeddings,
    )


# ---------------------------------------------------------------------------
# 管理器
# ---------------------------------------------------------------------------
class VectorStoreManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.bundle = build_embedding_bundle(self.settings)
        self._store: Any = None
        self._meta: dict[str, Any] = {}
        self._hashes: set[str] = set()
        self._lock = threading.RLock()
        self.warnings: list[str] = []

    # ---------------- 持久化 ----------------
    @property
    def _index_file(self) -> Path:
        return self.settings.faiss_dir / "index.faiss"

    @property
    def _meta_file(self) -> Path:
        return self.settings.faiss_dir / "index_meta.json"

    def _read_meta(self) -> dict[str, Any]:
        try:
            return json.loads(self._meta_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def load(self) -> Any:
        """读取已有索引；指纹不符时拒绝加载，避免维度错乱。"""
        with self._lock:
            if self._store is not None:
                return self._store
            if not self._index_file.exists():
                return None
            meta = self._read_meta()
            if meta.get("embedding_signature") != self.bundle.signature:
                msg = (
                    f"索引由 {meta.get('embedding_signature')} 生成，当前配置是 "
                    f"{self.bundle.signature}；请重建索引。"
                )
                self.warnings.append(msg)
                logger.warning(msg)
                return None
            try:
                from langchain_community.vectorstores import FAISS

                self._store = FAISS.load_local(
                    str(self.settings.faiss_dir),
                    self.bundle.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._meta = meta
                self._hashes = set(meta.get("hashes") or [])
                return self._store
            except Exception as exc:
                msg = f"索引加载失败：{exc}"
                self.warnings.append(msg)
                logger.warning(msg)
                return None

    @staticmethod
    def _hash_doc(doc: Document) -> str:
        source = (doc.metadata or {}).get("source", "")
        return hashlib.md5(f"{source}|{doc.page_content}".encode("utf-8")).hexdigest()

    def add_documents(self, docs: list[Document], save: bool = True) -> int:
        if not docs:
            return 0
        with self._lock:
            # 必须先加载已有索引，否则会在空 store 上新建并覆盖掉原有向量
            self.load()
            fresh = []
            for doc in docs:
                digest = self._hash_doc(doc)
                if digest in self._hashes:
                    continue
                self._hashes.add(digest)
                fresh.append(doc)
            if not fresh:
                return 0
            if self._store is None:
                from langchain_community.vectorstores import FAISS

                self._store = FAISS.from_documents(fresh, self.bundle.embeddings)
            else:
                self._store.add_documents(fresh)
            if save:
                self.save()
            return len(fresh)

    def replace_source_documents(self, source: str, docs: list[Document]) -> int:
        """先删掉该来源的旧分块，再写入新分块（避免旧手册残留）。"""
        with self._lock:
            self.delete_source(source, save=False)
            added = self.add_documents(docs, save=False)
            self.save()
            return added

    def delete_source(self, source: str, save: bool = True) -> int:
        with self._lock:
            self.load()
            if self._store is None:
                return 0
            ids = [
                key
                for key, doc in self._store.docstore._dict.items()
                if (doc.metadata or {}).get("source") == source
            ]
            if not ids:
                return 0
            self._store.delete(ids)
            self._hashes = {
                self._hash_doc(doc) for doc in self._store.docstore._dict.values()
            }
            if save:
                self.save()
            return len(ids)

    def save(self) -> None:
        with self._lock:
            if self._store is None:
                return
            self.settings.faiss_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.settings.faiss_dir))
            sources: dict[str, int] = {}
            for doc in self._store.docstore._dict.values():
                key = str((doc.metadata or {}).get("source") or "")
                sources[key] = sources.get(key, 0) + 1
            meta = {
                "embedding_signature": self.bundle.signature,
                "embedding_dim": self.bundle.dim,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "sources": sources,
                "hashes": sorted(self._hashes),
            }
            tmp = self._meta_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._meta_file)
            self._meta = meta

    def reset(self) -> None:
        with self._lock:
            self._store = None
            self._meta = {}
            self._hashes = set()
            for path in self.settings.faiss_dir.glob("*"):
                if path.name == ".gitkeep":
                    continue
                try:
                    path.unlink()
                except Exception:
                    pass

    # ---------------- 查询 ----------------
    def get_retriever(self, k: int | None = None) -> Any:
        store = self.load()
        if store is None:
            raise VectorStoreNotReady(
                "向量库为空或未建立。请先同步官网文档（数据同步页 / scripts/sync_docs.py），"
                "或把 PDF 放进 data/manuals 后执行 scripts/build_index.py。"
            )
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or self.settings.retriever_k},
        )

    def stats(self) -> dict[str, Any]:
        store = self.load()
        meta = self._meta or self._read_meta()
        return {
            "ready": store is not None,
            "vectors": int(getattr(store.index, "ntotal", 0)) if store is not None else 0,
            "embedding_signature": self.bundle.signature,
            "embedding_dim": self.bundle.dim,
            "updated_at": meta.get("updated_at", ""),
            "sources": len(meta.get("sources", {})),
            "warnings": list(dict.fromkeys(self.warnings)),
        }


# ---------------------------------------------------------------------------
# 单例门面
# ---------------------------------------------------------------------------
_manager: VectorStoreManager | None = None
_manager_lock = threading.Lock()


def get_manager(force: bool = False) -> VectorStoreManager:
    global _manager
    if force or _manager is None:
        with _manager_lock:
            if force or _manager is None:
                _manager = VectorStoreManager()
    return _manager


def reset_singleton() -> None:
    global _manager
    with _manager_lock:
        _manager = None


def get_vectorstore() -> Any:
    return get_manager().load()


def add_documents(docs: list[Document]) -> int:
    return get_manager().add_documents(docs)


def get_retriever(k: int | None = None) -> Any:
    return get_manager().get_retriever(k)
