"""脚本：构建/重建向量索引。

用法：
    python scripts/build_index.py                  # 增量：PDF + 工单
    python scripts/build_index.py --reset          # 清空后重建
    python scripts/build_index.py --from-cache     # 不联网，用 docs_cache + hash_records 重建官方文档
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import get_settings  # noqa: E402
from src.data_loader import load_all, split_documents  # noqa: E402
from src.vector_store import get_manager  # noqa: E402


def load_cached_docs(settings) -> list[Document]:
    """从本地 Markdown 缓存重建官方文档（hash_records 里存了元数据）。"""
    if not settings.hash_file.exists():
        return []
    try:
        records = json.loads(settings.hash_file.read_text(encoding="utf-8")).get("records", {})
    except Exception as exc:
        print(f"读取 hash_records 失败：{exc}")
        return []

    docs: list[Document] = []
    missing = 0
    for url, meta in records.items():
        page_id = url.rstrip("/").split("/")[-1]
        matches = list(settings.docs_cache_dir.rglob(f"{page_id}.md"))
        if not matches:
            missing += 1
            continue
        text = matches[0].read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 80:
            continue
        docs.append(
            Document(
                page_content=f"【手册】{meta.get('manual', '')}｜【章节】{meta.get('section', '')}\n\n{text}",
                metadata={
                    "manual": meta.get("manual", ""),
                    "section": meta.get("section", ""),
                    "title": meta.get("title", ""),
                    "source": url,
                    "update_time": meta.get("update_time", ""),
                    "doc_type": "unitree_doc",
                },
            )
        )
    print(f"缓存文档：{len(docs)} 篇（缺缓存 {missing} 篇）")
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="构建本地索引")
    parser.add_argument("--reset", action="store_true", help="先清空现有索引")
    parser.add_argument("--from-cache", action="store_true", help="只用本地文档缓存重建，不联网")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    manager = get_manager()

    if args.reset:
        manager.reset()
        print("已清空旧索引。")

    total = 0
    if args.from_cache:
        cached = load_cached_docs(settings)
        chunks = split_documents(cached, settings)
        added = manager.add_documents(chunks)
        print(f"官方文档分块：{len(chunks)}，新增 {added}")
        total += added

    docs, stats = load_all(settings)
    chunks = split_documents(docs, settings)
    print(f"本地文件：{stats.summary()}")
    added = manager.add_documents(chunks)
    print(f"本地分块：{len(chunks)}，新增 {added}")
    total += added

    print(f"\n本次新增：{total}")
    print(f"当前向量数：{manager.stats()['vectors']}")
    print(f"Embedding：{settings.embedding_signature}")
    print(f"索引已保存到： {settings.faiss_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
