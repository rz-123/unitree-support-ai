"""脚本：命令行同步宇树官网文档。

用法：
    python scripts/sync_docs.py                 # 全量增量同步
    python scripts/sync_docs.py --space H1_developer --limit 5 --dry-run
    python scripts/sync_docs.py --force         # 忽略 MD5 重抓
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import get_settings  # noqa: E402
from src.crawler_sync import CrawlerSync  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="同步宇树官网文档到向量库")
    parser.add_argument("--space", default=None, help="只同步指定 space，如 H1_developer")
    parser.add_argument("--limit", type=int, default=None, help="每个手册最多处理页数")
    parser.add_argument("--force", action="store_true", help="忽略 MD5 强制重抓")
    parser.add_argument("--dry-run", action="store_true", help="只检查不写入")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    print(f"监控配置：{settings.urls_file}")
    print(f"向量库：  {settings.faiss_dir}")
    print(f"Embedding：{settings.embedding_signature}")

    report = CrawlerSync(settings).sync(
        only_space=args.space,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print(report.summary())
    if report.failed:
        print("\n失败明细：")
        for item in report.items:
            if item.status == "failed":
                print(f"  - {item.title}: {item.message}")
    return 0 if report.failed == 0 and not report.error else 2


if __name__ == "__main__":
    sys.exit(main())
