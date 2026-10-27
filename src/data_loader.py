"""数据加载：本地 PDF 手册 + Excel 工单，统一成带元数据的 Document。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LoadStats:
    pdf_files: int = 0
    pdf_pages: int = 0
    work_orders: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = f"PDF {self.pdf_files} 个（{self.pdf_pages} 页），工单 {self.work_orders} 条"
        if self.errors:
            text += f"，失败 {len(self.errors)} 项"
        return text


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def load_pdfs(settings: Settings, stats: LoadStats) -> list[Document]:
    docs: list[Document] = []
    if not settings.manuals_dir.exists():
        return docs
    for path in sorted(settings.manuals_dir.rglob("*.pdf")):
        try:
            text = _read_pdf(path)
        except Exception as exc:  # 单个文件失败不影响整批
            stats.errors.append(f"{path.name}: {exc}")
            logger.warning("PDF 解析失败：%s（%s）", path, exc)
            continue
        text = (text or "").strip()
        if len(text) < 50:
            continue
        stats.pdf_files += 1
        stats.pdf_pages += text.count("\n") and 1 or 1
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "manual": path.stem,
                    "section": path.stem,
                    "title": path.stem,
                    "source": str(path),
                    "update_time": "",
                    "doc_type": "pdf_manual",
                },
            )
        )
    return docs


def load_work_orders(settings: Settings, stats: LoadStats) -> list[Document]:
    import pandas as pd

    path = settings.work_orders_file
    if not path.exists():
        return []
    try:
        frame = pd.read_excel(path)
    except Exception as exc:
        stats.errors.append(f"{path.name}: {exc}")
        logger.warning("工单读取失败：%s", exc)
        return []

    docs: list[Document] = []
    for _, row in frame.iterrows():
        order_id = str(row.get("工单ID") or "").strip()
        if not order_id:
            continue
        model = str(row.get("产品型号") or "").strip()
        fault = str(row.get("故障类型") or "").strip()
        body_lines = [
            f"【工单ID】{order_id}",
            f"【产品型号】{model}",
            f"【故障类型】{fault}",
            f"【故障现象】{row.get('故障现象') or ''}",
            f"【诊断过程】{row.get('诊断过程') or ''}",
            f"【解决方案】{row.get('解决方案') or ''}",
            f"【更换配件】{row.get('更换配件') or ''}",
        ]
        docs.append(
            Document(
                page_content="\n".join(body_lines),
                metadata={
                    "manual": "历史工单库",
                    "section": f"{order_id} · {model} · {fault}",
                    "title": order_id,
                    "source": order_id,
                    "update_time": str(row.get("发生日期") or ""),
                    "doc_type": "work_order",
                    "model": model,
                    "fault_type": fault,
                },
            )
        )
        stats.work_orders += 1
    return docs


def split_documents(docs: list[Document], settings: Settings | None = None) -> list[Document]:
    """按中文友好的分隔符切分，保留全部元数据。"""
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
        keep_separator=True,
    )
    chunks: list[Document] = []
    for doc in docs:
        for piece in splitter.split_text(doc.page_content or ""):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))
    return chunks


def load_all(settings: Settings | None = None) -> tuple[list[Document], LoadStats]:
    settings = settings or get_settings()
    stats = LoadStats()
    docs = load_pdfs(settings, stats) + load_work_orders(settings, stats)
    return docs, stats
