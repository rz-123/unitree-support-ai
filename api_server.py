"""FastAPI 入口：把 RAG、同步、工单与报表暴露成 HTTP 接口。

启动：uvicorn api_server:app --host 0.0.0.0 --port 8000
文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.crawler_sync import CrawlerSync
from src.rag_chain import get_rag
from src.report import build_report, export_report_excel
from src.vector_store import get_manager
from src.work_order import WorkOrderData, create_work_order_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
settings = get_settings()

app = FastAPI(title="宇树售后 AI 助手 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryItem(BaseModel):
    role: str = "user"
    content: str = ""


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[HistoryItem] = Field(default_factory=list)
    language: str = Field(default="zh")


class SyncRequest(BaseModel):
    space: str | None = None
    limit: int | None = Field(default=None, ge=0)
    force: bool = False
    dry_run: bool = False


class WorkOrderRequest(BaseModel):
    order_id: str = ""
    model: str = ""
    symptom: str = ""
    diagnosis: str = ""
    solution: str = ""
    parts: str = ""
    engineer: str = ""
    fault_type: str = ""
    note: str = ""
    source_refs: str = ""
    duration: str = ""


@app.get("/health")
def health() -> dict[str, Any]:
    stats = get_manager().stats()
    return {
        "status": "ok",
        "llm": settings.llm_model if settings.llm_ready else "retrieval-only",
        "embedding": settings.embedding_signature,
        "vectors": stats["vectors"],
        "time": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/config")
def api_config() -> dict[str, str]:
    return settings.describe()


@app.post("/api/ask")
def api_ask(request: AskRequest) -> dict[str, Any]:
    try:
        result = get_rag().ask(
            request.question,
            [item.model_dump() for item in request.history],
            language=request.language,
        )
        return result.to_dict()
    except Exception as exc:
        logger.exception("诊断失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/sync")
def api_sync(request: SyncRequest) -> dict[str, Any]:
    try:
        report = CrawlerSync(settings).sync(
            only_space=request.space,
            limit=request.limit,
            force=request.force,
            dry_run=request.dry_run,
        )
        return report.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/work-order")
def api_work_order(request: WorkOrderRequest) -> StreamingResponse:
    try:
        pdf = create_work_order_pdf(WorkOrderData(**request.model_dump()), settings=settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    filename = f"{request.order_id or 'work_order'}.pdf"
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/report")
def api_report() -> dict[str, Any]:
    report = build_report(settings=settings)
    return {
        "total": report.total,
        "summary": report.summary,
        "date_range": report.date_range,
        "errors": report.errors,
        "by_fault": report.by_fault.to_dict(orient="records"),
        "by_model": report.by_model.to_dict(orient="records"),
        "by_month": report.by_month.to_dict(orient="records"),
        "top_parts": report.top_parts.to_dict(orient="records"),
    }


@app.get("/api/report/export")
def api_report_export() -> StreamingResponse:
    report = build_report(settings=settings)
    data = export_report_excel(report)
    filename = f"fault_report_{datetime.now():%Y%m%d}.xlsx"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
