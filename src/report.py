"""故障统计与 Excel 导出。"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["工单ID", "产品型号", "故障现象", "诊断过程", "解决方案", "更换配件"]

FAULT_COLORS = {
    "关节异响": "#4C78A8",
    "电机过热": "#F58518",
    "电池鼓包": "#E45756",
    "通信中断": "#72B7B2",
    "IMU数据漂移": "#54A24B",
}


@dataclass
class Report:
    total: int = 0
    by_fault: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_model: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_month: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_parts: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""
    insights: list[str] = field(default_factory=list)
    date_range: str = ""
    errors: list[str] = field(default_factory=list)


def _classify(frame: pd.DataFrame) -> pd.DataFrame:
    """按现象关键词归类，缺失时回退到已有列。"""
    frame = frame.copy()
    if "故障类型" not in frame.columns:
        frame["故障类型"] = ""
    keywords = [
        ("关节异响", ["异响", "咯噔", "摩擦声", "咔哒"]),
        ("电机过热", ["过热", "发烫", "高温", "温度报警"]),
        ("电池鼓包", ["鼓包", "漏液", "电池变形"]),
        ("通信中断", ["失联", "掉线", "通信", "断开", "中断"]),
        ("IMU数据漂移", ["漂移", "姿态", "陀螺"]),
    ]

    def classify(row) -> str:
        current = str(row.get("故障类型") or "").strip()
        if current:
            return current
        text = f"{row.get('故障现象', '')} {row.get('诊断过程', '')}"
        for name, words in keywords:
            if any(word in text for word in words):
                return name
        return "其他"

    frame["故障类型"] = frame.apply(classify, axis=1)
    return frame


def build_report(path: Path | None = None, settings: Settings | None = None) -> Report:
    settings = settings or get_settings()
    path = Path(path) if path else settings.work_orders_file
    report = Report()

    if not path.exists():
        report.errors.append(f"工单文件不存在：{path}")
        return report

    try:
        frame = pd.read_excel(path)
    except Exception as exc:
        report.errors.append(f"读取工单失败：{exc}")
        return report

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        report.errors.append(f"缺少必需列：{', '.join(missing)}")
        return report

    frame = _classify(frame)
    report.raw = frame
    report.total = len(frame)
    if report.total == 0:
        report.errors.append("工单表为空")
        return report

    fault_counts = frame["故障类型"].value_counts().reset_index()
    fault_counts.columns = ["故障类型", "数量"]
    fault_counts["占比"] = (fault_counts["数量"] / report.total * 100).round(1)
    report.by_fault = fault_counts

    model_counts = (
        frame["产品型号"].fillna("未填写").replace("", "未填写").value_counts().reset_index()
    )
    model_counts.columns = ["产品型号", "数量"]
    model_counts["占比"] = (model_counts["数量"] / report.total * 100).round(1)
    report.by_model = model_counts

    frame["_日期"] = (
        pd.to_datetime(frame["发生日期"], errors="coerce")
        if "发生日期" in frame.columns
        else pd.NaT
    )
    valid = frame["_日期"].dropna()
    if not valid.empty:
        report.date_range = f"{valid.min():%Y-%m-%d} ~ {valid.max():%Y-%m-%d}"
        monthly = frame.dropna(subset=["_日期"]).copy()
        monthly["月份"] = monthly["_日期"].dt.strftime("%Y-%m")
        report.by_month = monthly.groupby("月份").size().reset_index(name="工单数")

    parts: dict[str, int] = {}
    for value in frame.get("更换配件", pd.Series(dtype=str)).tolist():
        text = str(value or "").strip()
        if not text or text in {"无", "None", "nan", "—"}:
            continue
        for part in re.split(r"[、,，;；/]", text):
            part = part.strip()
            if part and part not in {"无", "nan"}:
                parts[part] = parts.get(part, 0) + 1
    if parts:
        report.top_parts = pd.DataFrame(
            sorted(parts.items(), key=lambda kv: kv[1], reverse=True)[:10],
            columns=["配件名称", "更换次数"],
        )

    top = report.by_fault.iloc[0]
    report.summary = (
        f"共 {report.total} 条工单，覆盖 {report.by_model.shape[0]} 个型号；"
        f"最高频故障为「{top['故障类型']}」（{top['数量']} 条，占 {top['占比']}%）。"
    )
    if not report.by_month.empty and len(report.by_month) >= 2:
        latest = report.by_month.iloc[-1]
        report.insights.append(
            f"{latest['月份']} 共 {int(latest['工单数'])} 条工单，可结合当月交付量判断是否为异常波动。"
        )
    if not report.top_parts.empty:
        part = report.top_parts.iloc[0]
        report.insights.append(
            f"配件消耗集中在「{part['配件名称']}」（{int(part['更换次数'])} 次），建议核查库存与备件策略。"
        )
    report.insights.append("建议把高频故障的排查步骤整理成标准话术，减少重复答疑。")
    return report


def export_report_excel(report: Report) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        report.by_fault.to_excel(writer, sheet_name="故障类型", index=False)
        report.by_model.to_excel(writer, sheet_name="产品型号", index=False)
        if not report.by_month.empty:
            report.by_month.to_excel(writer, sheet_name="月度趋势", index=False)
        if not report.top_parts.empty:
            report.top_parts.to_excel(writer, sheet_name="配件消耗", index=False)
        report.raw.drop(columns=["_日期"], errors="ignore").to_excel(
            writer, sheet_name="原始工单", index=False
        )
    return buffer.getvalue()
