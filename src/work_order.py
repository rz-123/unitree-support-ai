"""维修工单 PDF：fpdf2 + 自动查找中文字体。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from .config import Settings, get_settings


class ChineseFontUnavailable(RuntimeError):
    """找不到可用中文字体。"""


CANDIDATE_FONTS = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


@dataclass
class WorkOrderData:
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


def get_chinese_font(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    if settings.pdf_font_path:
        path = Path(settings.pdf_font_path)
        if path.exists():
            return path
    if settings.fonts_dir.exists():
        for path in sorted(settings.fonts_dir.iterdir()):
            if path.suffix.lower() in {".ttf", ".ttc", ".otf"}:
                return path
    for candidate in CANDIDATE_FONTS:
        if Path(candidate).exists():
            return Path(candidate)
    raise ChineseFontUnavailable(
        "未找到中文字体。请把 ttf/ttc 放到 data/fonts/，或设置 PDF_FONT_PATH。"
    )


def _clean(text: str) -> str:
    """PDF 字体缺字会让 fpdf 抛错：替换常见符号并去掉非 BMP 字符（emoji）。"""
    text = (text or "").replace("›", ">").replace("→", "->").replace("←", "<-")
    text = text.replace("·", "·").replace("\u00a0", " ")
    text = "".join(ch for ch in text if ord(ch) <= 0xFFFF)
    return text


def _section(pdf: FPDF, title: str, content: str) -> None:
    pdf.set_font(pdf.font_family, "B", 11)
    pdf.set_text_color(31, 78, 121)
    pdf.cell(0, 7, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)
    pdf.set_font(pdf.font_family, "", 10)
    pdf.multi_cell(0, 5.5, _clean(content or "（未填写）"))
    pdf.ln(2)


def create_work_order_pdf(data: WorkOrderData | dict, settings: Settings | None = None) -> bytes:
    settings = settings or get_settings()
    if isinstance(data, dict):
        data = WorkOrderData(**data)

    font_path = get_chinese_font(settings)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("CJK", "", str(font_path))
    # 只有一个字重时也注册为粗体，避免 set_font(..., "B") 报 Undefined font
    pdf.add_font("CJK", "B", str(font_path))
    pdf.set_font("CJK", "", 10)

    pdf.set_font("CJK", "B", 18)
    pdf.set_text_color(31, 56, 100)
    pdf.cell(0, 12, "宇树机器人维修工单", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CJK", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_draw_color(200, 210, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("CJK", "", 10)
    meta = [
        ("工单编号", data.order_id),
        ("产品型号", data.model),
        ("故障类型", data.fault_type),
        ("处理工程师", data.engineer),
        ("处理时长", f"{data.duration} 小时" if data.duration else ""),
    ]
    for index, (key, value) in enumerate(meta):
        if index % 2 == 0:
            pdf.set_font("CJK", "", 9)
            pdf.set_text_color(120, 120, 120)
        pdf.cell(28, 7, _clean(key), border=0)
        pdf.set_font("CJK", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(62, 7, _clean(value or "—"), border=0)
        if index % 2 == 1:
            pdf.ln(7)
    pdf.ln(3)

    _section(pdf, "一、故障现象", data.symptom)
    _section(pdf, "二、诊断过程", data.diagnosis)
    _section(pdf, "三、解决方案", data.solution)
    _section(pdf, "四、更换配件", data.parts)
    _section(pdf, "五、参考来源", data.source_refs)
    _section(pdf, "六、备注", data.note)

    pdf.ln(4)
    pdf.set_font("CJK", "", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(
        0,
        4.5,
        "免责声明：本工单由 AI 辅助生成，内容需由持证工程师现场核实后归档；"
        "涉及安全责任判定、保修范围的问题，以宇树官方售后流程为准。",
    )

    pdf.set_auto_page_break(False)
    pdf.set_y(-14)
    pdf.set_font("CJK", "", 8)
    pdf.cell(0, 6, f"第 {pdf.page_no()}/{pdf.pages_count} 页", align="C")

    output = pdf.output()
    if output is None:
        raise RuntimeError("PDF 生成失败")
    return bytes(output)


_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "故障现象": ("故障现象", "Symptom"),
    "可能原因": ("可能原因", "Possible Causes", "Possible Cause"),
    "排查步骤": ("排查步骤", "Troubleshooting", "Steps"),
    "解决方案": ("解决方案", "Solution"),
    "参考来源": ("参考来源", "References", "Sources"),
}
_ALL_NAMES = "|".join(re.escape(alias) for aliases in _SECTION_ALIASES.values() for alias in aliases)
_SECTION_END = rf"(?=^【(?:{_ALL_NAMES})】|\Z)"


def parse_answer_sections(answer: str) -> dict[str, str]:
    """抽取五段式答案；中英文小节名都支持。"""
    found: dict[str, str] = {}
    for canonical, aliases in _SECTION_ALIASES.items():
        for name in aliases:
            match = re.search(
                rf"^【{re.escape(name)}】[ \t]*\n?(?P<body>.*?){_SECTION_END}",
                answer or "",
                flags=re.S | re.M,
            )
            if match:
                found[canonical] = match.group("body").strip()
                break

    diagnosis = "\n".join(
        part for part in (found.get("可能原因", ""), found.get("排查步骤", "")) if part
    )
    return {
        "symptom": found.get("故障现象", ""),
        "diagnosis": diagnosis,
        "solution": found.get("解决方案", ""),
        "references": found.get("参考来源", ""),
    }


def safe_filename(order_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", order_id or "work_order").strip("_")
    return f"{cleaned or 'work_order'}.pdf"


def save_work_order_pdf(data, path: Path, settings: Settings | None = None) -> Path:
    content = create_work_order_pdf(data, settings=settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path
