"""Streamlit 前端：故障诊断 / 数据同步 / 故障报告 / 系统设置（中英双语）。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import get_settings  # noqa: E402
from src.crawler_sync import CrawlerSync, load_url_specs  # noqa: E402
from src.data_loader import load_all, split_documents  # noqa: E402
from src.i18n import CONFIG_LABELS, LANG_LABELS, normalize_lang, tr  # noqa: E402
from src.rag_chain import detect_models, get_rag  # noqa: E402
from src.report import build_report, export_report_excel  # noqa: E402
from src.translate import split_blocks, translate_blocks  # noqa: E402
from src.vector_store import get_manager  # noqa: E402
from src.work_order import (  # noqa: E402
    ChineseFontUnavailable,
    WorkOrderData,
    create_work_order_pdf,
    parse_answer_sections,
    safe_filename,
)

settings = get_settings()

st.set_page_config(
    page_title=settings.app_title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 隐藏 Streamlit 自带的英文菜单 / Deploy 按钮
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden; height: 0;}
footer {visibility: hidden;}
[data-testid="stChatInput"] textarea {font-size: 15px;}
[data-testid="stCaptionContainer"] p {font-size: 0.92rem; color: #8b95a5;}
</style>
""",
    unsafe_allow_html=True,
)

PAGE_KEYS = ("diagnosis", "sync", "report", "settings")
PAGE_ICONS = {"diagnosis": "🔧", "sync": "🔄", "report": "📊", "settings": "⚙️"}


def lang() -> str:
    return normalize_lang(st.session_state.get("lang", "zh"))


def L(key: str, **kwargs) -> str:
    return tr(lang(), key, **kwargs)


def page_label(key: str) -> str:
    return f"{PAGE_ICONS[key]} {L(f'nav.{key}')}"


@st.cache_resource(show_spinner=False)
def cached_rag(embedding_signature: str, llm_model: str):
    return get_rag(force=True)


def invalidate_rag_cache() -> None:
    from src import rag_chain

    rag_chain.reset_rag()
    st.cache_resource.clear()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("wo_pdf", None)
    st.session_state.setdefault("wo_name", "work_order.pdf")
    st.session_state.setdefault("sync_report", None)
    st.session_state.setdefault("lang", "zh")
    st.session_state.setdefault("nav", "diagnosis")
    st.session_state.setdefault("bilingual_auto", False)


def set_language(choice: str) -> None:
    st.session_state.lang = "en" if str(choice).startswith("En") else "zh"


def go_settings() -> None:
    st.session_state.nav = "settings"


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.wo_pdf = None


def render_topbar() -> None:
    left, right = st.columns([6, 4], vertical_alignment="center")
    with left:
        st.caption("")
    with right:
        col_lang, col_bi, col_set, col_clear = st.columns(
            [1.5, 1.2, 0.9, 1.1], vertical_alignment="center"
        )
        with col_lang:
            current = LANG_LABELS[lang()]
            choice = st.segmented_control(
                L("top.lang"),
                options=list(LANG_LABELS.values()),
                default=current,
                key=f"lang_switch_{current}",
                label_visibility="collapsed",
            )
            if choice and choice != current:
                set_language(choice)
                st.rerun()
        with col_bi:
            st.toggle(L("top.bilingual"), key="bilingual_auto", help="开启后新回答自动生成中英对照")
        with col_set:
            st.button(f"⚙️ {L('top.settings')}", on_click=go_settings, width="stretch", key="top_settings")
        with col_clear:
            st.button(
                f"🧹 {L('top.clear')}",
                on_click=clear_chat,
                width="stretch",
                key="top_clear",
                disabled=not st.session_state.messages,
            )


def render_sidebar() -> str:
    with st.sidebar:
        st.title(L("app.title"))
        st.caption(L("app.subtitle"))
        page = st.radio(
            L("nav.label"),
            options=list(PAGE_KEYS),
            format_func=page_label,
            key="nav",
            label_visibility="collapsed",
        )
        st.divider()
        try:
            stats = get_manager().stats()
            st.metric(L("side.vectors"), stats["vectors"])
            st.caption(f"{L('side.embedding')}：{stats['embedding_signature']}")
        except Exception as exc:
            st.warning(f"{L('side.stat_error')}：{exc}")

        if settings.llm_ready:
            st.success(f"{L('side.llm_ok')}：{settings.llm_model}")
        else:
            st.info(L("side.no_llm"))
        st.divider()
        st.caption(L("side.tip"))
    return page


def _source_field(source, key: str, default: str = "") -> str:
    if isinstance(source, dict):
        return str(source.get(key) or default)
    return str(getattr(source, key, default) or default)


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"📚 {L('src.expander')}（{len(sources)}）", expanded=False):
        for index, source in enumerate(sources, start=1):
            manual = _source_field(source, "manual", L("src.unknown_manual"))
            section = _source_field(source, "section", L("src.unknown_section"))
            snippet = _source_field(source, "snippet")
            link = _source_field(source, "source")
            st.markdown(f"**{index}. {manual} › {section}**")
            if snippet:
                st.caption(snippet)
            if link.startswith("http"):
                st.markdown(f"[{L('src.open')}]({link})")
            elif link:
                st.caption(f"{L('src.link')}：{link}")


def translate_message(message: dict) -> None:
    if not settings.llm_ready:
        st.warning(L("tr.no_llm"))
        return
    blocks = split_blocks(message.get("content") or "")
    if not blocks:
        return
    target = "zh" if lang() == "en" else "en"
    with st.spinner(L("tr.loading")):
        translated = translate_blocks(blocks, target=target)
    if not translated:
        st.warning(L("tr.failed"))
        return
    message["pairs"] = list(zip(blocks, translated))
    message["show_bilingual"] = True
    st.rerun()


def render_messages() -> None:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            pairs = message.get("pairs") or []
            if message["role"] == "assistant" and pairs and message.get("show_bilingual", True):
                for original, translated in pairs:
                    st.markdown(original)
                    st.caption(translated)
            else:
                st.markdown(message["content"])

            if message["role"] == "assistant":
                if pairs:
                    label = L("tr.hide") if message.get("show_bilingual", True) else L("tr.show")
                    if st.button(label, key=f"tr_toggle_{index}"):
                        message["show_bilingual"] = not message.get("show_bilingual", True)
                        st.rerun()
                elif st.button(L("tr.show"), key=f"tr_make_{index}"):
                    translate_message(message)

            if message.get("meta"):
                st.caption(message["meta"])
            render_sources(message.get("sources") or [])


def page_diagnosis() -> None:
    st.title(L("diag.title"))
    st.caption(L("diag.caption"))

    warnings = settings.warnings()
    if warnings:
        with st.expander(f"⚠️ {L('diag.selfcheck')}（{len(warnings)}）", expanded=False):
            for item in warnings:
                st.warning(item)

    render_messages()

    question = st.chat_input(L("diag.placeholder"))
    if question:
        handle_question(question)

    render_work_order_panel()


def handle_question(text: str) -> None:
    text = text.strip()
    if not text:
        st.warning(L("diag.empty"))
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    result = None
    answer_text = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("…")
        try:
            rag = cached_rag(settings.embedding_signature, settings.llm_model)
            for kind, payload in rag.ask_stream(text, history, language=lang()):
                if kind == "token":
                    answer_text += payload
                    placeholder.markdown(answer_text + " ▌")
                elif kind == "done":
                    result = payload
        except Exception as exc:
            placeholder.empty()
            st.error(f"{L('diag.fail')}：{type(exc).__name__}: {exc}")
        if answer_text:
            placeholder.markdown(answer_text)

    if result is not None:
        mode_text = {
            "llm": f"{L('diag.mode_llm')} · {result.elapsed:.1f}s",
            "retrieval_only": f"{L('diag.mode_retrieval')} · {result.elapsed:.1f}s",
            "error": L("diag.mode_error"),
        }.get(result.mode, result.mode)

        message = {
            "role": "assistant",
            "content": result.answer,
            "sources": [source.to_dict() for source in result.sources],
            "meta": mode_text,
        }
        if st.session_state.get("bilingual_auto") and result.answer:
            blocks = split_blocks(result.answer)
            target = "zh" if lang() == "en" else "en"
            with st.spinner(L("tr.loading")):
                translated = translate_blocks(blocks, target=target)
            if translated:
                message["pairs"] = list(zip(blocks, translated))
                message["show_bilingual"] = True
        st.session_state.last_result = result
        st.session_state.messages.append(message)
        st.session_state.wo_pdf = None
    st.rerun()


def render_work_order_panel() -> None:
    result = st.session_state.get("last_result")
    if result is None:
        return

    st.divider()
    st.subheader(L("wo.title"))
    parsed = parse_answer_sections(result.answer)
    models = detect_models(result.question)

    with st.expander(L("wo.expander"), expanded=False):
        with st.form("work_order_form"):
            col1, col2, col3 = st.columns(3)
            order_id = col1.text_input(L("wo.order_id"), value=f"WO-{datetime.now():%Y%m%d-%H%M}")
            model = col2.text_input(L("wo.model"), value=models[0] if models else "", placeholder=L("wo.model_ph"))
            engineer = col3.text_input(L("wo.engineer"), value="")
            col4, col5 = st.columns(2)
            fault_type = col4.text_input(L("wo.fault_type"), value="")
            duration = col5.text_input(L("wo.duration"), value="")
            parts = st.text_input(L("wo.parts"), value="", placeholder=L("wo.parts_ph"))
            note = st.text_area(L("wo.note"), height=70)
            submitted = st.form_submit_button(L("wo.generate"), type="primary")

        if submitted:
            try:
                pdf_bytes = create_work_order_pdf(
                    WorkOrderData(
                        order_id=order_id,
                        model=model,
                        symptom=parsed.get("symptom") or result.question,
                        diagnosis=parsed.get("diagnosis") or result.answer,
                        solution=parsed.get("solution") or "",
                        parts=parts,
                        engineer=engineer,
                        fault_type=fault_type,
                        note=note,
                        source_refs=parsed.get("references") or "",
                        duration=duration,
                    )
                )
                st.session_state.wo_pdf = pdf_bytes
                st.session_state.wo_name = safe_filename(order_id)
                st.success(L("wo.done"))
            except ChineseFontUnavailable as exc:
                st.error(f"{L('wo.nofont')}：{exc}")
            except Exception as exc:
                st.error(f"{L('wo.fail')}：{type(exc).__name__}: {exc}")

        if st.session_state.get("wo_pdf"):
            st.download_button(
                L("wo.download"),
                data=st.session_state.wo_pdf,
                file_name=st.session_state.wo_name,
                mime="application/pdf",
                width="stretch",
            )


def page_sync() -> None:
    st.title(L("sync.title"))
    st.caption(L("sync.caption"))

    try:
        specs = load_url_specs(settings.urls_file)
    except Exception as exc:
        st.error(f"{L('sync.empty')}（{exc}）")
        specs = []

    if specs:
        frame = pd.DataFrame(
            [
                {
                    L("sync.col_enabled"): "✅" if spec.enabled else "—",
                    L("sync.col_name"): spec.name,
                    L("sync.col_type"): spec.type,
                    L("sync.col_space"): spec.space or spec.url,
                    L("sync.col_note"): spec.note,
                }
                for spec in specs
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.info(L("sync.empty"))

    records = {}
    if settings.hash_file.exists():
        try:
            records = json.loads(settings.hash_file.read_text(encoding="utf-8")).get("records", {})
        except Exception:
            records = {}
    checked = [item.get("checked_at", "") for item in records.values() if item.get("checked_at")]
    last_checked = max(checked) if checked else "—"

    col1, col2, col3 = st.columns(3)
    col1.metric(L("sync.pages"), len(records))
    col2.metric(L("sync.last"), last_checked.replace("T", " "))
    col3.metric(L("sync.vectors"), get_manager().stats()["vectors"])

    with st.form("sync_form"):
        col1, col2, col3 = st.columns(3)
        limit = col1.number_input(
            L("sync.limit"), min_value=0, max_value=500, value=0, step=5, help=L("sync.limit_help")
        )
        force = col2.checkbox(L("sync.force"), value=False)
        dry_run = col3.checkbox(L("sync.dry"), value=False)
        run = st.form_submit_button(L("sync.run"), type="primary")

    if run:
        with st.status(L("sync.running"), expanded=True) as status:
            try:
                report = CrawlerSync(settings).sync(
                    limit=int(limit) or None, force=bool(force), dry_run=bool(dry_run)
                )
                st.session_state.sync_report = report
                status.update(label=report.summary(), state="complete" if not report.error else "error")
            except Exception as exc:
                st.session_state.sync_report = None
                status.update(label=f"Sync error: {exc}", state="error")

    report = st.session_state.get("sync_report")
    if report is not None:
        st.divider()
        st.subheader(L("sync.result"))
        if report.error:
            st.error(report.error)
        st.write(report.summary())
        if report.items:
            table = pd.DataFrame([item.to_dict() for item in report.items])
            table[L("col.status")] = table["status"].map(
                lambda key: L(f"status.{key}") if L(f"status.{key}") != f"status.{key}" else key
            )
            table = table.rename(
                columns={
                    "title": L("col.page"),
                    "message": L("col.message"),
                    "chunks": L("col.chunks"),
                    "status": L("col.status"),
                }
            )
            st.dataframe(
                table[[L("col.status"), L("col.page"), L("col.message"), L("col.chunks")]],
                hide_index=True,
                width="stretch",
            )
        with st.expander(L("sync.logs"), expanded=False):
            st.code("\n".join(report.logs) or L("sync.nolog"), language="text")

    with st.expander(L("sync.help_title"), expanded=False):
        st.markdown(L("sync.help_body"))


@st.cache_data(ttl=30, show_spinner=False)
def load_report_cached(mtime: float):
    return build_report(settings=settings)


def _localize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "故障类型": L("col.fault"),
            "数量": L("col.count"),
            "产品型号": L("col.model"),
            "占比": L("col.ratio"),
            "月份": L("col.month"),
            "工单数": L("col.orders"),
            "配件名称": L("col.part"),
            "更换次数": L("col.times"),
        }
    )


def page_report() -> None:
    st.title(L("rep.title"))
    st.caption(L("rep.caption"))

    if not settings.work_orders_file.exists():
        st.warning(L("rep.no_data"))
        if st.button(L("rep.gen_mock"), type="primary"):
            from scripts.init_mock_data import build_rows

            pd.DataFrame(build_rows(50)).to_excel(settings.work_orders_file, index=False)
            st.cache_data.clear()
            st.success(L("rep.gen_ok"))
            st.rerun()

    mtime = settings.work_orders_file.stat().st_mtime if settings.work_orders_file.exists() else 0.0
    report = load_report_cached(mtime)
    if report.total == 0:
        for error in report.errors:
            st.error(error)
        st.info(L("rep.empty"))
        return

    top = report.by_fault.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(L("rep.total"), report.total)
    col2.metric(L("rep.models"), report.by_model.shape[0])
    col3.metric(L("rep.top_fault"), str(top["故障类型"]), f"{top['占比']}%")
    col4.metric(L("rep.range"), report.date_range or L("rep.unknown"))

    if lang() == "zh":
        st.info(report.summary)
        for insight in report.insights:
            st.markdown(f"- {insight}")

    left, right = st.columns(2)
    with left:
        st.subheader(L("rep.dist_fault"))
        st.bar_chart(_localize_frame(report.by_fault).set_index(L("col.fault"))[L("col.count")], height=280)
        st.dataframe(_localize_frame(report.by_fault), hide_index=True, width="stretch")
    with right:
        st.subheader(L("rep.dist_model"))
        st.bar_chart(_localize_frame(report.by_model).set_index(L("col.model"))[L("col.count")], height=280)
        st.dataframe(_localize_frame(report.by_model), hide_index=True, width="stretch")

    if not report.by_month.empty:
        st.subheader(L("rep.trend"))
        month_frame = _localize_frame(report.by_month)
        st.line_chart(month_frame.set_index(L("col.month"))[L("col.orders")], height=260)

    if not report.top_parts.empty:
        st.subheader(L("rep.parts"))
        st.dataframe(_localize_frame(report.top_parts), hide_index=True, width="stretch")

    with st.expander(L("rep.raw"), expanded=False):
        st.dataframe(report.raw.drop(columns=["_日期"], errors="ignore"), hide_index=True, width="stretch")

    st.download_button(
        L("rep.export"),
        data=export_report_excel(report),
        file_name=f"fault_report_{datetime.now():%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def page_settings() -> None:
    st.title(L("set.title"))

    st.subheader(L("set.config"))
    labels = CONFIG_LABELS.get(lang(), {})
    rows = [(labels.get(key, key), labels.get(str(value), value)) for key, value in settings.describe().items()]
    st.table(pd.DataFrame(rows, columns=[L("set.item"), L("set.value")]))

    warnings = settings.warnings()
    if warnings:
        st.subheader(L("set.selfcheck"))
        for item in warnings:
            st.warning(item)

    st.subheader(L("set.vstore"))
    stats = get_manager().stats()
    col1, col2, col3 = st.columns(3)
    col1.metric(L("set.vectors"), stats["vectors"])
    col2.metric(L("set.sources"), stats["sources"])
    col3.metric(L("set.updated"), str(stats["updated_at"]).replace("T", " "))
    st.code(json.dumps(stats, ensure_ascii=False, indent=2), language="json")

    st.divider()
    st.subheader(L("set.maintain"))
    col1, col2 = st.columns(2)
    with col1:
        if st.button(L("set.rebuild"), width="stretch"):
            with st.spinner("..."):
                try:
                    docs, load_stats = load_all(settings=settings)
                    chunks = split_documents(docs, settings)
                    added = get_manager().add_documents(chunks) if chunks else 0
                    get_manager().save()
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")
                else:
                    invalidate_rag_cache()
                    st.toast(f"{load_stats.summary()}; +{added}")
                    st.rerun()
    with col2:
        confirm = st.checkbox(L("set.confirm"), value=False)
        if st.button(L("set.clear"), disabled=not confirm, width="stretch"):
            get_manager().reset()
            invalidate_rag_cache()
            st.toast("OK")
            st.rerun()

    st.divider()
    st.subheader(L("set.env"))
    env_path = settings.base_dir / ".env"
    st.code(
        f"BASE_DIR: {settings.base_dir}\n"
        f".env: {env_path} ({L('env.exists') if env_path.exists() else L('env.missing')})\n"
        f"data: {settings.data_dir}\n"
        f"vstore: {settings.faiss_dir}\n"
        f"embedding: {settings.embedding_signature}\n",
        language="text",
    )

    if st.button(L("set.reload")):
        from src import rag_chain, vector_store

        get_settings.cache_clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        vector_store.reset_singleton()
        rag_chain.reset_rag()
        st.rerun()

    st.divider()
    st.caption(L("set.footer"))


def main() -> None:
    init_state()
    render_topbar()
    page = render_sidebar()
    if page == "diagnosis":
        page_diagnosis()
    elif page == "sync":
        page_sync()
    elif page == "report":
        page_report()
    else:
        page_settings()


if __name__ == "__main__":
    main()
