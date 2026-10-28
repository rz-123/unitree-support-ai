"""RAG 问答链：检索 + 多轮改写 + 强制结构 + 安全兜底 + 引用核验。

实现说明（LangChain 0.3.x）：
- create_history_aware_retriever 处理“它/这个故障”这类指代；
- create_stuff_documents_chain 把检索结果塞进 Prompt；
- create_retrieval_chain 串成 LCEL 链。
- 单轮跳过改写，省一次大模型往返。

容错：没有 Key / 调用失败 / 知识库为空，都会退化成「仅检索」或明确提示，不崩。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

from .config import Settings, get_settings
from .vector_store import get_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是宇树科技（Unitree）售后技术支持助手，服务对象是售后工程师、代理商和终端客户。

【回答规则】
1. 先判断机型。H1 与 H1-2 的底层接口、故障码、关节电机顺序是两套独立内容，绝不能混用。
   如果用户没有说机型、且资料也无法唯一确定，必须在答案末尾用「【需确认】」请用户补充机型。
2. 只把「参考资料」里的内容当作事实依据。资料没写到的，一个字都不能补：
   不得编造扭矩/电压/电流/温度等数值，不得编造配件编号、轴承型号、螺栓规格、
   ROS 话题与命令、故障码、接口引脚、手册章节名。资料没给出的具体参数，就写「资料未给出」。
3. 每条事实性陈述后面标注出处，格式为：【出处】手册名 › 章节名。
   引用历史工单写成：【出处】历史工单库 › 工单ID。只能引用本次资料中真实出现的条目。
   【可能原因】的每一条都必须以「手册有据」或「经验推断」开头。
4. 安全优先：涉及冒烟、焦味、外壳变形、电池鼓包/漏液、关节异常发热或卡死、跌倒失控时，
   必须在答案最前面单独写一段「⚠️ 安全提示」，先讲断电、隔离、停止操作，再讲排查。
5. 排查步骤要可执行：说清楚看哪个接口/话题/字段、怎么判断是 A 还是 B。
6. 涉及「能不能互换 / 兼不兼容」时，资料没有明确结论就必须回答「知识库中没有跨型号兼容性的官方依据，不能确认」，
   只给验证方法，不得替用户下结论，也不得用“工单里没提不兼容”去反推兼容。
7. 排查到头、资料确实没覆盖时，直接说「手册未覆盖，建议转人工技术支持」。
8. 回答要短：整篇控制在 700 字以内，【排查步骤】不超过 6 条。

【输出格式】（严格使用以下五个小节）
【故障现象】一句话复述问题。
【可能原因】按可能性排序，每条带【出处】。
【排查步骤】
1. 第一步（可执行动作）
2. 第二步
【解决方案】处理方案；需要换配件的写清名称。
【参考来源】本次实际引用的手册/工单，格式：手册名 › 章节名。

某一节不适用时写“不适用”，但不要删除小节标题。"""

SYSTEM_PROMPT_EN = """You are the Unitree after-sales technical support assistant.

[RULES]
1. Identify the model first. H1 and H1-2 have independent interfaces, fault codes and joint orders.
   If the model is missing, end with "[Need model]" asking for it.
2. Use only the REFERENCES as facts. Never invent torque/voltage/current values, part numbers,
   ROS topics/commands, fault codes or section names; say "not stated in the references" instead.
3. Cite every factual statement as [Source] Manual › Section (or Work order › ID). Each item under
   [Possible Causes] must start with "manual-backed" or "inferred".
4. Safety first: for smoke, burning smell, battery swelling, joint overheating/jamming or falls,
   start with a "⚠️ Safety notice" paragraph.
5. Steps must be actionable: which interface/topic/field to read, how to tell A from B.
6. For compatibility questions without an explicit conclusion, answer
   "no official basis in the knowledge base, cannot confirm" and give verification steps only.
7. If not covered, say "not covered by the manual; escalate to technical support".
8. Be brief: under 350 words, at most 6 steps.

[OUTPUT FORMAT] (exactly these five sections, in English)
[Symptom] one sentence.
[Possible Causes] ordered, each with a [Source].
[Troubleshooting]
1. first actionable step
[Solution] the fix.
[References] cited manuals/work orders."""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定对话历史和用户的最新问题，把最新问题改写成一个不依赖上下文、可以独立检索的完整问题。"
            "保留机型、故障码、接口名等关键信息。只输出改写后的问题本身。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "以下是知识库检索到的参考资料：\n\n{context}\n\n用户问题：{input}\n\n"
            "请严格按系统提示的五个小节输出。",
        ),
    ]
)

QA_PROMPT_EN = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_EN),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "References retrieved from the knowledge base:\n\n{context}\n\n"
            "User question: {input}\n\nAnswer strictly in the five sections.",
        ),
    ]
)

DOCUMENT_PROMPT = ChatPromptTemplate.from_template(
    "[资料] 手册：{manual}｜章节：{section}｜更新时间：{update_time}\n{page_content}"
)


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------
@dataclass
class SourceSnippet:
    title: str
    manual: str
    section: str
    source: str
    snippet: str
    doc_type: str = "unknown"

    def label(self) -> str:
        return f"{self.manual} › {self.section}"

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "manual": self.manual,
            "section": self.section,
            "source": self.source,
            "snippet": self.snippet,
            "doc_type": self.doc_type,
        }


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[SourceSnippet] = field(default_factory=list)
    mode: str = "llm"
    elapsed: float = 0.0
    safety_notice: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "elapsed": round(self.elapsed, 2),
            "safety_notice": self.safety_notice,
            "error": self.error,
            "sources": [item.to_dict() for item in self.sources],
        }


# ---------------------------------------------------------------------------
# 安全规则（代码层兜底，不依赖模型是否听话）
# ---------------------------------------------------------------------------
SAFETY_RULES: list[dict[str, str]] = [
    {
        "pattern": r"冒烟|焦味|糊味|烧焦|起火|明火|smoke|burning|fire",
        "zh_label": "冒烟/焦味", "zh_advice": "立即断电，停止一切操作，远离易燃物；不要复位或重启，联系官方售后。",
        "en_label": "smoke / burning smell", "en_advice": "Power off immediately and keep away from flammables; do not reset or reboot.",
    },
    {
        "pattern": r"鼓包|漏液|电解液|电池变形|swollen|swelling|leak",
        "zh_label": "电池鼓包/漏液", "zh_advice": "按危险品处理：不要充电、不要挤压或拆解电池，将整机移至通风空旷处并联系售后。",
        "en_label": "battery swelling / leakage", "en_advice": "Treat as hazardous: do not charge, crush or disassemble; move to a ventilated area.",
    },
    {
        "pattern": r"发烫|烫手|过热|高温|overheat|too hot",
        "zh_label": "异常发热", "zh_advice": "停止该关节或电机的调试指令，等待自然冷却后再检查，避免二次损伤。",
        "en_label": "abnormal heat", "en_advice": "Stop commands to that joint or motor and let it cool before inspection.",
    },
    {
        "pattern": r"跌倒|摔倒|失控|乱动|急停|撞人|fall|fell|uncontrolled",
        "zh_label": "跌倒/失控", "zh_advice": "先隔离人员、确认急停可用，再排查；不要在有人靠近时继续上电测试。",
        "en_label": "fall / loss of control", "en_advice": "Clear people, confirm the e-stop works, then troubleshoot.",
    },
    {
        "pattern": r"卡死|抱死|别住|转不动|jammed|seized|stuck",
        "zh_label": "关节卡死", "zh_advice": "禁止继续下发运动指令，避免减速器/电机进一步损坏，先手动检查机械干涉。",
        "en_label": "joint jam", "en_advice": "Do not send motion commands; check mechanical interference by hand.",
    },
    {
        "pattern": r"异响|咔哒|摩擦声|啸叫|abnormal noise|clicking|grinding",
        "zh_label": "关节异响", "zh_advice": "停止该关节的高负载动作，记录出现异响的角度、速度与负载，再按排查步骤检查。",
        "en_label": "joint noise", "en_advice": "Stop high-load motion, record angle/speed/load, then follow the steps.",
    },
]

MODEL_PATTERNS: list[tuple[str, str]] = [
    ("H1-2", r"H1[\s\-_－]?2"),
    ("G1", r"(?<![A-Za-z0-9])G1(?:-D)?(?![A-Za-z0-9])"),
    ("H2", r"(?<![A-Za-z0-9])H2(?![A-Za-z0-9])"),
    ("R1", r"(?<![A-Za-z0-9])R1(?:-A)?(?![A-Za-z0-9])"),
    ("Go2", r"Go\s?2(?:-W)?"),
    ("B2", r"(?<![A-Za-z0-9])B2(?![A-Za-z0-9])"),
    ("A2", r"(?<![A-Za-z0-9])A2(?:W)?(?![A-Za-z0-9])"),
    ("As2", r"As\s?2"),
    ("H1", r"(?<![A-Za-z0-9\-])H1(?![A-Za-z0-9])"),
]

FAULT_HINTS = (
    "故障", "报错", "报警", "异常", "不动", "停机", "掉线", "断开", "漂移",
    "过热", "异响", "抖动", "无法", "失灵", "死机", "重启",
)

QUERY_SYNONYMS: dict[str, str] = {
    "故障码": "错误码 错误代码 故障代码",
    "错误码": "故障码 错误代码",
    "报错": "错误 错误码 异常",
    "异响": "噪声 响声 摩擦声",
    "掉线": "失联 断开 通信中断",
    "固件": "固件升级 firmware",
    "遥控器": "遥控 remote",
}

EN_GLOSSARY: dict[str, str] = {
    "fault code": "故障码 错误码",
    "error code": "错误码 故障码",
    "battery": "电池",
    "charging": "充电",
    "joint": "关节",
    "motor": "电机",
    "remote": "遥控器",
    "firmware": "固件",
    "imu": "IMU 姿态",
    "lidar": "激光雷达",
    "calibration": "标定",
    "temperature": "温度",
    "overheat": "过热",
    "noise": "异响 噪声",
    "communication": "通信",
    "disconnect": "断开 掉线",
    "interface": "接口",
}

FAULT_TERMS: dict[str, tuple[str, ...]] = {
    "关节异响": ("异响", "咔哒", "摩擦声", "啸叫", "咯噔"),
    "电机过热": ("过热", "过温", "发烫", "温度过高", "高温"),
    "电池鼓包": ("鼓包", "漏液", "电池变形", "电解液"),
    "通信中断": ("通信", "失联", "掉线", "断开", "中断", "超时", "丢包"),
    "IMU数据漂移": ("imu", "漂移", "姿态", "陀螺", "加速度计"),
}


def expand_query(question: str) -> str:
    """检索前补同义词与英文术语的中文对应，不改变给模型看的问题。"""
    extras = [value for key, value in QUERY_SYNONYMS.items() if key in question]
    lowered = question.lower()
    extras.extend(zh for en, zh in EN_GLOSSARY.items() if en in lowered)
    if not extras:
        return question
    return f"{question} {' '.join(dict.fromkeys(extras))}"


def detect_safety(text: str, lang: str = "zh") -> str:
    english = str(lang).lower().startswith("en")
    hits: list[tuple[str, str]] = []
    for rule in SAFETY_RULES:
        if re.search(rule["pattern"], text, flags=re.IGNORECASE):
            hits.append(
                (rule["en_label"], rule["en_advice"])
                if english
                else (rule["zh_label"], rule["zh_advice"])
            )
    if not hits:
        return ""
    if english:
        labels = ", ".join(label for label, _ in hits)
        advice = "\n".join(f"- {label}: {text_}" for label, text_ in hits)
        return (
            "> ⚠️ **Safety notice (safety first, then troubleshooting)**\n"
            f"> The description involves: **{labels}**.\n>\n"
            f"> {advice.replace(chr(10), chr(10) + '> ')}\n>\n"
            "> For qualified engineers only; does not replace on-site judgement."
        )
    labels = "、".join(label for label, _ in hits)
    advice = "\n".join(f"- {label}：{text_}" for label, text_ in hits)
    return (
        f"> ⚠️ **安全提示（先处理安全，再排查故障）**\n"
        f"> 检测到问题描述涉及：**{labels}**。\n>\n"
        f"> {advice.replace(chr(10), chr(10) + '> ')}\n>\n"
        f"> 以下技术内容仅供具备资质的工程师参考，不替代现场安全判断。"
    )


def detect_models(text: str) -> list[str]:
    found: list[str] = []
    for model, pattern in MODEL_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(model)
    if "H1-2" in found and "H1" in found:
        found.remove("H1")
    return found


def _looks_like_fault(question: str) -> bool:
    return any(hint in question for hint in FAULT_HINTS)


def _canon_units(text: str) -> str:
    text = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*N\s*[·.．]?\s*m(?![a-zA-Z])", r"\1Nm", text)
    text = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*(?:°\s*C|℃)", r"\1C", text)
    text = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*M\s*Ω", r"\1MOHM", text)
    return text


def _doc_mentions_model(doc: Document, models: list[str]) -> bool:
    meta = doc.metadata or {}
    haystack = " ".join(
        [
            str(meta.get("model", "")),
            str(meta.get("manual", "")),
            str(meta.get("section", "")),
            str(meta.get("title", "")),
            doc.page_content or "",
        ]
    )
    for model in models:
        if model == "H1":
            pattern = r"H1(?![\s\-_－]?2)(?![A-Za-z0-9])"
        else:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(model)}(?![A-Za-z0-9])"
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return True
    return False


def _doc_matches_fault(doc: Document, question: str) -> bool:
    meta = doc.metadata or {}
    content = f"{meta.get('section', '')} {meta.get('title', '')} {doc.page_content or ''}".lower()
    for terms in FAULT_TERMS.values():
        if any(term in question.lower() for term in terms) and any(
            term in content for term in terms
        ):
            return True
    return False


class ModelAwareRetriever(BaseRetriever):
    """先多召回，再把「命中目标机型 + 同类故障」的文档排前面，最后整页回填。"""

    base: Any
    k: int = 4
    page_index: dict[str, list[str]] = {}

    @staticmethod
    def _expand_page(doc: Document, page_index: dict[str, list[str]]) -> Document:
        source = doc.metadata.get("source") or ""
        parts = page_index.get(source) or []
        if not parts or len(parts) > 8:
            return doc
        joined = "\n\n".join(part for part in parts if part).strip()
        if not joined or len(joined) > 2500 or len(joined) <= len(doc.page_content or ""):
            return doc
        return Document(page_content=joined, metadata=dict(doc.metadata))

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        query = expand_query(query)
        docs = list(self.base.invoke(query))
        models = detect_models(query)
        if docs:
            docs = sorted(
                docs,
                key=lambda doc: (
                    0 if models and _doc_mentions_model(doc, models) else 1,
                    0 if _doc_matches_fault(doc, query) else 1,
                ),
            )
        expanded: list[Document] = []
        seen: set[str] = set()
        for doc in docs:
            key = doc.metadata.get("source") or str(id(doc))
            if key in seen:
                continue
            seen.add(key)
            expanded.append(self._expand_page(doc, self.page_index))
        return expanded[: self.k]


class RobotSupportRAG:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._chains_by_lang: dict[str, Any] = {}
        self._qa_by_lang: dict[str, Any] = {}
        self._retriever: Any = None
        self._lock = threading.Lock()
        self._llm: Any = None

    # ---------------- 构建 ----------------
    def _build_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        if not self.settings.llm_ready:
            return None
        try:
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model": self.settings.llm_model,
                "api_key": self.settings.llm_api_key,
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
                "timeout": 45,
                "max_retries": 1,
            }
            if self.settings.llm_base_url:
                kwargs["base_url"] = self.settings.llm_base_url
            self._llm = ChatOpenAI(**kwargs)
        except Exception as exc:
            logger.error("大模型初始化失败，降级为仅检索：%s", exc)
            self._llm = None
        return self._llm

    def _get_retriever(self) -> Any:
        if self._retriever is not None:
            return self._retriever
        with self._lock:
            if self._retriever is None:
                fetch_k = max(self.settings.retriever_k * 8, 32)
                base = get_manager().get_retriever(fetch_k)
                self._retriever = ModelAwareRetriever(
                    base=base,
                    k=self.settings.retriever_k,
                    page_index=self._build_page_index(),
                )
        return self._retriever

    @staticmethod
    def _build_page_index() -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        try:
            store = get_manager().load()
            if store is None:
                return index
            for doc in store.docstore._dict.values():
                source = (doc.metadata or {}).get("source") or ""
                if source:
                    index.setdefault(str(source), []).append(doc.page_content or "")
        except Exception as exc:
            logger.warning("构建整页索引失败：%s", exc)
        return index

    @staticmethod
    def _chains():
        try:
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains.history_aware_retriever import create_history_aware_retriever
            from langchain.chains.retrieval import create_retrieval_chain
        except ImportError:
            # LangChain 1.x：这些构造器迁到了 langchain-classic，且只从子模块导出
            from langchain_classic.chains.combine_documents import (  # type: ignore
                create_stuff_documents_chain,
            )
            from langchain_classic.chains.history_aware_retriever import (  # type: ignore
                create_history_aware_retriever,
            )
            from langchain_classic.chains.retrieval import (  # type: ignore
                create_retrieval_chain,
            )
        return create_history_aware_retriever, create_retrieval_chain, create_stuff_documents_chain

    def _build_chain(self, retriever: Any, language: str = "zh") -> Any:
        lang = "en" if str(language).lower().startswith("en") else "zh"
        if lang in self._chains_by_lang:
            return self._chains_by_lang[lang]
        with self._lock:
            if lang in self._chains_by_lang:
                return self._chains_by_lang[lang]
            llm = self._build_llm()
            if llm is None:
                return None
            try:
                (
                    create_history_aware_retriever,
                    create_retrieval_chain,
                    create_stuff_documents_chain,
                ) = self._chains()
                history_retriever = create_history_aware_retriever(llm, retriever, CONDENSE_PROMPT)
                qa_chain = create_stuff_documents_chain(
                    llm,
                    QA_PROMPT_EN if lang == "en" else QA_PROMPT,
                    document_prompt=DOCUMENT_PROMPT,
                    document_separator="\n\n",
                )
                self._qa_by_lang[lang] = qa_chain
                self._chains_by_lang[lang] = create_retrieval_chain(history_retriever, qa_chain)
            except Exception as exc:
                logger.error("RAG 链构建失败：%s", exc)
                self._chains_by_lang[lang] = None
            return self._chains_by_lang[lang]

    # ---------------- 工具 ----------------
    @staticmethod
    def _to_messages(history: Sequence[Any] | None, window: int = 6) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for item in list(history or [])[-window:]:
            try:
                if isinstance(item, BaseMessage):
                    messages.append(item)
                elif isinstance(item, dict):
                    role = str(item.get("role") or "user")
                    content = str(item.get("content") or "")
                    if not content:
                        continue
                    messages.append(
                        AIMessage(content=content)
                        if role in {"ai", "assistant", "bot"}
                        else HumanMessage(content=content)
                    )
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    messages.append(HumanMessage(content=str(item[0])))
                    messages.append(AIMessage(content=str(item[1])))
            except Exception:
                continue
        return messages

    @staticmethod
    def _to_sources(docs: Sequence[Document]) -> list[SourceSnippet]:
        sources: list[SourceSnippet] = []
        seen: set[tuple[str, str, str]] = set()
        for doc in docs:
            meta = doc.metadata or {}
            manual = str(meta.get("manual") or "未标注手册")
            section = str(meta.get("section") or "未标注章节")
            source = str(meta.get("source") or "")
            key = (manual, section, source)
            if key in seen:
                continue
            seen.add(key)
            snippet = re.sub(r"\s+", " ", doc.page_content or "").strip()
            sources.append(
                SourceSnippet(
                    title=str(meta.get("title") or section),
                    manual=manual,
                    section=section,
                    source=source,
                    snippet=snippet[:220],
                    doc_type=str(meta.get("doc_type") or "unknown"),
                )
            )
        return sources

    @staticmethod
    def _sources_block(sources: Sequence[SourceSnippet], lang: str = "zh") -> str:
        english = str(lang).lower().startswith("en")
        if not sources:
            return "- No usable references retrieved" if english else "- 本次未检索到可用资料"
        return "\n".join(
            f"- {item.manual} › {item.section}{f' ({item.source})' if item.source else ''}"
            for item in sources
        )

    def _retrieval_only_answer(
        self, question: str, docs: Sequence[Document], reason: str, lang: str = "zh"
    ) -> str:
        english = str(lang).lower().startswith("en")
        sources = self._to_sources(docs)
        if not docs:
            if english:
                return (
                    f"[Symptom] {question}\n"
                    "[Possible Causes] No relevant material found.\n"
                    "[Troubleshooting]\n1. Provide model, firmware version, fault code and reproduction steps.\n"
                    "[Solution] Insufficient references; escalate to technical support.\n"
                    "[References] none"
                )
            return (
                f"【故障现象】{question}\n"
                "【可能原因】知识库中没有检索到相关资料，暂时无法判断。\n"
                "【排查步骤】\n1. 请补充机型、软件版本、故障码与复现步骤。\n"
                "2. 或先在「数据同步」页导入对应机型手册。\n"
                "【解决方案】资料不足，建议转人工技术支持。\n"
                "【参考来源】无"
            )

        excerpts = []
        for index, doc in enumerate(docs[:4], start=1):
            text = re.sub(r"\s+", " ", doc.page_content or "").strip()[:400]
            manual = doc.metadata.get("manual", "未标注手册")
            section = doc.metadata.get("section", "未标注章节")
            excerpts.append(f"{index}. ({manual} › {section})\n{text}")
        excerpt = "\n\n".join(excerpts)
        if english:
            return (
                f"[Symptom] {question}\n"
                f"[Possible Causes] {reason}. Raw excerpts:\n\n{excerpt}\n\n"
                "[Troubleshooting]\n1. Check the model and interfaces against the excerpts.\n"
                "2. Rule out items in order and record the fault code.\n"
                "[Solution] Follow the original text; escalate if still unclear.\n"
                f"[References]\n{self._sources_block(sources, lang)}"
            )
        return (
            f"【故障现象】{question}\n"
            f"【可能原因】{reason}。以下是与问题最相关的资料原文，未做归纳：\n\n{excerpt}\n\n"
            "【排查步骤】\n1. 先按上方资料中的章节核对机型与接口定义。\n"
            "2. 再按资料给出的顺序逐项排除，并记录故障码。\n"
            "【解决方案】请以资料原文为准；若仍无法定位，建议转人工技术支持。\n"
            f"【参考来源】\n{self._sources_block(sources, lang)}"
        )

    @staticmethod
    def _verify_grounding(answer: str, docs: Sequence[Document], lang: str = "zh") -> str:
        """核对答案里的硬事实是否真的出现在检索资料中（只提示，不改写）。"""
        context_parts = []
        for doc in docs:
            meta = doc.metadata or {}
            header = " ".join(
                str(meta.get(key) or "") for key in ("manual", "section", "title", "source")
            )
            context_parts.append(f"{header} {doc.page_content or ''}")
        context_norm = _canon_units("\n".join(context_parts))
        answer_norm = _canon_units(answer)

        patterns = [
            (r"WO-\d{6}-\d{3}", "工单号"),
            (r"(?<![A-Za-z0-9])[A-Z]{2,5}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*", "配件/型号代码"),
            (r"\d+(?:\.\d+)?Nm|\d+(?:\.\d+)?MOHM|\d+(?:\.\d+)?C", "具体数值"),
        ]
        problems: list[str] = []
        for pattern, label in patterns:
            for token in dict.fromkeys(re.findall(pattern, answer_norm)):
                if token in context_norm:
                    continue
                if label == "配件/型号代码" and re.fullmatch(r"WO-\d{6}-\d{3}", token):
                    continue
                if str(lang).lower().startswith("en"):
                    en_label = {
                        "工单号": "work order ID",
                        "配件/型号代码": "part/model code",
                        "具体数值": "value",
                    }.get(label, label)
                    problems.append(f"{en_label}: {token}")
                else:
                    problems.append(f"{label}：{token}")

        if not problems:
            return ""
        lines = "\n".join(f"> - {item}" for item in problems[:8])
        if str(lang).lower().startswith("en"):
            return (
                "> ⚠️ **Citation check failed**: these values are not in the retrieved references:\n"
                f"{lines}"
            )
        return (
            "> ⚠️ **引用核验未通过**：下列内容没有出现在本次检索资料中，"
            "可能是模型自行补充的，请人工确认后再使用：\n" + lines
        )

    # ---------------- 对外 ----------------
    def ask(
        self,
        question: str,
        history: Sequence[Any] | None = None,
        language: str = "zh",
    ) -> AnswerResult:
        started = time.time()
        lang = "en" if str(language).lower().startswith("en") else "zh"
        question = (question or "").strip()
        if not question:
            return AnswerResult(
                question="",
                answer="Please describe the symptom first." if lang == "en" else "请先输入故障现象或想咨询的问题。",
                mode="error",
                error="empty question",
            )

        safety_notice = detect_safety(question, lang)
        models = detect_models(question)
        need_model = _looks_like_fault(question) and not models

        try:
            retriever = self._get_retriever()
        except Exception as exc:
            if lang == "en":
                answer = (
                    "The knowledge base is not ready.\n\n"
                    "1. Open Data Sync and sync official docs;\n"
                    "2. Put PDFs into data/manuals and run scripts/build_index.py.\n\n"
                    f"(Details: {exc})"
                )
            else:
                answer = (
                    "知识库还没有建立，暂时无法检索。\n\n"
                    "请任选一种方式初始化：\n"
                    "1. 打开「数据同步」页，点击「一键同步官网最新文档」；\n"
                    "2. 把 PDF 手册放进 data/manuals，执行 `python scripts/build_index.py`；\n"
                    "3. 执行 `python scripts/init_mock_data.py` 生成模拟工单后再建索引。\n\n"
                    f"（技术细节：{exc}）"
                )
            return AnswerResult(
                question=question,
                answer=self._with_notices(answer, safety_notice, need_model, lang),
                mode="error",
                elapsed=time.time() - started,
                safety_notice=safety_notice,
                error=str(exc),
            )

        if not self.settings.llm_ready:
            docs = retriever.invoke(question)
            reason = (
                "No LLM API key configured, switched to retrieval-only mode"
                if lang == "en"
                else "当前未配置大模型 API Key，已切换为「仅检索」模式"
            )
            answer = self._retrieval_only_answer(question, docs, reason, lang)
            return AnswerResult(
                question=question,
                answer=self._with_notices(answer, safety_notice, need_model, lang),
                sources=self._to_sources(docs),
                mode="retrieval_only",
                elapsed=time.time() - started,
                safety_notice=safety_notice,
            )

        chain = self._build_chain(retriever, lang)
        if chain is not None:
            try:
                messages = self._to_messages(history)
                if messages:
                    result = chain.invoke({"input": question, "chat_history": messages})
                    docs = result.get("context") or []
                    answer = str(result.get("answer") or "").strip()
                else:
                    docs = retriever.invoke(question)
                    answer = str(
                        self._qa_by_lang[lang].invoke(
                            {"context": docs, "input": question, "chat_history": []}
                        )
                        or ""
                    ).strip()
                if not answer:
                    raise ValueError("empty answer from model")
                warning = self._verify_grounding(answer, docs, lang)
                if warning:
                    answer = f"{answer}\n\n{warning}"
                return AnswerResult(
                    question=question,
                    answer=self._with_notices(answer, safety_notice, need_model, lang),
                    sources=self._to_sources(docs),
                    mode="llm",
                    elapsed=time.time() - started,
                    safety_notice=safety_notice,
                )
            except Exception as exc:
                logger.exception("RAG 调用失败，降级为仅检索")
                error_text = f"{type(exc).__name__}: {exc}"

        docs = retriever.invoke(question)
        reason = (
            "大模型调用失败，已自动降级为「仅检索」模式"
            if lang == "zh"
            else "LLM call failed, switched to retrieval-only mode"
        )
        answer = self._retrieval_only_answer(question, docs, reason, lang)
        return AnswerResult(
            question=question,
            answer=self._with_notices(answer, safety_notice, need_model, lang),
            sources=self._to_sources(docs),
            mode="retrieval_only",
            elapsed=time.time() - started,
            safety_notice=safety_notice,
            error=error_text if "error_text" in locals() else "",
        )

    # ---------------- 流式问答 ----------------
    def ask_stream(
        self,
        question: str,
        history: Sequence[Any] | None = None,
        language: str = "zh",
    ):
        """逐段产出事件，供前端做流式渲染。

        事件：("meta", {...}) / ("sources", [...]) / ("token", str) / ("done", AnswerResult)
        """
        started = time.time()
        lang = "en" if str(language).lower().startswith("en") else "zh"
        question = (question or "").strip()
        if not question:
            return

        safety_notice = detect_safety(question, lang)
        models = detect_models(question)
        need_model = _looks_like_fault(question) and not models
        yield "meta", {"lang": lang}
        if safety_notice:
            yield "token", safety_notice + "\n\n---\n\n"

        try:
            retriever = self._get_retriever()
        except Exception as exc:
            answer = (
                "知识库还没有建立，暂时无法检索。请先在「数据同步」页同步官网文档，"
                f"或把 PDF 放进 data/manuals 后重建索引。（{exc}）"
                if lang == "zh"
                else f"Knowledge base is not ready: {exc}"
            )
            yield "token", answer
            yield "done", AnswerResult(
                question=question,
                answer=self._with_notices(answer, safety_notice, need_model, lang),
                mode="error",
                elapsed=time.time() - started,
                safety_notice=safety_notice,
                error=str(exc),
            )
            return

        messages = self._to_messages(history)
        search_question = question
        llm = self._build_llm()
        if messages and llm is not None:
            try:
                rewritten = llm.invoke(
                    CONDENSE_PROMPT.format_messages(chat_history=messages, input=question)
                )
                text = str(getattr(rewritten, "content", "") or "").strip()
                if text:
                    search_question = text
            except Exception as exc:
                logger.warning("多轮问题改写失败，按原问题检索：%s", exc)

        docs = retriever.invoke(search_question)
        sources = self._to_sources(docs)
        yield "sources", sources

        if not self.settings.llm_ready or llm is None:
            reason = (
                "当前未配置大模型 API Key，已切换为「仅检索」模式"
                if lang == "zh"
                else "No LLM API key configured, retrieval-only mode"
            )
            answer = self._retrieval_only_answer(question, docs, reason, lang)
            yield "token", answer
            yield "done", AnswerResult(
                question=question,
                answer=self._with_notices(answer, safety_notice, need_model, lang),
                sources=sources,
                mode="retrieval_only",
                elapsed=time.time() - started,
                safety_notice=safety_notice,
            )
            return

        self._build_chain(retriever, lang)
        qa_chain = self._qa_by_lang.get(lang)
        if qa_chain is None:
            answer = self._retrieval_only_answer(
                question,
                docs,
                "大模型调用失败，已自动降级为「仅检索」模式" if lang == "zh" else "LLM unavailable",
                lang,
            )
            yield "token", answer
            yield "done", AnswerResult(
                question=question,
                answer=self._with_notices(answer, safety_notice, need_model, lang),
                sources=sources,
                mode="retrieval_only",
                elapsed=time.time() - started,
                safety_notice=safety_notice,
            )
            return

        parts: list[str] = []
        try:
            for chunk in qa_chain.stream(
                {"context": docs, "input": question, "chat_history": messages}
            ):
                text = chunk if isinstance(chunk, str) else str(getattr(chunk, "content", "") or "")
                if text:
                    parts.append(text)
                    yield "token", text
        except Exception as exc:
            logger.exception("流式生成失败，整段重试")
            try:
                full = qa_chain.invoke(
                    {"context": docs, "input": question, "chat_history": messages}
                )
                text = str(full or "")
                parts = [text]
                yield "token", text
            except Exception as exc2:
                answer = self._retrieval_only_answer(
                    question,
                    docs,
                    "大模型调用异常，已自动降级为「仅检索」模式" if lang == "zh" else "LLM failed",
                    lang,
                )
                yield "token", answer
                yield "done", AnswerResult(
                    question=question,
                    answer=self._with_notices(answer, safety_notice, need_model, lang),
                    sources=sources,
                    mode="retrieval_only",
                    elapsed=time.time() - started,
                    safety_notice=safety_notice,
                    error=f"{type(exc).__name__}: {exc}; {type(exc2).__name__}: {exc2}",
                )
                return

        answer = "".join(parts).strip()
        warning = self._verify_grounding(answer, docs, lang)
        if warning:
            answer = f"{answer}\n\n{warning}"
            yield "token", "\n\n" + warning

        final = self._with_notices(answer, safety_notice, need_model, lang)
        if need_model:
            yield "token", "\n\n" + final.split("\n\n---\n\n")[-1]

        yield "done", AnswerResult(
            question=question,
            answer=final,
            sources=sources,
            mode="llm",
            elapsed=time.time() - started,
            safety_notice=safety_notice,
        )

    @staticmethod
    def _with_notices(
        answer: str, safety_notice: str, need_model: bool, language: str = "zh"
    ) -> str:
        parts: list[str] = []
        if safety_notice:
            parts.append(safety_notice)
        parts.append(answer.strip())
        if need_model:
            if str(language).lower().startswith("en"):
                parts.append(
                    "[Need model] Please specify the model (H1 / H1-2 / G1 / G1-D / H2 / R1 / "
                    "R1-A / Go2 / B2 / A2 / As2), firmware version and fault code."
                )
            else:
                parts.append(
                    "【需确认】请补充具体机型（H1 / H1-2 / G1 / G1-D / H2 / R1 / R1-A / "
                    "Go2 / B2 / A2 / As2），以及软件版本和故障码。"
                    "宇树不同机型的接口与故障码不通用，确认后才能给出准确排查路径。"
                )
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_rag: RobotSupportRAG | None = None
_rag_lock = threading.Lock()


def get_rag(force: bool = False) -> RobotSupportRAG:
    global _rag
    if force or _rag is None:
        with _rag_lock:
            if force or _rag is None:
                _rag = RobotSupportRAG()
    return _rag


def answer_question(
    question: str, history: Sequence[Any] | None = None, language: str = "zh"
) -> AnswerResult:
    return get_rag().ask(question, history, language=language)


def reset_rag() -> None:
    global _rag
    with _rag_lock:
        _rag = None
