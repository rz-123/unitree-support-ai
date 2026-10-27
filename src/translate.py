"""用大模型做中英对照翻译：一次调用翻译整篇，按段落返回等长数组。"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是机器人售后技术文档的专职译者。把用户给出的每一段中文译成英文。

要求：
1. 逐段翻译，段落数量与顺序必须和原文完全一致，不得合并或拆分；
2. 保留所有标记与代码原样：如【故障现象】、【出处】、`rt/lowstate`、`0x1A`、H1-2、Go2、WO-202501-001；
   小节标题按固定译法：故障现象→Symptom，可能原因→Possible Causes，排查步骤→Troubleshooting，
   解决方案→Solution，参考来源→References；
3. 术语用宇树官方英文文档的常见写法：关节 joint、减速器 reducer、故障码 fault code、电池 battery、
   遥控器 remote controller、标定 calibration、灵巧手 dexterous hand；
4. 只输出 JSON 数组，形如 ["译文1", "译文2"]，不要输出任何解释或代码块标记。"""


def split_blocks(text: str) -> list[str]:
    blocks = [block.strip("\n") for block in re.split(r"\n\s*\n", text or "")]
    return [block for block in blocks if block.strip()]


@lru_cache(maxsize=4)
def _client(signature: str) -> Any:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": 0.0,
        "max_tokens": 2600,
        "timeout": 45,
        "max_retries": 1,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**kwargs)


def _loads_array(raw: str) -> list[str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def translate_blocks(blocks: list[str], target: str = "en") -> list[str]:
    blocks = [block for block in blocks if block.strip()]
    if not blocks:
        return []

    settings: Settings = get_settings()
    if not settings.llm_ready:
        return []

    target_name = "English" if target == "en" else "Chinese"
    numbered = "\n\n".join(f"[{i}]\n{block}" for i, block in enumerate(blocks, start=1))
    human = (
        f"请把下面 {len(blocks)} 段内容译成{target_name}，按 [1] [2] 的顺序返回 JSON 数组，"
        f"数组长度必须是 {len(blocks)}。\n\n{numbered}"
    )
    try:
        client = _client(settings.embedding_signature + "|" + settings.llm_model)
        message = client.invoke([("system", SYSTEM_PROMPT), ("human", human)])
        pairs = _loads_array(str(getattr(message, "content", "")))
    except Exception as exc:
        logger.warning("翻译调用失败：%s", exc)
        return []

    if len(pairs) != len(blocks):
        logger.warning("翻译段落数不匹配：期望 %s，实际 %s", len(blocks), len(pairs))
        return []
    return pairs
