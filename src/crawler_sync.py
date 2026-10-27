"""宇树官网文档的抓取与增量同步。

站点结构（已实测）：
- https://support.unitree.com 是 Vue 单页应用，直接抓 HTML 只有空壳；
- 手册清单  https://robot-api.unitree.com/doc/spaces2?locale=zh
- 手册目录  https://robot-api.unitree.com/doc/?space=<space>&locale=zh
- 每页正文  doc-cdn.unitree.com 上的 Markdown
- 目录接口前有腾讯 EdgeOne 校验，部分出口返回 HTTP 567；此时用 curl_cffi
  模拟浏览器 TLS 指纹重试，仍失败再退回文本代理。

增量逻辑：正文 MD5 比对 -> 有变化才切分 -> 先删该来源旧分块再写新分块。
"""

from __future__ import annotations

import hashlib
import html as html_module
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlparse

import requests
from langchain_core.documents import Document

from .config import Settings, get_settings
from .data_loader import split_documents
from .vector_store import VectorStoreManager, get_manager

logger = logging.getLogger(__name__)

UNITREE_API_BASE = "https://robot-api.unitree.com"
UNITREE_TOC_URL = f"{UNITREE_API_BASE}/doc/?space={{space}}&locale=zh"
UNITREE_PROXY_PREFIX = "https://r.jina.ai/"

STATUS_NEW = "new"
STATUS_UPDATED = "updated"
STATUS_UNCHANGED = "unchanged"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_PLANNED = "planned"


class CrawlError(RuntimeError):
    pass


class AntiBotError(CrawlError):
    pass


@dataclass
class UrlSpec:
    name: str
    type: str = "html"
    url: str = ""
    space: str = ""
    enabled: bool = True
    note: str = ""


@dataclass
class PageTarget:
    url: str
    title: str
    section: str
    manual: str
    update_time: str = ""
    space: str = ""


@dataclass
class SyncItem:
    title: str
    url: str
    status: str
    message: str = ""
    chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "status": self.status,
            "message": self.message,
            "chunks": self.chunks,
        }


@dataclass
class SyncReport:
    started_at: str = ""
    finished_at: str = ""
    items: list[SyncItem] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    error: str = ""

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {message}")
        logger.info(message)

    def add(self, item: SyncItem) -> None:
        self.items.append(item)
        self.log(f"{item.status.upper():<9} {item.title}｜{item.message}")

    def count(self, status: str) -> int:
        return sum(1 for item in self.items if item.status == status)

    @property
    def updated(self) -> int:
        return self.count(STATUS_UPDATED) + self.count(STATUS_NEW) + self.count(STATUS_PLANNED)

    @property
    def failed(self) -> int:
        return self.count(STATUS_FAILED)

    @property
    def total_chunks(self) -> int:
        return sum(item.chunks for item in self.items)

    def summary(self) -> str:
        if self.error:
            return f"同步失败：{self.error}"
        return (
            f"检查 {len(self.items)} 页，更新 {self.updated} 页，"
            f"未变化 {self.count(STATUS_UNCHANGED)} 页，失败 {self.failed} 页，"
            f"新增分块 {self.total_chunks} 条"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary(),
            "error": self.error,
            "logs": self.logs,
            "items": [item.to_dict() for item in self.items],
        }


def load_url_specs(path: Path) -> list[UrlSpec]:
    if not path.exists():
        raise CrawlError(f"监控配置不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CrawlError(f"监控配置不是合法 JSON：{path}（{exc}）") from exc

    entries = raw.get("urls", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise CrawlError("urls.json 中 urls 字段必须是数组")

    specs: list[UrlSpec] = []
    for entry in entries:
        if isinstance(entry, str):
            specs.append(UrlSpec(name=entry, type="html", url=entry))
            continue
        if not isinstance(entry, dict):
            continue
        specs.append(
            UrlSpec(
                name=str(entry.get("name") or entry.get("url") or entry.get("space") or "未命名"),
                type=str(entry.get("type") or "html").lower(),
                url=str(entry.get("url") or "").strip(),
                space=str(entry.get("space") or "").strip(),
                enabled=bool(entry.get("enabled", True)),
                note=str(entry.get("note") or ""),
            )
        )
    return specs


def _flatten_toc(nodes: Iterable[dict[str, Any]], parents: list[str], manual: str, space: str):
    for node in nodes or []:
        title = str(node.get("name") or "").strip()
        url = str(node.get("url") or "").strip()
        if url:
            trail = parents + ([title] if title else [])
            yield PageTarget(
                url=url,
                title=title or "未命名页面",
                section=" › ".join(trail) if trail else manual,
                manual=manual,
                update_time=str(node.get("updateTime") or ""),
                space=space,
            )
        children = node.get("children") or []
        if children:
            yield from _flatten_toc(
                children, parents + ([title] if title else []), manual, space
            )


class CrawlerSync:
    def __init__(
        self,
        settings: Settings | None = None,
        manager: VectorStoreManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.manager = manager or get_manager()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept": "application/json, text/markdown, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": "https://support.unitree.com/",
            }
        )
        self._last_request_at = 0.0

    # ---------------- 网络 ----------------
    def _throttle(self) -> None:
        wait = self.settings.crawl_delay - (time.time() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.time()

    def _request(self, url: str) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.crawl_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.settings.request_timeout)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("请求失败（第 %s 次）：%s（%s）", attempt, url, exc)
                time.sleep(min(2 ** attempt, 8))
                continue

            if response.status_code == 567:
                raise AntiBotError(
                    "官网接口触发了人机校验（HTTP 567）。可稍后重试，或改用本地 PDF 导入。"
                )
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = CrawlError(f"HTTP {response.status_code}")
                logger.warning("服务端异常（第 %s 次）：%s HTTP %s", attempt, url, response.status_code)
                time.sleep(min(2 ** attempt, 8))
                continue
            if response.status_code >= 400:
                raise CrawlError(f"HTTP {response.status_code}：{url}")
            return response
        raise CrawlError(f"请求重试 {self.settings.crawl_retries} 次仍失败：{url}（{last_error}）")

    def _request_json(self, url: str) -> dict[str, Any]:
        try:
            text = self._request(url).content.decode("utf-8-sig", errors="replace")
        except AntiBotError:
            logger.warning("目录接口直连被拦截，改用浏览器指纹重试：%s", url)
            text = self._request_impersonate(url)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            if "<html" in text[:500].lower():
                raise AntiBotError("官网接口返回了 HTML 校验页而不是 JSON。") from exc
            raise CrawlError(f"接口返回非 JSON：{url}") from exc
        if not isinstance(data, dict):
            raise CrawlError(f"接口返回结构异常：{url}")
        return data

    def _request_impersonate(self, url: str) -> str:
        try:
            from curl_cffi import requests as curl_requests
        except ImportError:
            logger.warning("未安装 curl_cffi，退回文本代理通道")
            return self._request_via_proxy(url)

        self._throttle()
        try:
            response = curl_requests.get(
                url,
                impersonate="chrome",
                timeout=self.settings.request_timeout,
                headers={
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                    "Referer": "https://support.unitree.com/",
                },
            )
        except Exception as exc:
            raise CrawlError(f"浏览器指纹请求失败：{url}（{exc}）") from exc
        if response.status_code != 200:
            raise CrawlError(f"浏览器指纹请求返回 HTTP {response.status_code}：{url}")
        return response.text

    def _request_via_proxy(self, url: str) -> str:
        time.sleep(2.0)
        response = self._request(UNITREE_PROXY_PREFIX + url)
        raw = response.content.decode("utf-8", errors="replace")
        start = raw.find("{")
        if start < 0:
            raise CrawlError(f"代理通道没有返回 JSON：{url}")
        try:
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError as exc:
            raise CrawlError(f"代理通道返回内容无法解析：{url}") from exc
        return json.dumps(data, ensure_ascii=False)

    # ---------------- 目录 ----------------
    def fetch_space_pages(self, spec: UrlSpec) -> list[PageTarget]:
        if not spec.space:
            raise CrawlError(f"{spec.name} 缺少 space 字段")
        url = UNITREE_TOC_URL.format(space=quote(spec.space, safe=""))
        payload = self._request_json(url)
        if payload.get("code") not in (100, "100", 0):
            raise CrawlError(f"目录接口返回异常：{payload.get('errorMsg') or payload.get('code')}")

        data = payload.get("data") or {}
        manual = str(data.get("name") or spec.name)
        pages = list(_flatten_toc(data.get("directory") or [], [], manual, spec.space))

        seen: set[str] = set()
        unique: list[PageTarget] = []
        for page in pages:
            if page.url in seen:
                continue
            seen.add(page.url)
            unique.append(page)
        return unique

    def fetch_html_page(self, spec: UrlSpec) -> list[PageTarget]:
        if not spec.url:
            raise CrawlError(f"{spec.name} 缺少 url 字段")
        return [PageTarget(url=spec.url, title=spec.name, section=spec.name, manual=spec.name)]

    def expand(self, spec: UrlSpec) -> list[PageTarget]:
        if spec.type in {"unitree_space", "unitree"}:
            return self.fetch_space_pages(spec)
        return self.fetch_html_page(spec)

    # ---------------- 正文 ----------------
    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?is)<style.*?</style>", " ", text)

        def _table(match: re.Match[str]) -> str:
            block = match.group(0)
            block = re.sub(r"(?is)</t[dh]\s*>", " | ", block)
            block = re.sub(r"(?is)</tr\s*>", "\n", block)
            block = re.sub(r"(?is)<[^>]+>", " ", block)
            return "\n" + block + "\n"

        text = re.sub(r"(?is)<table.*?</table>", _table, text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = html_module.unescape(text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r" *\| *", " | ", text)
        text = re.sub(r"(?m)^ +", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def fetch_page_text(self, page: PageTarget) -> str:
        response = self._request(page.url)
        content_type = (response.headers.get("content-type") or "").lower()
        raw = response.content
        text = ""
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not text:
            text = raw.decode("utf-8", errors="replace")

        looks_html = "html" in content_type or text.lstrip()[:200].lower().startswith("<!doctype")
        if looks_html:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(text, "lxml")
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                    tag.decompose()
                root = soup.find("article") or soup.find("main") or soup.body or soup
                text = root.get_text("\n")
            except Exception as exc:
                logger.warning("HTML 解析失败：%s（%s）", page.url, exc)
                text = ""
        return self._clean_text(text)

    # ---------------- 哈希与缓存 ----------------
    def _load_records(self) -> dict[str, Any]:
        if not self.settings.hash_file.exists():
            return {"version": 1, "records": {}}
        try:
            data = json.loads(self.settings.hash_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("records"), dict):
                raise ValueError("结构异常")
            return data
        except Exception as exc:
            logger.warning("哈希记录损坏，按首次同步处理：%s", exc)
            return {"version": 1, "records": {}}

    def _save_records(self, data: dict[str, Any]) -> None:
        try:
            tmp = self.settings.hash_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.settings.hash_file)
        except Exception as exc:
            logger.warning("保存哈希记录失败：%s", exc)

    def _write_cache(self, page: PageTarget, text: str) -> None:
        try:
            safe_space = re.sub(r"[^\w\-]+", "_", page.space or "html")[:40]
            page_id = urlparse(page.url).path.rstrip("/").split("/")[-1] or "page"
            target_dir = self.settings.docs_cache_dir / safe_space
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / f"{page_id}.md").write_text(text, encoding="utf-8")
        except Exception as exc:
            logger.warning("缓存原文失败：%s（%s）", page.url, exc)

    # ---------------- 单页 ----------------
    def _sync_page(
        self, page: PageTarget, records: dict[str, Any], force: bool, dry_run: bool
    ) -> SyncItem:
        old_md5 = (records["records"].get(page.url) or {}).get("md5")
        try:
            text = self.fetch_page_text(page)
        except AntiBotError as exc:
            return SyncItem(page.title, page.url, STATUS_FAILED, str(exc))
        except Exception as exc:
            return SyncItem(page.title, page.url, STATUS_FAILED, f"{type(exc).__name__}: {exc}")

        text = (text or "").strip()
        if len(text) < 80:
            return SyncItem(page.title, page.url, STATUS_SKIPPED, "正文过短（<80 字），疑似空壳页")

        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        if old_md5 == digest and not force:
            return SyncItem(page.title, page.url, STATUS_UNCHANGED, f"MD5 未变化（{digest[:8]}）")
        if dry_run:
            action = "内容有更新" if old_md5 else "新页面"
            return SyncItem(page.title, page.url, STATUS_PLANNED, f"dry-run：{action}，{len(text)} 字")

        docs = split_documents(
            [
                Document(
                    # 正文常是纯表格，标题里才有机型；把手册名与章节名写进被向量化的文本
                    page_content=f"【手册】{page.manual}｜【章节】{page.section}\n\n{text}",
                    metadata={
                        "manual": page.manual,
                        "section": page.section,
                        "title": page.title,
                        "source": page.url,
                        "update_time": page.update_time,
                        "doc_type": "unitree_doc",
                        "space": page.space,
                    },
                )
            ]
        )
        if not docs:
            return SyncItem(page.title, page.url, STATUS_SKIPPED, "切分后为空")

        try:
            chunks = self.manager.replace_source_documents(page.url, docs)
        except Exception as exc:
            return SyncItem(page.title, page.url, STATUS_FAILED, f"写入向量库失败：{exc}")

        self._write_cache(page, text)
        records["records"][page.url] = {
            "md5": digest,
            "title": page.title,
            "manual": page.manual,
            "section": page.section,
            "update_time": page.update_time,
            "chars": len(text),
            "chunks": len(docs),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        status = STATUS_UPDATED if old_md5 else STATUS_NEW
        message = f"{len(text)} 字 / {len(docs)} 段" if old_md5 else f"首次收录 {len(text)} 字"
        return SyncItem(page.title, page.url, status, message, chunks)

    # ---------------- 批量 ----------------
    def sync(
        self,
        only_space: str | None = None,
        limit: int | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> SyncReport:
        report = SyncReport(started_at=datetime.now().isoformat(timespec="seconds"))
        records = self._load_records()

        try:
            specs = load_url_specs(self.settings.urls_file)
        except Exception as exc:
            report.error = str(exc)
            report.finished_at = datetime.now().isoformat(timespec="seconds")
            return report

        if only_space:
            specs = [s for s in specs if s.space == only_space]
        active = [s for s in specs if s.enabled]
        if not active:
            report.error = "没有启用的监控项，请检查 data/urls.json"
            report.finished_at = datetime.now().isoformat(timespec="seconds")
            return report

        changed = False
        for spec in active:
            try:
                pages = self.expand(spec)
            except Exception as exc:
                report.add(SyncItem(spec.name, spec.url or spec.space, STATUS_FAILED, str(exc)))
                continue

            selected = pages[:limit] if limit and limit > 0 else pages
            report.log(f"{spec.name}：发现 {len(pages)} 页，本次处理 {len(selected)} 页")

            for page in selected:
                item = self._sync_page(page, records, force=force, dry_run=dry_run)
                report.add(item)
                if item.status in {STATUS_UPDATED, STATUS_NEW}:
                    changed = True
                if item.status != STATUS_SKIPPED:
                    self._save_records(records)

        if changed:
            try:
                self.manager.save()
            except Exception as exc:
                report.log(f"索引落盘失败：{exc}")

        report.finished_at = datetime.now().isoformat(timespec="seconds")
        report.log(report.summary())
        return report
