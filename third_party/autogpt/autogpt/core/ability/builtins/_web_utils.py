from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

try:  # Optional dependency for robust HTML parsing
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    BeautifulSoup = None  # type: ignore[assignment]

try:  # Optional dependency for better article extraction
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    Document = None  # type: ignore[assignment]

try:  # Optional dependency for HTTP
    import requests  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    requests = None  # type: ignore[assignment]


_DEFAULT_USER_AGENT = "AutoGPT-WebTool/1.0 (+https://github.com/openai)"


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", str(text or "")).strip()


def _is_probably_html(content_type: str | None) -> bool:
    if not content_type:
        return True
    lowered = content_type.lower()
    return "text/html" in lowered or "application/xhtml" in lowered


def _extract_title(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup is None:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        return _compact_whitespace(match.group(1)) if match else ""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else ""
    return _compact_whitespace(title or "")


def _extract_main_html(html: str) -> str:
    if not html:
        return ""
    if Document is None:
        return html
    try:
        doc = Document(html)
        return doc.summary(html_partial=True) or html
    except Exception:  # pragma: no cover - best-effort extraction
        return html


def _extract_text_from_html(html: str) -> str:
    if not html:
        return ""
    html = _extract_main_html(html)
    if BeautifulSoup is None:
        # Very rough fallback: strip tags.
        text = re.sub(r"(?is)<(script|style).*?>.*?</\\\\1>", " ", html)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        return _compact_whitespace(text)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            pass
    text = soup.get_text(separator=" ")
    return _compact_whitespace(text)


def _extract_code_blocks_from_html(
    html: str,
    *,
    max_blocks: int = 8,
    max_chars_per_block: int = 2000,
) -> List[str]:
    if not html or max_blocks <= 0:
        return []
    if BeautifulSoup is None:
        blocks = []
        for match in re.finditer(r"(?is)<pre[^>]*>(.*?)</pre>", html):
            content = re.sub(r"(?is)<[^>]+>", "", match.group(1) or "")
            cleaned = (content or "").strip()
            if cleaned:
                blocks.append(cleaned[:max_chars_per_block])
            if len(blocks) >= max_blocks:
                break
        return blocks

    soup = BeautifulSoup(html, "html.parser")
    blocks: List[str] = []
    for pre in soup.find_all("pre"):
        text = pre.get_text("\\n")
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        blocks.append(cleaned[:max_chars_per_block])
        if len(blocks) >= max_blocks:
            break
    return blocks


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str


def fetch_url_text(
    url: str,
    *,
    timeout_s: float = 10.0,
    max_bytes: int = 2_000_000,
    user_agent: str | None = None,
) -> FetchedPage:
    if not url:
        raise ValueError("url must be non-empty")
    ua = _DEFAULT_USER_AGENT if not user_agent else str(user_agent)

    if requests is None:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(getattr(resp, "status", 200))
            headers = getattr(resp, "headers", {})
            content_type = str(headers.get("Content-Type", ""))
            raw = resp.read(max_bytes if max_bytes > 0 else None)
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = raw.decode("utf-8", errors="replace")
            return FetchedPage(
                url=str(url),
                final_url=str(getattr(resp, "url", url)),
                status_code=status,
                content_type=content_type,
                text=text,
            )

    resp = requests.get(
        url,
        headers={"User-Agent": ua},
        timeout=float(timeout_s),
    )
    content_type = str(resp.headers.get("Content-Type", ""))
    raw = resp.content
    if max_bytes > 0 and len(raw) > max_bytes:
        raw = raw[:max_bytes]
    encoding = resp.encoding or "utf-8"
    try:
        text = raw.decode(encoding, errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")
    return FetchedPage(
        url=str(url),
        final_url=str(resp.url or url),
        status_code=int(resp.status_code),
        content_type=content_type,
        text=text,
    )


def scrape_url(
    url: str,
    *,
    timeout_s: float = 10.0,
    max_chars: int = 12_000,
    max_bytes: int = 2_000_000,
    include_code: bool = True,
    max_code_blocks: int = 8,
    user_agent: str | None = None,
) -> Dict[str, Any]:
    page = fetch_url_text(url, timeout_s=timeout_s, max_bytes=max_bytes, user_agent=user_agent)
    raw_text = page.text or ""
    if _is_probably_html(page.content_type):
        title = _extract_title(raw_text)
        extracted_text = _extract_text_from_html(raw_text)
        code_blocks = (
            _extract_code_blocks_from_html(raw_text, max_blocks=max_code_blocks)
            if include_code
            else []
        )
    else:
        title = ""
        extracted_text = _compact_whitespace(raw_text)
        code_blocks = []

    if max_chars > 0 and len(extracted_text) > max_chars:
        extracted_text = extracted_text[:max_chars]

    return {
        "url": page.url,
        "final_url": page.final_url,
        "status_code": page.status_code,
        "content_type": page.content_type,
        "title": title,
        "text": extracted_text,
        "code_blocks": code_blocks,
    }

