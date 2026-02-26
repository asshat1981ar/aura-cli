"""Skill: fetch web pages or search DuckDuckGo Lite for documentation and references."""
from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

# HTML tag stripper
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s{3,}")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</(script|style)>", re.DOTALL | re.IGNORECASE)

_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AURA-Skill/1.0; +https://github.com/aura-cli)",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
    "Accept-Language": "en-US,en;q=0.5",
}

_TIMEOUT = 10  # seconds


def _strip_html(html: str) -> str:
    """Remove script/style blocks and HTML tags; collapse whitespace."""
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = _SPACE_RE.sub("  ", text)
    return text.strip()


def _fetch_url(url: str, max_chars: int, timeout: int = _TIMEOUT) -> Dict[str, Any]:
    """Fetch a URL and return stripped text content."""
    req = urllib.request.Request(url, headers=_DEFAULT_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(max_chars * 10)  # read more than needed, we'll truncate after stripping
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            text = raw.decode(charset, errors="replace")
            if "text/html" in content_type or text.lstrip().startswith("<"):
                text = _strip_html(text)
            # Try to extract title
            title_m = re.search(r"<title[^>]*>(.*?)</title>", raw.decode("utf-8", errors="replace"), re.IGNORECASE | re.DOTALL)
            title = _strip_html(title_m.group(1)).strip() if title_m else ""
            truncated = len(text) > max_chars
            return {
                "text": text[:max_chars],
                "url": url,
                "title": title[:200],
                "truncated": truncated,
                "content_type": content_type,
                "source": "fetch",
            }
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.reason}", "url": url, "source": "fetch"}
    except urllib.error.URLError as exc:
        return {"error": f"URL error: {exc.reason}", "url": url, "source": "fetch"}
    except Exception as exc:
        return {"error": str(exc), "url": url, "source": "fetch"}


def _ddg_search(query: str, max_chars: int) -> Dict[str, Any]:
    """Search DuckDuckGo Lite and return text results."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    result = _fetch_url(url, max_chars=max_chars * 3)
    if "error" in result:
        return result
    # Extract result snippets from DDG Lite HTML (table-based)
    raw_text = result["text"]
    # Pull out lines that look like search result titles/snippets
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() and len(ln.strip()) > 20]
    # Filter out navigation noise
    lines = [ln for ln in lines if not ln.lower().startswith(("next", "prev", "more", "back", "duckduckgo"))]
    combined = "\n".join(lines[:60])
    return {
        "text": combined[:max_chars],
        "query": query,
        "url": url,
        "title": f"DuckDuckGo results for: {query}",
        "truncated": len(combined) > max_chars,
        "source": "ddg_search",
    }


class WebFetcherSkill(SkillBase):
    name = "web_fetcher"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url: Optional[str] = input_data.get("url")
        query: Optional[str] = input_data.get("query")
        max_chars: int = int(input_data.get("max_chars", 4000))

        if not url and not query:
            return {"error": "Provide 'url' or 'query' in input_data"}

        if url:
            result = _fetch_url(url, max_chars=max_chars)
            log_json("INFO", "web_fetcher_fetch", details={"url": url[:80], "chars": len(result.get("text", ""))})
            return result

        result = _ddg_search(query, max_chars=max_chars)
        log_json("INFO", "web_fetcher_search", details={"query": query[:80], "chars": len(result.get("text", ""))})
        return result
