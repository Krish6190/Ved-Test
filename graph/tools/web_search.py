"""DuckDuckGo web search tool.

Uses the HTML endpoint (no API key, no new dependency — just `requests` and
`beautifulsoup4`, both already in requirements.txt). Returns a list of
{"title", "url", "snippet"} dicts. Never raises — callers treat empty list
as "no useful web results".
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT = 10  # seconds

# DuckDuckGo's HTML endpoint — POST with a `q` form field.
_DDG_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[Dict]:
    """Run a DuckDuckGo text search and return up to `max_results` hits.

    Each hit is `{"title": str, "url": str, "snippet": str}`.
    Returns [] on any error (network, parse, empty query).
    """
    if not query or not query.strip():
        return []
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.post(
            _DDG_URL,
            data={"q": query.strip()},
            headers={"User-Agent": _USER_AGENT, "Referer": "https://duckduckgo.com/"},
            timeout=timeout,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        results: List[Dict] = []
        for node in soup.select(".result")[:max_results]:
            title_node = node.select_one(".result__a")
            snippet_node = node.select_one(".result__snippet")
            title = title_node.get_text(strip=True) if title_node else ""
            href = title_node.get("href", "") if title_node else ""
            snippet = snippet_node.get_text(strip=True) if snippet_node else ""
            if title or snippet:
                results.append({"title": title, "url": href, "snippet": snippet})
        return results
    except Exception as exc:
        logger.warning("[Web Search] Failed for query %r: %s", query, exc)
        return []


def format_web_results_block(results: List[Dict]) -> str:
    """Format web search results into a labeled block for prompt injection.

    Returns "" if results is empty.
    """
    if not results:
        return ""
    lines = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        lines.append(f"({i}) {title}\n    URL: {url}\n    {snippet}")
    return "\n\n".join(lines)
