import json
import logging
from typing import Callable, ClassVar, List, Optional

try:  # Optional dependency for web search
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    DDGS = None

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema

from ._web_utils import scrape_url


class DocumentationTool(Ability):
    """Find documentation pages for a library/API and return scraped excerpts."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.DocumentationTool",
        ),
        packages_required=["duckduckgo-search", "requests", "beautifulsoup4", "readability-lxml"],
        performance_hint=4.0,
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        *,
        search_client: Optional[Callable[[str, int], List[dict]]] = None,
    ) -> None:
        self._logger = logger
        self._configuration = configuration
        self._search_client = search_client

    description: ClassVar[str] = (
        "Search for documentation sources for a library/API query, then scrape top pages and return excerpts."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Library/API name or documentation query (e.g. 'requests Session retries').",
            required=True,
        ),
        "max_sources": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Number of documentation pages to scrape (default 2).",
            default=2,
        ),
        "max_chars_per_source": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Maximum extracted characters per source (default 8000).",
            default=8000,
        ),
        "timeout_s": JSONSchema(
            type=JSONSchema.Type.NUMBER,
            description="Request timeout in seconds (default 10).",
            default=10.0,
        ),
    }

    async def __call__(
        self,
        query: str,
        *,
        max_sources: int = 2,
        max_chars_per_source: int = 8000,
        timeout_s: float = 10.0,
    ) -> AbilityResult:
        query = (query or "").strip()
        if not query:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=False,
                message="query must be a non-empty string.",
            )

        max_sources = max(1, int(max_sources))
        max_chars_per_source = max(500, int(max_chars_per_source))
        timeout_s = float(timeout_s)

        try:
            results = self._perform_search(f"{query} documentation", max_results=max(3, max_sources * 3))
        except Exception as err:
            self._logger.exception("DocumentationTool search failed: %s", err)
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=False,
                message=str(err),
            )

        if not results:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=True,
                message="No documentation sources found.",
            )

        picked = []
        seen = set()
        for item in results:
            url = str(item.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            picked.append(item)
            if len(picked) >= max_sources:
                break

        sources = []
        lines: List[str] = []
        for item in picked:
            url = str(item.get("url") or "")
            title = str(item.get("title") or url)
            try:
                scraped = scrape_url(url, timeout_s=timeout_s, max_chars=max_chars_per_source, include_code=True)
            except Exception as err:
                scraped = {
                    "url": url,
                    "final_url": url,
                    "status_code": 0,
                    "content_type": "",
                    "title": title,
                    "text": "",
                    "code_blocks": [],
                    "error": str(err),
                }
            sources.append({**item, "scrape": scraped})
            lines.append(f"- {title} ({scraped.get('final_url') or url})")

        payload = {
            "query": query,
            "sources": sources,
        }

        knowledge = Knowledge(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            content_type=ContentType.TEXT,
            content_metadata={"source": "documentation_tool", "query": query},
        )

        return AbilityResult(
            ability_name=self.name(),
            ability_args={
                "query": query,
                "max_sources": str(max_sources),
                "max_chars_per_source": str(max_chars_per_source),
                "timeout_s": str(timeout_s),
            },
            success=True,
            message="Documentation sources:\n" + "\n".join(lines),
            new_knowledge=knowledge,
        )

    # ------------------------------------------------------------------
    def _perform_search(self, query: str, *, max_results: int) -> List[dict]:
        if self._search_client is not None:
            return self._search_client(query, max_results)
        if DDGS is None:
            raise RuntimeError("duckduckgo-search package is not available; provide a custom search_client.")
        search = DDGS()
        results = search.text(query, max_results=max_results)
        parsed: List[dict] = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            parsed.append(
                {
                    "title": item.get("title"),
                    "url": item.get("href"),
                    "snippet": item.get("body"),
                }
            )
        return parsed
