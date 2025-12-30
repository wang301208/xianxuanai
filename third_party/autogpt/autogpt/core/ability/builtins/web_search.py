import json
import logging
from typing import Callable, ClassVar, List, Optional

try:  # Optional dependency for real web search
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    DDGS = None

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema


class WebSearch(Ability):
    """Query a web search API and return a concise result set."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.WebSearch",
        ),
        packages_required=["duckduckgo-search"],
        performance_hint=2.0,
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
        "Execute a web search for the given query and return the top results with snippets."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Natural language query to look up online resources.",
        ),
        "max_results": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Maximum number of search results to return (default 5).",
            default=5,
        ),
    }

    async def __call__(self, query: str, *, max_results: int = 5) -> AbilityResult:
        query = (query or "").strip()
        if not query:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=False,
                message="Query must be a non-empty string.",
            )

        try:
            results = self._perform_search(query, max_results=max_results)
        except Exception as err:
            self._logger.exception("WebSearch failed: %s", err)
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query, "max_results": max_results},
                success=False,
                message=str(err),
            )

        if not results:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query, "max_results": max_results},
                success=True,
                message="No results found for the given query.",
            )

        summary_lines = []
        for item in results:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            summary_lines.append(f"- {title} ({url})")

        payload = json.dumps(
            {"query": query, "results": results},
            ensure_ascii=False,
            indent=2,
        )

        knowledge = Knowledge(
            content=payload,
            content_type=ContentType.TEXT,
            content_metadata={"source": "web_search", "query": query},
        )

        message = "Results:\n" + "\n".join(summary_lines[:max_results])
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"query": query, "max_results": max_results},
            success=True,
            message=message,
            new_knowledge=knowledge,
        )

    # ------------------------------------------------------------------#
    def _perform_search(self, query: str, *, max_results: int) -> List[dict]:
        if self._search_client is not None:
            return self._search_client(query, max_results)

        if DDGS is None:
            raise RuntimeError(
                "duckduckgo-search package is not available; provide a custom search_client."
            )

        search = DDGS()
        results = search.text(query, max_results=max_results)
        parsed: List[dict] = []
        for item in results or []:
            parsed.append(
                {
                    "title": item.get("title"),
                    "url": item.get("href"),
                    "snippet": item.get("body"),
                }
            )
        return parsed
