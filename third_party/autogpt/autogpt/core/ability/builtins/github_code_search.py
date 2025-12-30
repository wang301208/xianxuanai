import json
import logging
from typing import Callable, ClassVar, Dict, List, Optional

try:  # Optional dependency for HTTP
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema


class GitHubCodeSearch(Ability):
    """Search public GitHub repositories for code matching a query."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.GitHubCodeSearch",
        ),
        packages_required=["requests"],
        performance_hint=2.5,
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        *,
        http_get: Optional[Callable[..., object]] = None,
    ) -> None:
        self._logger = logger
        self._configuration = configuration
        self._http_get = http_get

    description: ClassVar[str] = (
        "Search GitHub code via the GitHub Search API. Returns matched files and "
        "optionally snippet fragments (text matches) when available."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "GitHub code search query, e.g. 'repo:psf/requests Session auth' or "
                "'language:python requests retry'."
            ),
            required=True,
        ),
        "max_results": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Maximum number of results to return (default 5).",
            default=5,
        ),
        "token": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Optional GitHub token to increase rate limits. Not returned in outputs.",
        ),
    }

    async def __call__(self, query: str, *, max_results: int = 5, token: str | None = None) -> AbilityResult:
        query = (query or "").strip()
        if not query:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=False,
                message="query must be a non-empty string.",
            )
        max_results = max(1, int(max_results))

        if requests is None and self._http_get is None:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query, "max_results": str(max_results)},
                success=False,
                message="requests package not available; provide a custom http_get.",
            )

        try:
            payload = self._search(query, max_results=max_results, token=token)
        except Exception as err:
            self._logger.exception("GitHubCodeSearch failed: %s", err)
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query, "max_results": str(max_results)},
                success=False,
                message=str(err),
            )

        items = payload.get("results", [])
        if not items:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query, "max_results": str(max_results)},
                success=True,
                message="No GitHub code results found for the given query.",
            )

        lines: List[str] = []
        for item in items[:max_results]:
            repo = item.get("repository", "")
            path = item.get("path", "")
            url = item.get("html_url", "")
            lines.append(f"- {repo}/{path} ({url})")

        knowledge = Knowledge(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            content_type=ContentType.TEXT,
            content_metadata={"source": "github_code_search", "query": query},
        )

        message = "GitHub matches:\n" + "\n".join(lines)
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"query": query, "max_results": str(max_results)},
            success=True,
            message=message,
            new_knowledge=knowledge,
        )

    # ------------------------------------------------------------------
    def _search(self, query: str, *, max_results: int, token: str | None) -> Dict[str, object]:
        url = "https://api.github.com/search/code"
        per_page = min(100, max_results)
        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "User-Agent": "AutoGPT-GitHubCodeSearch/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        params = {"q": query, "per_page": per_page}

        if self._http_get is not None:
            response = self._http_get(url, headers=headers, params=params, timeout=10.0)
            return self._parse_response(response, query=query, max_results=max_results)

        resp = requests.get(url, headers=headers, params=params, timeout=10.0)  # type: ignore[union-attr]
        return self._parse_response(resp, query=query, max_results=max_results)

    def _parse_response(self, response: object, *, query: str, max_results: int) -> Dict[str, object]:
        status_code = int(getattr(response, "status_code", 200))
        text = getattr(response, "text", None)
        if hasattr(response, "json") and callable(getattr(response, "json")):
            json_payload = response.json()
        else:
            json_payload = json.loads(text or "{}")
        if status_code >= 400:
            message = json_payload.get("message") if isinstance(json_payload, dict) else None
            raise RuntimeError(f"GitHub API error {status_code}: {message or text or 'unknown error'}")

        items = json_payload.get("items", []) if isinstance(json_payload, dict) else []
        results: List[Dict[str, object]] = []
        for item in (items or [])[:max_results]:
            if not isinstance(item, dict):
                continue
            repo = item.get("repository") or {}
            repo_full = repo.get("full_name") if isinstance(repo, dict) else None
            text_matches = item.get("text_matches") or []
            fragments: List[str] = []
            if isinstance(text_matches, list):
                for match in text_matches[:3]:
                    if isinstance(match, dict) and match.get("fragment"):
                        fragments.append(str(match["fragment"]))
            results.append(
                {
                    "repository": str(repo_full or ""),
                    "path": str(item.get("path") or ""),
                    "html_url": str(item.get("html_url") or ""),
                    "score": float(item.get("score") or 0.0),
                    "fragments": fragments,
                }
            )

        return {
            "query": query,
            "total_count": int(json_payload.get("total_count") or 0) if isinstance(json_payload, dict) else 0,
            "results": results,
        }

