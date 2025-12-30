import json
import logging
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema

from ._web_utils import scrape_url


class WebScrape(Ability):
    """Fetch a URL and extract readable text (and optional code blocks)."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.WebScrape",
        ),
        packages_required=["requests", "beautifulsoup4", "readability-lxml"],
        performance_hint=3.5,
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
    ) -> None:
        self._logger = logger
        self._configuration = configuration

    description: ClassVar[str] = (
        "Fetch a web page at the given URL and extract the main readable text. "
        "Optionally also extract code blocks from <pre> sections."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Absolute URL to fetch (http/https).",
            required=True,
        ),
        "max_chars": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Maximum number of characters to return for extracted text (default 12000).",
            default=12_000,
        ),
        "timeout_s": JSONSchema(
            type=JSONSchema.Type.NUMBER,
            description="Request timeout in seconds (default 10).",
            default=10.0,
        ),
        "include_code": JSONSchema(
            type=JSONSchema.Type.BOOLEAN,
            description="Whether to extract code blocks from the page (default true).",
            default=True,
        ),
        "max_code_blocks": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Maximum number of code blocks to include (default 8).",
            default=8,
        ),
    }

    async def __call__(
        self,
        url: str,
        *,
        max_chars: int = 12_000,
        timeout_s: float = 10.0,
        include_code: bool = True,
        max_code_blocks: int = 8,
    ) -> AbilityResult:
        url = (url or "").strip()
        if not url:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"url": url},
                success=False,
                message="url must be a non-empty string.",
            )

        try:
            payload = scrape_url(
                url,
                timeout_s=float(timeout_s),
                max_chars=int(max_chars),
                include_code=bool(include_code),
                max_code_blocks=int(max_code_blocks),
            )
        except Exception as err:
            self._logger.exception("WebScrape failed: %s", err)
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"url": url},
                success=False,
                message=str(err),
            )

        ok = 200 <= int(payload.get("status_code") or 0) < 400
        summary = (
            f"Scraped {payload.get('title') or payload.get('final_url') or url} "
            f"(status={payload.get('status_code')}). "
            f"Extracted {len(payload.get('text') or '')} chars"
            + (
                f", {len(payload.get('code_blocks') or [])} code blocks."
                if include_code
                else "."
            )
        )

        knowledge = Knowledge(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            content_type=ContentType.TEXT,
            content_metadata={"source": "web_scrape", "url": payload.get("final_url") or url},
        )

        return AbilityResult(
            ability_name=self.name(),
            ability_args={
                "url": url,
                "max_chars": max_chars,
                "timeout_s": timeout_s,
                "include_code": include_code,
                "max_code_blocks": max_code_blocks,
            },
            success=bool(ok),
            message=summary,
            new_knowledge=knowledge,
        )

