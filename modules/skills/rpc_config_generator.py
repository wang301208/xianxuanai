from __future__ import annotations

"""Extract RPC configuration from documentation text.

This helper is used to turn human-facing docs (README, API docs snippets, etc.)
into a `rpc_config` mapping compatible with `SkillRPCClient`/skill manifests.

The generator is dependency-light:
- Default path uses deterministic regex/URL heuristics.
- Optional LLM path can be enabled by passing an OpenAI-style client and model.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
from urllib.parse import parse_qsl, urlparse

logger = logging.getLogger(__name__)


def _clip(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    value = str(text or "")
    return value if len(value) <= max_chars else value[:max_chars] + "..."


def _safe_json_loads(payload: str) -> Optional[Any]:
    try:
        return json.loads(payload)
    except Exception:
        return None


def _extract_json_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    fence_re = re.compile(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", flags=re.DOTALL)
    for match in fence_re.finditer(text or ""):
        body = match.group(1)
        if body:
            blocks.append(body.strip())
    return blocks


def _normalize_headers(value: Any) -> Dict[str, str]:
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items() if k}
    return {}


def _normalize_rpc_config(mapping: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = dict(mapping or {})
    candidate = raw.get("rpc_config") if isinstance(raw.get("rpc_config"), Mapping) else None
    if isinstance(candidate, Mapping):
        raw = dict(candidate)

    protocol = str(raw.get("protocol") or raw.get("rpc_protocol") or "").strip().lower()
    if not protocol:
        protocol = "grpc" if any(key in raw for key in ("grpc", "rpc_method", "method")) and str(raw.get("endpoint", "")).count(":") == 1 else "http"
    if protocol in {"https"}:
        protocol = "http"
    if protocol not in {"http", "grpc", "ray"}:
        protocol = "http"

    endpoint = raw.get("endpoint") or raw.get("base_url") or raw.get("url") or raw.get("target")
    endpoint_str = str(endpoint).strip() if endpoint not in (None, "") else ""
    path = raw.get("path")
    path_str = str(path).strip() if path not in (None, "") else ""

    method = str(raw.get("method") or "POST").strip().upper()
    headers = _normalize_headers(raw.get("headers"))
    query_raw = raw.get("query")
    query = dict(query_raw) if isinstance(query_raw, Mapping) else {}
    options_raw = raw.get("options") or {}
    options = dict(options_raw) if isinstance(options_raw, Mapping) else {}

    if protocol == "grpc":
        if not endpoint_str:
            return None
        return {
            "protocol": "grpc",
            "endpoint": endpoint_str,
            "timeout": raw.get("timeout"),
            "options": options,
        }

    if protocol == "ray":
        if not (endpoint_str or options):
            return None
        out: Dict[str, Any] = {
            "protocol": "ray",
            "endpoint": endpoint_str or None,
            "timeout": raw.get("timeout"),
            "options": options,
        }
        return {k: v for k, v in out.items() if v not in (None, "", {}, [])}

    # HTTP
    if endpoint_str and endpoint_str.startswith(("http://", "https://")):
        parsed = urlparse(endpoint_str)
        if parsed.scheme and parsed.netloc and parsed.path and not path_str:
            path_str = parsed.path
        endpoint_str = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else endpoint_str
        if parsed.query and not query:
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    if path_str and not path_str.startswith("/") and "://" not in path_str:
        path_str = "/" + path_str

    if not endpoint_str and not path_str:
        return None

    out = {
        "protocol": "http",
        "endpoint": endpoint_str or None,
        "path": path_str or None,
        "method": method,
        "headers": headers,
        "query": query,
        "timeout": raw.get("timeout"),
        "options": options,
    }
    return {k: v for k, v in out.items() if v not in (None, "", {}, [])}


def _score_url(url: str) -> int:
    score = 0
    lower = url.lower()
    if "/invoke" in lower:
        score += 5
    if "/infer" in lower:
        score += 3
    if "/v1/" in lower or "/api/" in lower:
        score += 2
    if "health" in lower:
        score -= 3
    if "docs" in lower or "swagger" in lower:
        score -= 1
    if lower.startswith("https://"):
        score += 1
    return score


def _infer_rpc_config_heuristic(text: str) -> Optional[Dict[str, Any]]:
    doc = str(text or "").replace("\\r", "\r").replace("\\n", "\n")
    lower = doc.lower()

    # gRPC hint: prefer host:port + method like /Service/Invoke
    if "grpc" in lower:
        target_match = re.search(r"\b([a-zA-Z0-9_.-]+:\d{2,5})\b", doc)
        method_match = re.search(r"(/[\w.]+/[\w.]+)", doc)
        if target_match:
            options: Dict[str, Any] = {}
            if method_match:
                options["method"] = method_match.group(1)
            return _normalize_rpc_config({"protocol": "grpc", "endpoint": target_match.group(1), "options": options})

    # 1) Look for an explicit full URL.
    url_re = re.compile(r"(https?://[^\s\"'<>]+)")
    urls = [m.group(1).rstrip(").,") for m in url_re.finditer(doc)]
    if urls:
        urls_sorted = sorted(urls, key=_score_url, reverse=True)
        chosen = urls_sorted[0]
        method = "POST"
        method_match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\b", doc, flags=re.IGNORECASE)
        if method_match:
            method = method_match.group(1).upper()
        config = _normalize_rpc_config({"protocol": "http", "endpoint": chosen, "method": method})
        if config:
            return config

    # 2) Base URL + path pattern.
    base_match = re.search(r"\b(base url|endpoint)\s*[:=]\s*(https?://[^\s\"'<>]+)", lower, flags=re.IGNORECASE)
    base_url = None
    if base_match:
        base_url = base_match.group(2).rstrip(").,")
    else:
        simple_base = re.search(r"\bhttps?://[a-zA-Z0-9_.:-]+", doc)
        if simple_base:
            base_url = simple_base.group(0).rstrip(").,")

    path_match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s\"']+)", doc)
    method = path_match.group(1).upper() if path_match else "POST"
    path = path_match.group(2) if path_match else "/invoke"

    if base_url:
        return _normalize_rpc_config({"protocol": "http", "endpoint": base_url, "path": path, "method": method})

    # 3) Host:port + /invoke shorthand.
    host_port = re.search(r"\b([a-zA-Z0-9_.-]+:\d{2,5})\b", doc)
    if host_port:
        endpoint = f"http://{host_port.group(1)}"
        path_guess = "/invoke" if "/invoke" in lower else "/infer" if "/infer" in lower else "/invoke"
        return _normalize_rpc_config({"protocol": "http", "endpoint": endpoint, "path": path_guess, "method": "POST"})

    return None


def _try_parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, Mapping):
        cfg = _normalize_rpc_config(parsed)
        if cfg:
            return cfg
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = _safe_json_loads(raw[start : end + 1])
        if isinstance(parsed, Mapping):
            cfg = _normalize_rpc_config(parsed)
            if cfg:
                return cfg
    return None


@dataclass
class RPCConfigGenerationConfig:
    model: str | None = None
    request_timeout: float | None = 15.0
    max_input_chars: int = 8000
    allow_llm: bool = True


@dataclass
class RPCConfigGenerationResult:
    rpc_config: Optional[Dict[str, Any]]
    used_llm: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class SkillRPCConfigGenerator:
    """Generate `rpc_config` from documentation text.

    The `llm_client` is expected to be OpenAI-compatible (supports
    `chat.completions.create`).
    """

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        config: RPCConfigGenerationConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or RPCConfigGenerationConfig()

    def generate(self, docs_text: str, *, hint: str | None = None) -> RPCConfigGenerationResult:
        clipped = _clip(docs_text, max_chars=int(self.config.max_input_chars))

        # 1) Structured JSON blocks embedded in docs.
        for block in _extract_json_blocks(clipped):
            parsed = _safe_json_loads(block)
            if isinstance(parsed, Mapping):
                cfg = _normalize_rpc_config(parsed)
                if cfg:
                    return RPCConfigGenerationResult(
                        rpc_config=cfg,
                        used_llm=False,
                        diagnostics={"source": "json_block"},
                    )

        # 2) Optional LLM extraction.
        if self._can_use_llm():
            try:
                cfg = self._invoke_llm(clipped, hint=hint)
                if cfg:
                    return RPCConfigGenerationResult(
                        rpc_config=cfg,
                        used_llm=True,
                        diagnostics={"source": "llm"},
                    )
            except Exception as exc:  # pragma: no cover - depends on external LLM
                logger.warning("RPC config LLM extraction failed: %s", exc)

        # 3) Heuristic fallback.
        cfg = _infer_rpc_config_heuristic(clipped)
        return RPCConfigGenerationResult(
            rpc_config=cfg,
            used_llm=False,
            diagnostics={"source": "heuristic" if cfg else "none"},
        )

    def _can_use_llm(self) -> bool:
        return bool(self.config.allow_llm and self.llm_client is not None and self.config.model)

    def _invoke_llm(self, docs_text: str, *, hint: str | None = None) -> Optional[Dict[str, Any]]:
        model = self.config.model
        if not model:
            return None
        prompt = (
            "Extract an RPC configuration for invoking a remote skill/service.\n"
            "Return ONLY a JSON object compatible with SkillRPCClient rpc_config:\n"
            "- protocol: \"http\" | \"grpc\" | \"ray\"\n"
            "- endpoint: base URL (http) or host:port (grpc)\n"
            "- path: HTTP path (e.g. \"/invoke\")\n"
            "- method: HTTP method (e.g. \"POST\")\n"
            "- headers: object of header key/value strings\n"
            "- query: object\n"
            "- options: object\n"
            "Do not include markdown or extra keys.\n"
        )
        if hint:
            prompt += f"Hint: {hint}\n"
        prompt += "\nDocumentation:\n" + docs_text

        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured RPC configs from documentation. Output strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=self.config.request_timeout,
        )
        choice = getattr(response, "choices", [None])[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if not content:
            return None
        return _try_parse_llm_json(str(content))


__all__ = ["SkillRPCConfigGenerator", "RPCConfigGenerationConfig", "RPCConfigGenerationResult"]
