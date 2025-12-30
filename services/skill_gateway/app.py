
from __future__ import annotations

"""FastAPI application powering the production skill gateway."""

import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, MutableMapping, Optional

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.logging_config import configure_logging
from .config import GatewaySettings, get_settings

configure_logging()
logger = structlog.get_logger("skill-gateway")

app = FastAPI(
    title="Skill Gateway",
    version="1.0.0",
    description="Authenticated gateway for delegated skill execution.",
)


# --------------------------------------------------------------------------- #
# Data structures and helpers


class SkillExecutionError(RuntimeError):
    """Raised when a skill fails in a non-retryable manner."""

    def __init__(self, message: str, *, status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class InvocationContext:
    """Context passed to skill handlers."""

    skill: str
    payload: Mapping[str, Any]
    context: Mapping[str, Any]
    metadata: Mapping[str, Any]
    tenant: str
    principal: Optional[str]
    request_id: str


SkillHandler = Callable[[InvocationContext], Awaitable[MutableMapping[str, Any]] | MutableMapping[str, Any]]


class SkillRegistry:
    """Simple registry mapping skill names to handler callables."""

    def __init__(self) -> None:
        self._handlers: Dict[str, SkillHandler] = {}

    def register(self, name: str, handler: SkillHandler) -> None:
        key = name.lower().strip()
        if not key:
            raise ValueError("Skill name cannot be empty.")
        self._handlers[key] = handler
        logger.debug("skill_handler_registered", skill=key)

    def get(self, name: str) -> Optional[SkillHandler]:
        return self._handlers.get(name.lower().strip())


registry = SkillRegistry()


# --------------------------------------------------------------------------- #
# Dependency helpers


def _get_header(request: Request, header_name: str) -> Optional[str]:
    header_name = header_name.lower()
    for key, value in request.headers.items():
        if key.lower() == header_name:
            return value
    return None


async def resolve_principal(request: Request, settings: GatewaySettings = Depends(get_settings)) -> Optional[str]:
    """Authenticate the inbound request and return the principal identifier."""

    if not settings.require_auth or not settings.normalized_api_keys:
        return None

    candidates = []
    auth_header = _get_header(request, "authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        candidates.append(auth_header.split(" ", 1)[1].strip())

    primary_header = settings.api_key_header
    header_value = _get_header(request, primary_header)
    if header_value:
        candidates.append(header_value.strip())

    # Fallback to legacy header name.
    legacy_header = "X-API-Key"
    if primary_header.lower() != legacy_header.lower():
        legacy_value = _get_header(request, legacy_header)
        if legacy_value:
            candidates.append(legacy_value.strip())

    for candidate in candidates:
        if candidate in settings.normalized_api_keys:
            return candidate

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token.")


async def resolve_tenant(
    request: Request,
    settings: GatewaySettings = Depends(get_settings),
) -> str:
    header_value = _get_header(request, settings.tenant_header)
    if not header_value:
        if settings.require_tenant_header:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required header '{settings.tenant_header}'.",
            )
        header_value = settings.default_tenant

    tenant = header_value.strip()
    if not tenant:
        tenant = settings.default_tenant

    if not settings.is_tenant_allowed(tenant):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tenant '{tenant}' is not registered for this gateway.",
        )

    return tenant


# --------------------------------------------------------------------------- #
# Routes


@app.get("/healthz", tags=["system"])
async def healthz() -> Dict[str, str]:
    """Basic liveness signal."""

    return {"status": "ok"}


@app.get("/readyz", tags=["system"])
async def readyz(settings: GatewaySettings = Depends(get_settings)) -> JSONResponse:
    """Readiness probe that validates downstream dependencies."""

    if not settings.readiness_targets:
        return JSONResponse({"status": "ok", "checks": {}}, status_code=status.HTTP_200_OK)

    checks: Dict[str, Any] = {}
    overall_ok = True

    async with httpx.AsyncClient(timeout=settings.readiness_timeout) as client:
        for target in settings.readiness_targets:
            if not target:
                continue
            try:
                response = await client.get(target)
                checks[target] = {"status": response.status_code}
                if response.status_code >= 400:
                    overall_ok = False
            except httpx.HTTPError as exc:
                overall_ok = False
                checks[target] = {"error": str(exc)}

    status_code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    body = {"status": "ok" if overall_ok else "degraded", "checks": checks}
    return JSONResponse(body, status_code=status_code)


class InvokeRequest(BaseModel):
    """Payload expected from clients invoking skills."""

    skill: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


async def _call_handler(handler: SkillHandler, ctx: InvocationContext) -> MutableMapping[str, Any]:
    result = handler(ctx)
    if inspect.isawaitable(result):
        result = await result  # type: ignore[assignment]
    if not isinstance(result, MutableMapping):
        raise SkillExecutionError("Skill response must be a mapping.", status_code=status.HTTP_502_BAD_GATEWAY)
    return result


@app.post("/invoke", tags=["skills"])
async def invoke_skill(
    request_body: InvokeRequest,
    principal: Optional[str] = Depends(resolve_principal),
    tenant: str = Depends(resolve_tenant),
    settings: GatewaySettings = Depends(get_settings),
) -> MutableMapping[str, Any]:
    """Route a skill invocation to the registered handler."""

    skill_name = request_body.skill.strip().lower()
    handler = registry.get(skill_name)
    if handler is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{request_body.skill}' not registered.",
        )

    request_id = str(uuid.uuid4())
    ctx = InvocationContext(
        skill=skill_name,
        payload=request_body.payload,
        context=request_body.context,
        metadata=request_body.metadata,
        tenant=tenant,
        principal=principal,
        request_id=request_id,
    )

    started = time.perf_counter()
    try:
        result = await _call_handler(handler, ctx)
    except SkillExecutionError as exc:
        logger.warning(
            "skill_failed",
            skill=skill_name,
            tenant=tenant,
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "skill_exception",
            skill=skill_name,
            tenant=tenant,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Skill execution failed unexpectedly.",
        ) from exc
    finally:
        elapsed = time.perf_counter() - started
        if settings.log_invocations:
            logger.info(
                "skill_invocation_completed",
                skill=skill_name,
                tenant=tenant,
                principal=principal or "anonymous",
                request_id=request_id,
                latency_seconds=elapsed,
            )

    result.setdefault("status", "ok")
    result.setdefault("tenant", tenant)
    result.setdefault("request_id", request_id)
    return result


# --------------------------------------------------------------------------- #
# Built-in demo skills compatible with existing smoke tests


def _register_builtin_skills() -> None:
    async def demo_echo(ctx: InvocationContext) -> Dict[str, Any]:
        return {
            "status": "ok",
            "skill": ctx.skill,
            "result": ctx.payload,
            "mode": ctx.metadata.get("execution_mode", "rpc"),
        }

    async def demo_search(ctx: InvocationContext) -> Dict[str, Any]:
        query = ctx.payload.get("query")
        if not query:
            raise SkillExecutionError("Missing 'query' field in payload.", status_code=status.HTTP_400_BAD_REQUEST)
        return {
            "status": "ok",
            "results": [
                {
                    "id": f"result-{ctx.request_id[:8]}",
                    "score": 0.92,
                    "snippet": f"Result snippet for '{query}' (tenant={ctx.tenant})",
                }
            ],
        }

    async def demo_fail(_: InvocationContext) -> Dict[str, Any]:
        raise SkillExecutionError("Simulated upstream failure.")

    registry.register("demo.echo", demo_echo)
    registry.register("demo.search", demo_search)
    registry.register("demo.fail", demo_fail)


_register_builtin_skills()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = str(uuid.uuid4())
    logger.exception(
        "unhandled_exception",
        path=str(request.url),
        request_id=request_id,
    )
    return JSONResponse(
        {"detail": "Internal server error", "request_id": request_id},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
