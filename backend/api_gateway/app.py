from __future__ import annotations

"""FastAPI powered API gateway with service discovery integration."""

from typing import Dict, Optional, TypedDict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from backend.common.configuration import ConfigNotFoundError, ConfigurationHub, get_hub
from backend.common.logging import configure_from_settings
from backend.common.performance import MultiTierCache
from backend.service_registry import InMemoryServiceRegistry, ServiceInfo, ServiceRegistry

from .config import GatewaySettings, RouteConfig


class CachedResponse(TypedDict):
    status: int
    headers: Dict[str, str]
    body: bytes


def build_app(
    *,
    registry: Optional[ServiceRegistry] = None,
    hub: Optional[ConfigurationHub] = None,
) -> FastAPI:
    hub = hub or get_hub()
    configure_from_settings(hub)
    try:
        settings = hub.get("gateway", model=GatewaySettings)
    except ConfigNotFoundError:
        settings = GatewaySettings()
    registry = registry or InMemoryServiceRegistry()
    return _create_app(settings, registry)


def _create_app(settings: GatewaySettings, registry: ServiceRegistry) -> FastAPI:
    app = FastAPI(title="AutoAI API Gateway", version="0.1.0")
    app.state.registry = registry
    app.state.cache = MultiTierCache[CachedResponse](default_ttl_seconds=60)
    app.state.counters: Dict[str, int] = {}
    app.state.http_client = httpx.AsyncClient(timeout=settings.default_timeout_seconds)

    for route in settings.routes:
        _register_proxy_route(app, route)

    @app.on_event("shutdown")
    async def _shutdown_client() -> None:
        await app.state.http_client.aclose()

    return app


def _register_proxy_route(app: FastAPI, route: RouteConfig) -> None:
    prefix = _ensure_leading_slash(route.path.rstrip("/"))
    wildcard = f"{prefix}/{{full_path:path}}" if prefix else "/{full_path:path}"

    @app.api_route(route.path, methods=route.methods)
    @app.api_route(wildcard, methods=route.methods)
    async def _proxy(request: Request, full_path: str = "") -> Response:  # noqa: ARG001
        return await _proxy_request(app, request, route)


def _ensure_leading_slash(path: str) -> str:
    if not path:
        return ""
    return path if path.startswith("/") else f"/{path}"


async def _proxy_request(
    app: FastAPI,
    request: Request,
    route: RouteConfig,
) -> Response:
    registry: ServiceRegistry = app.state.registry
    instances = registry.list(route.target_service)
    if not instances:
        return JSONResponse(
            status_code=503,
            content={"detail": f"Service {route.target_service} unavailable"},
        )

    target = _select_instance(app, route.target_service, instances)
    forward_path = _compose_target_path(request.url.path, route)
    query = request.url.query
    url = f"{target.endpoint()}{forward_path}"
    if query:
        url = f"{url}?{query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    cache_candidate = request.method.upper() == "GET" and route.cache_ttl_seconds > 0
    cache_key = None
    cache: MultiTierCache[CachedResponse] = app.state.cache
    if cache_candidate:
        cache_key = f"{route.target_service}:{forward_path}"
        if query:
            cache_key = f"{cache_key}?{query}"
        cached = cache.get(cache_key)
        if cached:
            return Response(
                content=cached["body"],
                status_code=cached["status"],
                headers=dict(cached["headers"]),
            )

    body = await request.body()
    client: httpx.AsyncClient = app.state.http_client
    try:
        response = await client.request(
            request.method,
            url,
            headers=headers,
            content=body,
            timeout=route.timeout_seconds,
        )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={"detail": f"Upstream error contacting {route.target_service}: {exc}"},
        )

    filtered_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "connection"}
    }

    if cache_candidate and cache_key:
        cache.set(
            cache_key,
            {
                "status": response.status_code,
                "headers": filtered_headers,
                "body": response.content,
            },
            ttl_seconds=route.cache_ttl_seconds,
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=filtered_headers,
    )


def _compose_target_path(request_path: str, route: RouteConfig) -> str:
    prefix = _ensure_leading_slash(route.path.rstrip("/"))
    suffix = request_path[len(prefix) :] if prefix and request_path.startswith(prefix) else request_path
    base = _ensure_leading_slash(route.rewrite_path.rstrip("/")) if route.rewrite_path else prefix
    combined = (base + suffix) if suffix else base or "/"
    return combined or "/"


def _select_instance(
    app: FastAPI,
    service_name: str,
    instances: list[ServiceInfo],
) -> ServiceInfo:
    counter = app.state.counters.get(service_name, 0)
    target = instances[counter % len(instances)]
    app.state.counters[service_name] = counter + 1
    return target
