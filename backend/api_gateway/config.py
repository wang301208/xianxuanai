"""Configuration schema for the API gateway."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RouteConfig(BaseModel):
    path: str = Field(..., description="Gateway path prefix, e.g. /monitoring")
    target_service: str = Field(..., description="Service name to resolve via the registry")
    methods: List[str] = Field(default_factory=lambda: ["GET"])
    timeout_seconds: float = 10.0
    cache_ttl_seconds: int = 0
    rewrite_path: Optional[str] = Field(
        default=None,
        description="Optional replacement path forwarded to the downstream service.",
    )


class GatewaySettings(BaseModel):
    service_name: str = "api-gateway"
    host: str = "0.0.0.0"
    port: int = 8080
    default_timeout_seconds: float = 10.0
    routes: List[RouteConfig] = Field(default_factory=list)
