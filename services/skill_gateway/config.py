from __future__ import annotations

"""Runtime configuration utilities for the skill gateway service."""

from dataclasses import dataclass, field
from functools import lru_cache
import os
from typing import Iterable, List, Set


def _to_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class GatewaySettings:
    """Aggregated configuration derived from environment variables."""

    api_keys: Set[str] = field(default_factory=set)
    api_key_header: str = "X-API-Key"
    allowed_tenants: Set[str] = field(default_factory=set)
    default_tenant: str = "public"
    require_auth: bool = False
    require_tenant_header: bool = False
    tenant_header: str = "X-Tenant-ID"
    readiness_targets: List[str] = field(default_factory=list)
    readiness_timeout: float = 2.0
    log_invocations: bool = True

    @property
    def normalized_api_keys(self) -> Set[str]:
        return {token.strip() for token in self.api_keys if token.strip()}

    @property
    def normalized_tenants(self) -> Set[str]:
        return {tenant.lower() for tenant in self.allowed_tenants if tenant}

    def is_tenant_allowed(self, tenant: str) -> bool:
        allowed = self.normalized_tenants
        return not allowed or tenant.lower() in allowed


def _resolve_default_tenant(values: Iterable[str]) -> str:
    for item in values:
        if item:
            return item
    return "public"


@lru_cache(maxsize=1)
def get_settings() -> GatewaySettings:
    """Return cached gateway settings."""

    api_keys = set(_split_csv(os.getenv("GATEWAY_API_KEY") or os.getenv("GATEWAY_API_KEYS")))
    api_keys.update(_split_csv(os.getenv("GATEWAY_ADDITIONAL_API_KEYS")))

    tenants = _split_csv(os.getenv("GATEWAY_ALLOWED_TENANTS"))
    default_tenant = os.getenv("GATEWAY_DEFAULT_TENANT")
    default_tenant = _resolve_default_tenant([default_tenant, *(tenants or ["public"])])

    readiness_targets_env = os.getenv("GATEWAY_READINESS_TARGETS")
    readiness_targets = (
        _split_csv(readiness_targets_env)
        if readiness_targets_env
        else ["http://qdrant:6333/healthz", "http://neo4j:7474/"]
    )

    timeout = float(os.getenv("GATEWAY_READINESS_TIMEOUT", "2.5") or 2.5)

    return GatewaySettings(
        api_keys=api_keys,
        api_key_header=os.getenv("GATEWAY_API_KEY_HEADER", "X-API-Key"),
        allowed_tenants=set(tenants),
        default_tenant=default_tenant,
        require_auth=_to_bool(os.getenv("GATEWAY_REQUIRE_AUTH"), default=False),
        require_tenant_header=_to_bool(os.getenv("GATEWAY_REQUIRE_TENANT_HEADER"), default=False),
        tenant_header=os.getenv("GATEWAY_TENANT_HEADER", "X-Tenant-ID"),
        readiness_targets=readiness_targets,
        readiness_timeout=timeout,
        log_invocations=_to_bool(os.getenv("GATEWAY_LOG_INVOCATIONS"), default=True),
    )
