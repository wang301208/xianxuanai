from __future__ import annotations

"""Prototype RPC client for delegating skill execution to external services."""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

try:  # optional dependency
    import requests
except Exception:  # pragma: no cover - requests may be absent
    requests = None

logger = logging.getLogger(__name__)


class SkillRPCError(RuntimeError):
    """Base error raised for RPC invocation failures."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class SkillRPCConfigurationError(SkillRPCError):
    """Raised when the RPC client configuration is invalid."""


class SkillRPCTransportError(SkillRPCError):
    """Raised when the transport layer fails."""


class SkillRPCResponseError(SkillRPCError):
    """Raised when the remote service returns an invalid response."""


@dataclass
class SkillRPCConfig:
    """Resolved configuration for a remote skill invocation."""

    protocol: str = "http"
    endpoint: Optional[str] = None
    timeout: Optional[float] = None
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    path: Optional[str] = None
    query: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


class SkillRPCClient:
    """Dispatch skill execution to remote services via HTTP, gRPC, or Ray."""

    def __init__(
        self,
        *,
        routing: Optional[Mapping[str, Mapping[str, Any]]] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        default_timeout: float = 15.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        session: Optional["requests.Session"] = None,
    ) -> None:
        self._routing: Dict[str, Dict[str, Any]] = {
            key: dict(value) for key, value in (routing or {}).items()
        }
        self._defaults: Dict[str, Any] = dict(defaults or {})
        self._default_timeout = max(default_timeout, 0.1)
        self._max_retries = max(0, max_retries)
        self._backoff = max(0.0, backoff_seconds)
        self._session = session or (requests.Session() if requests is not None else None)

    # ------------------------------------------------------------------ factory helpers
    @classmethod
    def from_env(cls) -> "SkillRPCClient":
        """Instantiate a client using ``SKILL_RPC_*`` environment variables."""

        routing = {}
        defaults: Dict[str, Any] = {}

        raw_defaults = os.getenv("SKILL_RPC_DEFAULTS")
        if raw_defaults:
            try:
                defaults.update(json.loads(raw_defaults))
            except json.JSONDecodeError as err:
                logger.warning("Failed to parse SKILL_RPC_DEFAULTS: %s", err)

        base_url = os.getenv("SKILL_RPC_BASE_URL")
        if base_url:
            defaults.setdefault("endpoint", base_url)
        protocol = os.getenv("SKILL_RPC_PROTOCOL")
        if protocol:
            defaults.setdefault("protocol", protocol)

        raw_headers = os.getenv("SKILL_RPC_HEADERS")
        if raw_headers:
            try:
                headers = json.loads(raw_headers)
                if isinstance(headers, dict):
                    defaults.setdefault("headers", headers)
            except json.JSONDecodeError as err:
                logger.warning("Failed to parse SKILL_RPC_HEADERS: %s", err)

        raw_routing = os.getenv("SKILL_RPC_ROUTING")
        if raw_routing:
            try:
                routing_payload = json.loads(raw_routing)
                if isinstance(routing_payload, dict):
                    routing = {
                        str(name): dict(cfg) if isinstance(cfg, dict) else {}
                        for name, cfg in routing_payload.items()
                    }
            except json.JSONDecodeError as err:
                logger.warning("Failed to parse SKILL_RPC_ROUTING: %s", err)

        timeout = float(os.getenv("SKILL_RPC_TIMEOUT", "0") or 0)
        if timeout > 0:
            defaults.setdefault("timeout", timeout)

        retries = int(os.getenv("SKILL_RPC_RETRIES", "0") or 0)
        backoff = float(os.getenv("SKILL_RPC_BACKOFF", "0.5") or 0.5)

        return cls(
            routing=routing,
            defaults=defaults,
            default_timeout=timeout if timeout > 0 else 15.0,
            max_retries=retries,
            backoff_seconds=backoff,
        )

    # ------------------------------------------------------------------ public API
    def invoke(
        self,
        skill_name: str,
        payload: Mapping[str, Any],
        *,
        context: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Invoke ``skill_name`` on a remote service."""

        config = self._resolve_config(skill_name, metadata)
        protocol = config.protocol.lower()
        effective_timeout = timeout or config.timeout or self._default_timeout
        attempt = 0

        while True:
            try:
                if protocol in {"http", "https", "rest", "fastapi", "ray_serve"}:
                    return self._invoke_http(
                        config,
                        skill_name,
                        payload,
                        context or {},
                        metadata or {},
                        effective_timeout,
                    )
                if protocol == "grpc":
                    return self._invoke_grpc(
                        config,
                        skill_name,
                        payload,
                        context or {},
                        metadata or {},
                        effective_timeout,
                    )
                if protocol == "ray":
                    return self._invoke_ray(
                        config,
                        skill_name,
                        payload,
                        context or {},
                        metadata or {},
                        effective_timeout,
                    )
                raise SkillRPCConfigurationError(f"Unsupported RPC protocol '{protocol}'.")
            except SkillRPCError as exc:
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                sleep_for = self._backoff * (2 ** attempt)
                logger.debug(
                    "Retrying RPC for skill '%s' (attempt %s/%s) after %.2fs: %s",
                    skill_name,
                    attempt + 1,
                    self._max_retries,
                    sleep_for,
                    exc,
                )
                time.sleep(sleep_for)
                attempt += 1

    # ------------------------------------------------------------------ configuration
    def _resolve_config(
        self,
        skill_name: str,
        metadata: Optional[Mapping[str, Any]],
    ) -> SkillRPCConfig:
        resolved: Dict[str, Any] = dict(self._defaults)

        # Wildcard routing applies to all skills.
        if "*" in self._routing:
            resolved.update(self._routing["*"])
        if skill_name in self._routing:
            resolved.update(self._routing[skill_name])

        metadata = metadata or {}
        rpc_config = metadata.get("rpc_config")
        if isinstance(rpc_config, Mapping):
            resolved.update(rpc_config)

        # Flatten convenience keys like rpc_endpoint, rpc_protocol, etc.
        for key in ("endpoint", "protocol", "timeout", "method", "path", "headers", "query"):
            meta_key = f"rpc_{key}"
            if meta_key in metadata:
                resolved[key] = metadata[meta_key]

        protocol = str(resolved.get("protocol", "http") or "http").lower()
        endpoint = resolved.get("endpoint")
        method = str(resolved.get("method", "POST") or "POST").upper()
        timeout = resolved.get("timeout")
        headers = self._normalise_headers(resolved.get("headers"))
        path = resolved.get("path")
        query = resolved.get("query")
        options = resolved.get("options") or {}

        config = SkillRPCConfig(
            protocol=protocol,
            endpoint=endpoint,
            timeout=float(timeout) if timeout not in (None, "") else None,
            method=method,
            headers=headers,
            path=str(path) if path not in (None, "") else None,
            query=dict(query) if isinstance(query, Mapping) else {},
            options=dict(options) if isinstance(options, Mapping) else {},
        )
        return config

    @staticmethod
    def _normalise_headers(headers: Any) -> Dict[str, str]:
        if not headers:
            return {}
        if isinstance(headers, Mapping):
            return {str(k): str(v) for k, v in headers.items()}
        if isinstance(headers, Iterable):
            return {str(key): str(value) for key, value in headers}  # type: ignore[misc]
        return {}

    # ------------------------------------------------------------------ protocol handlers
    def _invoke_http(
        self,
        config: SkillRPCConfig,
        skill_name: str,
        payload: Mapping[str, Any],
        context: Mapping[str, Any],
        metadata: Mapping[str, Any],
        timeout: float,
    ) -> Any:
        if requests is None:
            raise SkillRPCConfigurationError("requests package is required for HTTP RPC calls.")
        if self._session is None:
            self._session = requests.Session()

        url = self._build_http_url(config)
        body = {
            "skill": skill_name,
            "payload": payload,
            "context": context,
            "metadata": dict(metadata),
        }
        method = config.method or "POST"
        try:
            response = self._session.request(
                method=method,
                url=url,
                headers=config.headers or {"Content-Type": "application/json"},
                params=config.query or None,
                json=body,
                timeout=timeout,
            )
        except requests.Timeout as exc:  # type: ignore[attr-defined]
            raise SkillRPCTransportError(f"HTTP request timed out for {url}", retryable=True) from exc
        except requests.RequestException as exc:  # type: ignore[attr-defined]
            raise SkillRPCTransportError(f"HTTP request failed: {exc}", retryable=True) from exc

        if response.status_code >= 400:
            raise SkillRPCResponseError(
                f"RPC service returned HTTP {response.status_code}: {response.text.strip()}",
                retryable=response.status_code >= 500,
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise SkillRPCResponseError("Failed to decode JSON response from RPC service.") from exc

    def _build_http_url(self, config: SkillRPCConfig) -> str:
        endpoint = config.endpoint or ""
        path = config.path or ""
        if path and not path.startswith("/") and not endpoint.endswith("/"):
            path = f"/{path}"
        if endpoint and path:
            return f"{endpoint.rstrip('/')}{path}"
        if path:
            return path
        if not endpoint:
            raise SkillRPCConfigurationError("HTTP RPC configuration requires an endpoint or path.")
        return endpoint

    def _invoke_grpc(
        self,
        config: SkillRPCConfig,
        skill_name: str,
        payload: Mapping[str, Any],
        context: Mapping[str, Any],
        metadata: Mapping[str, Any],
        timeout: float,
    ) -> Any:
        try:
            import grpc  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SkillRPCConfigurationError("grpcio package is required for gRPC RPC calls.") from exc

        target = config.endpoint or config.options.get("target")
        if not target:
            raise SkillRPCConfigurationError("gRPC configuration requires an endpoint/target.")

        method = config.options.get("method") or config.options.get("rpc_method") or "/SkillService/Invoke"
        serializer = config.options.get("request_serializer")
        deserializer = config.options.get("response_deserializer")
        metadata_headers = config.options.get("metadata") or config.options.get("headers") or {}
        channel_options = config.options.get("channel_options") or []
        secure = bool(config.options.get("secure", False))

        if serializer is None:
            serializer = lambda data: json.dumps(data).encode("utf-8")
        if deserializer is None:
            deserializer = lambda data: json.loads(data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data)

        call_headers: Iterable[Tuple[str, str]] = []
        if isinstance(metadata_headers, Mapping):
            call_headers = [(str(k), str(v)) for k, v in metadata_headers.items()]
        elif isinstance(metadata_headers, Iterable):
            call_headers = [(str(k), str(v)) for k, v in metadata_headers]  # type: ignore[misc]

        request_payload = {
            "skill": skill_name,
            "payload": payload,
            "context": context,
            "metadata": dict(metadata),
        }

        channel = None
        try:
            if secure:
                credentials = config.options.get("credentials")
                if credentials is None:
                    credentials = grpc.ssl_channel_credentials()
                channel = grpc.secure_channel(target, credentials, options=channel_options)
            else:
                channel = grpc.insecure_channel(target, options=channel_options)
            stub = channel.unary_unary(
                method,
                request_serializer=serializer,
                response_deserializer=deserializer,
            )
            response = stub(
                request_payload,
                timeout=timeout,
                metadata=list(call_headers) or None,
            )
            return response
        except grpc.RpcError as exc:
            code = exc.code()
            retryable = code in {
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            }
            raise SkillRPCTransportError(f"gRPC call failed: {code.name}", retryable=retryable) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise SkillRPCTransportError(f"gRPC invocation error: {exc}", retryable=True) from exc
        finally:
            if channel is not None:
                channel.close()

    def _invoke_ray(
        self,
        config: SkillRPCConfig,
        skill_name: str,
        payload: Mapping[str, Any],
        context: Mapping[str, Any],
        metadata: Mapping[str, Any],
        timeout: float,
    ) -> Any:
        try:
            import ray  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SkillRPCConfigurationError("ray package is required for Ray RPC calls.") from exc

        options = config.options or {}
        address = options.get("address") or config.endpoint
        namespace = options.get("namespace")
        init_kwargs = options.get("init_kwargs") or {}
        if not ray.is_initialized():
            ray.init(address=address or "auto", namespace=namespace, ignore_reinit_error=True, **init_kwargs)

        actor_name = options.get("actor")
        deployment = options.get("deployment")
        method_name = options.get("method") or "invoke"

        try:
            if actor_name:
                actor = ray.get_actor(actor_name, namespace=namespace)
                remote_callable = getattr(actor, method_name)
                ref = remote_callable.remote(
                    skill_name=skill_name,
                    payload=payload,
                    context=context,
                    metadata=dict(metadata),
                )
                return ray.get(ref, timeout=timeout)
            if deployment:
                try:
                    from ray import serve  # type: ignore
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise SkillRPCConfigurationError("ray[serve] is required for Ray Serve integration.") from exc
                handle = serve.get_app_handle(deployment, namespace=namespace)
                ref = handle.remote(
                    skill_name=skill_name,
                    payload=payload,
                    context=context,
                    metadata=dict(metadata),
                )
                return ray.get(ref, timeout=timeout)
        except ray.exceptions.GetTimeoutError as exc:  # type: ignore[attr-defined]
            raise SkillRPCTransportError("Ray remote call timed out.", retryable=True) from exc
        except ValueError as exc:
            raise SkillRPCTransportError(str(exc), retryable=True) from exc
        except AttributeError as exc:
            raise SkillRPCConfigurationError(f"Ray target missing method '{method_name}'.") from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            raise SkillRPCTransportError(f"Ray invocation error: {exc}", retryable=True) from exc

        raise SkillRPCConfigurationError("Ray RPC configuration requires either 'actor' or 'deployment'.")


__all__ = [
    "SkillRPCClient",
    "SkillRPCError",
    "SkillRPCConfigurationError",
    "SkillRPCTransportError",
    "SkillRPCResponseError",
]
