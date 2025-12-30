"""Remote tool protocol (TCP JSONL) for controlling another machine safely.

This module defines a tiny line-delimited JSON protocol so a local agent can
send :class:`~BrainSimulationSystem.environment.tool_bridge.ToolEnvironmentBridge`
actions to a remote host that runs a lightweight listener.

Security posture:
- Remote control is opt-in; callers should enforce allowlists.
- The server can require a shared auth token and restrict client IPs.
- Both client and server limit request/response sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import socketserver
import threading
import time
from typing import Any, Dict, Optional, Tuple


class RemoteToolError(RuntimeError):
    """Base class for remote tool errors."""


class RemoteToolConnectionError(RemoteToolError):
    """Raised when the remote endpoint cannot be reached or responds incorrectly."""


class RemoteToolAuthError(RemoteToolError):
    """Raised when the remote endpoint rejects authentication."""


class RemoteToolProtocolError(RemoteToolError):
    """Raised when the remote endpoint returns an invalid protocol message."""


def _json_dumps(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _json_loads(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


@dataclass(frozen=True)
class RemoteToolResponse:
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    info: Dict[str, Any]


class RemoteToolClient:
    """Simple one-shot client for the TCP JSONL remote tool protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        auth_token: Optional[str] = None,
        connect_timeout_s: float = 3.0,
        timeout_s: float = 10.0,
        max_response_bytes: int = 2_000_000,
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._auth_token = auth_token
        self._connect_timeout_s = float(connect_timeout_s)
        self._timeout_s = float(timeout_s)
        self._max_response_bytes = max(1_024, int(max_response_bytes))

    @property
    def endpoint(self) -> Tuple[str, int]:
        return self._host, self._port

    def request(self, payload: Dict[str, Any], *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        message = dict(payload)
        if self._auth_token and "token" not in message:
            message["token"] = self._auth_token
        message.setdefault("ts", time.time())

        try:
            with socket.create_connection(self.endpoint, timeout=self._connect_timeout_s) as sock:
                sock.settimeout(max(0.1, timeout))
                sock.sendall(_json_dumps(message))
                data = b""
                while not data.endswith(b"\n"):
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if len(data) > self._max_response_bytes:
                        raise RemoteToolProtocolError("response_too_large")
        except RemoteToolError:
            raise
        except Exception as exc:
            raise RemoteToolConnectionError(f"remote_connection_failed:{exc!r}") from exc

        if not data:
            raise RemoteToolConnectionError("empty_response")
        if not data.endswith(b"\n"):
            raise RemoteToolProtocolError("response_missing_newline")

        try:
            response = _json_loads(data)
        except Exception as exc:
            raise RemoteToolProtocolError(f"invalid_json_response:{exc!r}") from exc

        if not isinstance(response, dict):
            raise RemoteToolProtocolError("response_not_dict")
        if response.get("error") == "auth_failed":
            raise RemoteToolAuthError("auth_failed")
        return response

    def step(self, action: Dict[str, Any], *, timeout_s: Optional[float] = None) -> RemoteToolResponse:
        response = self.request({"method": "step", "action": action}, timeout_s=timeout_s)
        return _parse_step_response(response)

    def reset(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        response = self.request({"method": "reset"}, timeout_s=timeout_s)
        observation = response.get("observation")
        if not isinstance(observation, dict):
            raise RemoteToolProtocolError("reset_response_missing_observation")
        return observation


def _parse_step_response(response: Dict[str, Any]) -> RemoteToolResponse:
    if response.get("error"):
        info = response.get("info") if isinstance(response.get("info"), dict) else {}
        raise RemoteToolProtocolError(str(response.get("error") or "remote_error"))

    observation = response.get("observation")
    info = response.get("info")
    reward = response.get("reward")
    terminated = response.get("terminated")

    if not isinstance(observation, dict):
        raise RemoteToolProtocolError("missing_observation")
    if not isinstance(info, dict):
        raise RemoteToolProtocolError("missing_info")
    if not isinstance(reward, (int, float)):
        raise RemoteToolProtocolError("missing_reward")
    if not isinstance(terminated, bool):
        raise RemoteToolProtocolError("missing_terminated")

    return RemoteToolResponse(
        observation=observation,
        reward=float(reward),
        terminated=bool(terminated),
        info=info,
    )


class _RemoteToolTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _RemoteToolRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:  # pragma: no cover - exercised through integration tests
        server = self.server
        allowed_ips = getattr(server, "allowed_client_ips", None)
        max_request_bytes = int(getattr(server, "max_request_bytes", 1_000_000))

        client_ip = str(self.client_address[0])
        if allowed_ips and client_ip not in allowed_ips:
            self.wfile.write(_json_dumps({"error": "client_not_allowed"}))
            return

        while True:
            line = self.rfile.readline(max_request_bytes + 1)
            if not line:
                break
            if len(line) > max_request_bytes:
                self.wfile.write(_json_dumps({"error": "request_too_large"}))
                break

            try:
                request = _json_loads(line)
            except Exception as exc:
                self.wfile.write(_json_dumps({"error": "invalid_json", "info": {"exception": repr(exc)}}))
                continue

            if not isinstance(request, dict):
                self.wfile.write(_json_dumps({"error": "request_not_dict"}))
                continue

            required_token = getattr(server, "auth_token", None)
            if required_token is not None:
                provided = request.get("token")
                if str(provided or "") != str(required_token):
                    self.wfile.write(_json_dumps({"error": "auth_failed"}))
                    continue

            method = request.get("method")
            if method == "reset":
                try:
                    observation = server.tool.reset()
                except Exception as exc:
                    self.wfile.write(_json_dumps({"error": "reset_failed", "info": {"exception": repr(exc)}}))
                    continue
                if not isinstance(observation, dict):
                    observation = {"text": str(observation)}
                self.wfile.write(_json_dumps({"observation": observation}))
                continue

            if method == "step":
                action = request.get("action")
                if not isinstance(action, dict):
                    self.wfile.write(_json_dumps({"error": "action_must_be_dict"}))
                    continue
                try:
                    observation, reward, terminated, info = server.tool.step(action)
                except Exception as exc:
                    self.wfile.write(_json_dumps({"error": "step_failed", "info": {"exception": repr(exc)}}))
                    continue
                if not isinstance(observation, dict):
                    observation = {"text": str(observation)}
                if not isinstance(info, dict):
                    info = {"info": str(info)}
                self.wfile.write(
                    _json_dumps(
                        {
                            "observation": observation,
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "info": info,
                        }
                    )
                )
                continue

            self.wfile.write(_json_dumps({"error": "unknown_method", "info": {"method": method}}))


class RemoteToolServer:
    """Threaded TCP server exposing a tool-like object (reset/step) over JSONL."""

    def __init__(
        self,
        tool: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        auth_token: Optional[str] = None,
        allowed_client_ips: Optional[Tuple[str, ...]] = ("127.0.0.1", "::1"),
        max_request_bytes: int = 1_000_000,
    ) -> None:
        self._server = _RemoteToolTCPServer((str(host), int(port)), _RemoteToolRequestHandler)
        self._server.tool = tool
        self._server.auth_token = auth_token
        self._server.allowed_client_ips = tuple(allowed_client_ips or ())
        self._server.max_request_bytes = int(max_request_bytes)
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            return

        def _run() -> None:
            self._server.serve_forever(poll_interval=0.2)

        thread = threading.Thread(target=_run, name="RemoteToolServer", daemon=True)
        thread.start()
        self._thread = thread

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "RemoteToolServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


__all__ = [
    "RemoteToolError",
    "RemoteToolConnectionError",
    "RemoteToolAuthError",
    "RemoteToolProtocolError",
    "RemoteToolResponse",
    "RemoteToolClient",
    "RemoteToolServer",
]
