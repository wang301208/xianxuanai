"""Brain simulation REST API wiring built on Flask and Flask-Sockets."""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Set, Tuple

try:  # pragma: no cover - optional dependency guard
    import numpy as np
except ImportError:  # pragma: no cover - minimal environments without numpy
    np = None  # type: ignore
from flask import Flask
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from BrainSimulationSystem.api.routes import (
    create_cognitive_blueprint,
    create_sensory_blueprint,
    create_simulation_blueprint,
)
from BrainSimulationSystem.api.streaming import StreamingHub
from BrainSimulationSystem.core.cognitive_services import CognitiveControllerFactory
from BrainSimulationSystem.core.physiological_regions import BrainRegion
from BrainSimulationSystem.core.simulation_orchestrator import SimulationOrchestrator


class BrainAPI:
    """Entry point wiring together the Flask app, routes and services."""

    def __init__(
        self,
        brain_simulation: Optional[Any] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        enable_cors: bool = False,
        *,
        controller_factory: Optional[CognitiveControllerFactory] = None,
        streaming_hub: Optional[StreamingHub] = None,
        simulation_orchestrator_cls: type = SimulationOrchestrator,
    ) -> None:
        self.app = Flask(__name__)
        self.sockets = Sockets(self.app)
        self.streaming = streaming_hub or StreamingHub()
        self.sockets.route("/ws/stream")(self.streaming.create_ws_handler(self._initial_ws_event))

        self._latest_step: Optional[Any] = None
        self._latest_step_serialized: Optional[Dict[str, Any]] = None
        self._latest_module_bus: Optional[Dict[str, Any]] = None
        self._pending_inputs: Dict[str, Dict[str, Any]] = {}
        self._region_keys: Set[str] = {region.value for region in BrainRegion}

        self.brain_simulation = brain_simulation
        self.host = host
        self.port = port
        self.enable_cors = enable_cors

        if enable_cors:
            try:  # pragma: no cover - optional dependency
                from flask_cors import CORS  # type: ignore

                CORS(self.app)
            except Exception:  # pragma: no cover - defensive log path
                self.app.logger.warning("未找到flask_cors，CORS功能被禁用")

        factory = controller_factory or CognitiveControllerFactory()
        self.controller = factory.create()
        self.simulation = simulation_orchestrator_cls(self.controller)

        self._server: Optional[pywsgi.WSGIServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_lock = threading.Lock()
        self._is_serving = False

        self._register_blueprints()

    # ------------------------------------------------------------------
    # Blueprint wiring
    def _register_blueprints(self) -> None:
        self.app.register_blueprint(create_sensory_blueprint(self), url_prefix="/api")
        self.app.register_blueprint(create_cognitive_blueprint(self), url_prefix="/api")
        self.app.register_blueprint(create_simulation_blueprint(self), url_prefix="/api")

    # ------------------------------------------------------------------
    # Public helpers used by blueprint closures
    @property
    def region_keys(self) -> Set[str]:
        return self._region_keys

    @property
    def latest_module_bus(self) -> Optional[Dict[str, Any]]:
        return self._latest_module_bus

    @property
    def latest_step_serialized(self) -> Optional[Dict[str, Any]]:
        return self._latest_step_serialized

    @property
    def is_serving(self) -> bool:
        return self._is_serving

    def store_pending_input(self, region: str, values: Dict[str, Any], *, sticky: bool) -> None:
        entry = self._pending_inputs.setdefault(region, {})
        once_keys: Set[str] = entry.setdefault("__once_keys__", set())  # type: ignore[assignment]
        entry.update(values)
        if sticky:
            once_keys.difference_update(values.keys())
        else:
            once_keys.update(values.keys())
        if not once_keys:
            entry.pop("__once_keys__", None)

    def prepare_pending_inputs(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        call_inputs: Dict[str, Any] = {}
        retained_inputs: Dict[str, Dict[str, Any]] = {}
        for region, entry in self._pending_inputs.items():
            once_keys = set(entry.get("__once_keys__", set()))
            payload = {k: v for k, v in entry.items() if k != "__once_keys__"}
            if payload:
                call_inputs[region] = payload
                retained = dict(payload)
                for key in once_keys:
                    retained.pop(key, None)
                if retained:
                    retained_inputs[region] = retained
        return call_inputs, retained_inputs

    def replace_pending_inputs(self, pending: Dict[str, Dict[str, Any]]) -> None:
        self._pending_inputs = pending

    def broadcast_event(self, event: str, payload: Any) -> None:
        self.streaming.broadcast(event, payload, transform=self.to_serializable)

    def execute_brain_step(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        if not self.brain_simulation:
            raise RuntimeError("brain simulation not initialised")
        result = self.brain_simulation.step(inputs, dt)
        serial_result = self.to_serializable(result)
        self._latest_step = result
        self._latest_step_serialized = serial_result
        self._latest_module_bus = serial_result.get("module_bus")

        summary = {
            "simulation_time": serial_result.get("simulation_time"),
            "statistics": serial_result.get("statistics"),
            "control": serial_result.get("control"),
            "sensorimotor": serial_result.get("sensorimotor"),
        }
        self.broadcast_event("brain.step", summary)

        module_bus = serial_result.get("module_bus")
        if isinstance(module_bus, dict):
            topics = module_bus.get("topics")
            if isinstance(topics, dict):
                for topic_name, messages in topics.items():
                    for message in messages:
                        self.broadcast_event(topic_name, message)
        if self._latest_module_bus is not None:
            self.broadcast_event("module.bus", self._latest_module_bus)
        return serial_result

    def to_serializable(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {key: self.to_serializable(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self.to_serializable(item) for item in data]
        if isinstance(data, tuple):
            return [self.to_serializable(item) for item in data]
        if np is not None and isinstance(data, np.ndarray):
            return data.tolist()
        if np is not None and isinstance(data, (np.integer, np.floating)):
            return data.item()
        return data

    # ------------------------------------------------------------------
    # Websocket initial payload supplier
    def _initial_ws_event(self) -> Optional[Dict[str, Any]]:
        if self._latest_step_serialized is None:
            return None
        return self.streaming.make_event("brain.step", self._latest_step_serialized)

    # ------------------------------------------------------------------
    # Server lifecycle management
    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debug: Optional[bool] = None,
        threaded: Optional[bool] = None,
        block: bool = True,
        **flask_kwargs: Any,
    ) -> None:
        host = host if host is not None else self.host
        port = port if port is not None else self.port
        debug = debug if debug is not None else False
        threaded = bool(threaded) if threaded is not None else False

        with self._server_lock:
            if self._server is not None:
                raise RuntimeError("API server already running")

            self.host = host
            self.port = port
            self.app.debug = debug

            if threaded:
                # gevent's WSGIServer requires a running hub; in unit tests we
                # start the API in a background thread, so we fall back to the
                # built-in Werkzeug WSGI server (HTTP only; websockets disabled).
                from werkzeug.serving import make_server  # type: ignore

                self._server = make_server(host, port, self.app, **flask_kwargs)  # type: ignore[assignment]
                self.port = int(getattr(self._server, "server_port", 0) or self._server.server_address[1])  # type: ignore[union-attr]
                self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)  # type: ignore[union-attr]
            else:
                self._server = pywsgi.WSGIServer(
                    (host, port),
                    self.app,
                    handler_class=WebSocketHandler,
                    **flask_kwargs,
                )
                self.port = self._server.server_port
                self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            self._is_serving = True
        if block:
            try:
                if self._server_thread:
                    self._server_thread.join()
            except KeyboardInterrupt:  # pragma: no cover - interactive stop
                pass

    def stop(self, timeout: float = 5.0) -> None:
        with self._server_lock:
            server = self._server
            server_thread = self._server_thread
            self._server = None
            self._server_thread = None
            self._is_serving = False

        if self.simulation.is_running:
            self.simulation.stop(timeout=timeout)

        if server:
            try:
                if hasattr(server, "stop"):
                    server.stop(timeout=timeout)
                elif hasattr(server, "shutdown"):
                    server.shutdown()
                if hasattr(server, "close"):
                    server.close()
                elif hasattr(server, "server_close"):
                    server.server_close()
            except Exception:
                pass
        if server_thread and server_thread.is_alive():
            server_thread.join(timeout=timeout)


BrainSimulationAPI = BrainAPI


def main() -> None:  # pragma: no cover - manual entry point
    api = BrainAPI(host="0.0.0.0", port=5000)
    print("启动大脑模拟系统API服务器...")
    api.run()


if __name__ == "__main__":  # pragma: no cover
    main()
