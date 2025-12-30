"""Flask blueprints that expose the Brain API surface."""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - minimal environments without numpy
    np = None  # type: ignore
from flask import Blueprint, jsonify, request

from BrainSimulationSystem.core.module_interface import ModuleTopic
from BrainSimulationSystem.core.physiological_regions import BrainRegion


def _decode_frame(payload: Any, meta: Optional[Dict[str, Any]] = None) -> Any:
    if np is None:
        raise ValueError("numpy is required for stimulus decoding")
    meta = meta or {}
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32)
    if isinstance(payload, dict):
        data = payload.get("data")
        encoding = payload.get("encoding", meta.get("encoding", "json"))
        dtype = np.dtype(payload.get("dtype", meta.get("dtype", "float32")))
        shape = payload.get("shape") or meta.get("shape")
        if encoding == "base64":
            if data is None:
                raise ValueError("missing base64 data")
            raw = base64.b64decode(data)
            arr = np.frombuffer(raw, dtype=dtype)
            if shape:
                arr = arr.reshape(shape)
            return arr.astype(np.float32, copy=False)
        return np.asarray(data, dtype=dtype).astype(np.float32, copy=False)
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            raise ValueError("invalid stimulus string") from exc
        return np.asarray(parsed, dtype=np.float32)
    raise ValueError("unsupported stimulus format")


def create_sensory_blueprint(api: "BrainAPI") -> Blueprint:
    bp = Blueprint("sensory", __name__)

    @bp.route("/vision", methods=["POST"])
    def post_vision_input():
        if np is None:
            return jsonify({"error": "numpy is required for visual stimuli"}), 500
        if request.mimetype != "application/json":
            return jsonify({"error": "expected JSON body"}), 400
        data = request.get_json(silent=True) or {}
        stimulus = data.get("stimulus")
        if stimulus is None:
            return jsonify({"error": "missing stimulus field"}), 400
        try:
            frame = _decode_frame(stimulus, data)
        except (ValueError, binascii.Error) as exc:
            return jsonify({"error": f"invalid stimulus: {exc}"}), 400
        region = str(data.get("region", BrainRegion.PRIMARY_VISUAL_CORTEX.value))
        if region not in api.region_keys:
            region = BrainRegion.PRIMARY_VISUAL_CORTEX.value
        sticky = bool(data.get("sticky", False))
        api.store_pending_input(region, {"visual_stimulus": frame}, sticky=sticky)
        payload = {
            "region": region,
            "shape": list(frame.shape),
            "max": float(frame.max()) if frame.size else 0.0,
            "min": float(frame.min()) if frame.size else 0.0,
        }
        api.broadcast_event(ModuleTopic.SENSORY_VISUAL.value, payload)
        return jsonify({"status": "accepted", "region": region, "shape": payload["shape"]}), 202

    @bp.route("/motor", methods=["POST"])
    def post_motor_command():
        if request.mimetype != "application/json":
            return jsonify({"error": "expected JSON body"}), 400
        data = request.get_json(silent=True) or {}
        commands = data.get("commands")
        if commands is None:
            return jsonify({"error": "missing commands field"}), 400
        sticky = bool(data.get("sticky", False))
        region = str(data.get("region", BrainRegion.PRIMARY_MOTOR_CORTEX.value))
        if region not in api.region_keys:
            region = BrainRegion.PRIMARY_MOTOR_CORTEX.value
        try:
            if isinstance(commands, dict):
                plan = {str(k): float(v) for k, v in commands.items()}
            else:
                if np is None:
                    plan = [float(value) for value in commands]
                else:
                    plan = np.asarray(commands, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"invalid commands: {exc}"}), 400
        api.store_pending_input(region, {"motor_plan": plan}, sticky=sticky)
        summary: Dict[str, Any] = {"region": region}
        if isinstance(plan, dict):
            summary["keys"] = list(plan.keys())
        else:
            length = len(plan) if np is None else int(plan.size)
            summary["length"] = length
        api.broadcast_event(ModuleTopic.MOTOR_PLAN.value, summary)
        return jsonify({"status": "accepted", **summary}), 202

    @bp.route("/stream/topics", methods=["GET"])
    def get_stream_topics():
        return jsonify({"topics": [topic.value for topic in ModuleTopic]})

    return bp


def create_cognitive_blueprint(api: "BrainAPI") -> Blueprint:
    bp = Blueprint("cognitive", __name__)

    @bp.route("/cognitive/state", methods=["GET"])
    def get_cognitive_state():
        state = {
            "cognitive_state": api.controller.state.name,
            "state_history": [s.name for s in api.controller.state_history[-10:]],
            "neuromodulators": api.controller.neuromodulators,
        }
        return jsonify(state)

    @bp.route("/cognitive/process", methods=["POST"])
    def process_cognitive_input():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "无效的输入数据"}), 400
        try:
            result = api.controller.process(data)
            return jsonify(result)
        except Exception as exc:  # pragma: no cover - controller specific errors
            return jsonify({"error": str(exc)}), 500

    @bp.route("/attention/focus", methods=["GET"])
    def get_attention_focus():
        attention = api.controller.components.get("attention")
        if attention is None:
            return jsonify({"error": "注意力系统不可用"}), 404
        focus = getattr(attention, "focus", [])
        return jsonify({"focus": focus})

    @bp.route("/attention/params", methods=["GET", "PUT"])
    def attention_params():
        attention = api.controller.components.get("attention")
        if attention is None:
            return jsonify({"error": "注意力系统不可用"}), 404
        if request.method == "GET":
            params = {k: getattr(attention, k) for k in dir(attention) if not k.startswith("_")}
            return jsonify(params)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "无效的参数数据"}), 400
        for key, value in data.items():
            if hasattr(attention, key):
                setattr(attention, key, value)
        return jsonify({"status": "updated"})

    @bp.route("/memory/content", methods=["GET"])
    def get_memory_content():
        memory = api.controller.components.get("working_memory")
        if memory is None:
            return jsonify({"error": "工作记忆系统不可用"}), 404
        get_all = getattr(memory, "get_all_items", None)
        if callable(get_all):
            return jsonify({"memory": get_all()})
        return jsonify({"memory": getattr(memory, "memory_items", {})})

    @bp.route("/memory/item/<key>", methods=["GET", "PUT", "DELETE"])
    def memory_item(key: str):
        memory = api.controller.components.get("working_memory")
        if memory is None:
            return jsonify({"error": "工作记忆系统不可用"}), 404
        if request.method == "GET":
            getter = getattr(memory, "get_item", None)
            if callable(getter):
                item = getter(key)
            else:
                item = getattr(memory, "memory_items", {}).get(key)
            if item is None:
                return jsonify({"error": "未找到指定记忆项"}), 404
            return jsonify({"key": key, "value": item})
        if request.method == "PUT":
            data = request.get_json(silent=True)
            if not data or "value" not in data:
                return jsonify({"error": "无效的记忆项数据"}), 400
            value = data["value"]
            priority = data.get("priority", 0.5)
            store = getattr(memory, "_store_item", None)
            if callable(store):
                store(key, value, priority)
            else:  # pragma: no cover - fallback for unconventional memory impls
                getattr(memory, "memory_items", {})[key] = value
            return jsonify({"status": "success"})
        delete = getattr(memory, "_delete_item", None)
        if not callable(delete):
            return jsonify({"error": "不支持删除记忆项"}), 501
        delete(key)
        return jsonify({"status": "success"})

    @bp.route("/neuromodulators", methods=["GET", "PUT"])
    def neuromodulators():
        if request.method == "GET":
            return jsonify(api.controller.neuromodulators)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "无效的神经调质数据"}), 400
        for name, level in data.items():
            if name in api.controller.neuromodulators:
                api.controller.set_neuromodulator(name, float(level))
        return jsonify({"status": "success"})

    return bp


def create_simulation_blueprint(api: "BrainAPI") -> Blueprint:
    bp = Blueprint("simulation", __name__)

    @bp.route("/info", methods=["GET"])
    def get_system_info():
        info = {
            "name": "大脑模拟系统",
            "version": "1.0.0",
            "components": list(api.controller.components.keys()),
            "status": "running" if api.simulation.is_running else "idle",
        }
        if api.brain_simulation:
            info["brain_simulation"] = {
                "is_running": bool(getattr(api.brain_simulation, "is_running", False)),
                "current_time": api.to_serializable(getattr(api.brain_simulation, "current_time", 0.0)),
            }
        return jsonify(info)

    @bp.route("/brain/status", methods=["GET"])
    def get_brain_status():
        if not api.brain_simulation:
            return jsonify({"error": "大脑模拟器未初始化"}), 503
        config = getattr(api.brain_simulation, "config", {})
        if isinstance(config, dict):
            config_keys = list(config.keys())
        else:
            config_keys = []
        status = {
            "is_running": bool(getattr(api.brain_simulation, "is_running", False)),
            "current_time": api.to_serializable(getattr(api.brain_simulation, "current_time", 0.0)),
            "config_keys": config_keys,
            "api_running": api.is_serving,
        }
        return jsonify(status)

    @bp.route("/brain/module_bus", methods=["GET"])
    def get_latest_module_bus():
        bus = api.latest_module_bus
        if bus is None:
            return jsonify({"error": "no module bus snapshot available"}), 404
        return jsonify(bus)

    @bp.route("/brain/last_step", methods=["GET"])
    def get_last_step():
        step = api.latest_step_serialized
        if step is None:
            return jsonify({"error": "no simulation step recorded"}), 404
        return jsonify(step)

    @bp.route("/simulation/start", methods=["POST"])
    def start_simulation():
        if api.simulation.is_running:
            return jsonify({"error": "模拟已在运行中"}), 400
        data = request.get_json(silent=True) or {}
        try:
            result = api.simulation.start(
                steps=data.get("steps"),
                interval=data.get("interval"),
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @bp.route("/simulation/stop", methods=["POST"])
    def stop_simulation():
        if not api.simulation.is_running:
            return jsonify({"error": "模拟未在运行"}), 400
        result = api.simulation.stop()
        return jsonify(result)

    @bp.route("/simulation/status", methods=["GET"])
    def simulation_status():
        return jsonify(api.simulation.status())

    @bp.route("/simulation/results", methods=["GET"])
    def simulation_results():
        page = int(request.args.get("page", 0))
        page_size = int(request.args.get("page_size", 10))
        return jsonify(api.simulation.paginated_results(page, page_size))

    @bp.route("/brain/step", methods=["POST"])
    def step_brain_simulation():
        if not api.brain_simulation:
            return jsonify({"error": "brain simulation not initialised"}), 503
        payload = request.get_json(silent=True) or {}
        request_inputs = payload.get("inputs", {})
        dt = float(payload.get("dt", 1.0))
        pending_inputs, retained_inputs = api.prepare_pending_inputs()
        call_inputs: Dict[str, Any] = dict(pending_inputs)
        if isinstance(request_inputs, dict):
            for region, value in request_inputs.items():
                if (
                    region in call_inputs
                    and isinstance(call_inputs[region], dict)
                    and isinstance(value, dict)
                ):
                    call_inputs[region].update(value)
                else:
                    call_inputs[region] = value
        api.replace_pending_inputs(retained_inputs)
        try:
            serial_result = api.execute_brain_step(call_inputs, dt)
            return jsonify(serial_result)
        except Exception as exc:  # pragma: no cover - brain simulation level failures
            return jsonify({"error": str(exc)}), 500

    return bp


__all__ = [
    "create_sensory_blueprint",
    "create_cognitive_blueprint",
    "create_simulation_blueprint",
]
