from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
from typing import Any, Deque, DefaultDict, Dict, Iterable, List, Optional
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ConversationTurn:
    id: str
    role: str
    message: str
    timestamp: str
    channel: str = "chat"
    session_id: str = ""
    attachments: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "message": self.message,
            "timestamp": self.timestamp,
            "channel": self.channel,
            "session_id": self.session_id,
        }
        if self.attachments:
            payload["attachments"] = self.attachments
        return payload


class ConversationStore:
    def __init__(
        self,
        *,
        max_turns: int = 500,
        max_session_turns: int = 80,
        persist_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._turns: Deque[ConversationTurn] = deque(maxlen=int(max_turns))
        self._sessions: DefaultDict[str, Deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=int(max_session_turns))
        )
        self._lock = asyncio.Lock()
        self._persist_path = Path(persist_path).resolve() if persist_path else None
        if self._persist_path is not None:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, turn: ConversationTurn) -> None:
        async with self._lock:
            self._turns.append(turn)
            if turn.session_id:
                self._sessions[turn.session_id].append(turn)
            if self._persist_path is not None:
                try:
                    with self._persist_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(turn.as_dict(), ensure_ascii=False) + "\n")
                except Exception:
                    pass

    async def history(self, *, limit: int = 100, session_id: str | None = None) -> List[Dict[str, Any]]:
        async with self._lock:
            if session_id:
                turns = list(self._sessions.get(session_id, deque()))
            else:
                turns = list(self._turns)
        turns = list(reversed(turns))[: max(0, int(limit))]
        return [turn.as_dict() for turn in turns]


class InterruptState:
    def __init__(self) -> None:
        self._flag = asyncio.Event()

    def set(self, active: bool) -> None:
        if active:
            self._flag.set()
        else:
            self._flag.clear()

    def active(self) -> bool:
        return self._flag.is_set()


def mount_ui_api(app: Any) -> None:
    """Attach lightweight UI endpoints to the Forge FastAPI app.

    This keeps the Flutter dashboard self-contained when served from
    ``backend.forge.forge.app``.
    """

    try:
        from fastapi import Body, HTTPException, Request, WebSocket
    except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
        return

    if getattr(app.state, "_ui_api_mounted", False):
        return

    conversation_log_path = os.getenv("UI_CONVERSATION_LOG_PATH")
    store = ConversationStore(
        persist_path=conversation_log_path,
    )
    interrupt_state = InterruptState()

    app.state._ui_api_mounted = True
    app.state.ui_conversation_store = store
    app.state.ui_interrupt_state = interrupt_state
    app.state.ui_conversation_clients: set[WebSocket] = set()

    def _get_llm() -> Any:
        llm = getattr(app.state, "ui_llm_service", None)
        if llm is not None:
            return llm
        from BrainSimulationSystem.integration.llm_service import LLMService

        llm = LLMService()
        app.state.ui_llm_service = llm
        return llm

    def _get_knowledge_base() -> Any | None:
        kb = getattr(app.state, "ui_knowledge_base", None)
        if kb is not None:
            return kb
        try:
            from modules.knowledge import KnowledgeBase
        except Exception:
            return None
        if KnowledgeBase is None:
            return None
        try:
            kb = KnowledgeBase.from_env()
        except Exception:
            return None
        app.state.ui_knowledge_base = kb
        return kb

    async def _broadcast_conversation(turn: Dict[str, Any]) -> None:
        clients: set[WebSocket] = app.state.ui_conversation_clients
        if not clients:
            return
        dead: List[WebSocket] = []
        for client in clients:
            try:
                await client.send_json(turn)
            except Exception:
                dead.append(client)
        for client in dead:
            clients.discard(client)

    @app.get("/api/conversations/history")
    async def conversations_history(limit: int = 200, session_id: str | None = None) -> List[Dict[str, Any]]:
        return await store.history(limit=limit, session_id=session_id)

    @app.post("/api/chat")
    async def chat(
        request: Request,
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        session_id = payload.get("session_id") or payload.get("sessionId") or str(uuid4())
        message = (payload.get("message") or "").strip()
        messages = payload.get("messages")

        if messages is None:
            if not message:
                return {"session_id": session_id, "reply": "", "meta": {"empty": True}}
            messages = [{"role": "user", "content": message}]

        audio_frames = payload.get("audio_frames") or payload.get("audioFrames") or payload.get("audio_features")
        if audio_frames is not None:
            try:
                for msg in reversed(messages):
                    if (msg.get("role") or "").lower() == "user":
                        msg.setdefault("audio_frames", audio_frames)
                        break
            except Exception:
                pass

        image_base64 = payload.get("image_base64") or payload.get("imageBase64")
        image_mime = payload.get("image_mime") or payload.get("imageMime") or "image/png"
        attachments: Dict[str, Any] | None = None
        if isinstance(image_base64, str) and image_base64.strip():
            stripped = image_base64.strip()
            if stripped.startswith("data:"):
                match = re.match(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", stripped)
                if match:
                    image_mime = match.group("mime") or image_mime
                    stripped = match.group("data")
            max_b64_len = int(os.getenv("UI_MAX_IMAGE_BASE64_CHARS", "3000000"))
            if len(stripped) > max_b64_len:
                stripped = ""
            if stripped:
                attachments = {"image": {"base64": stripped, "mime": image_mime}}

        user_turn_text = message
        if not user_turn_text:
            try:
                for msg in reversed(messages):
                    if (msg.get("role") or "").lower() == "user":
                        user_turn_text = msg.get("content", "")
                        break
            except Exception:
                user_turn_text = ""

        user_turn = ConversationTurn(
            id=str(uuid4()),
            role="user",
            message=user_turn_text,
            timestamp=_utc_now_iso(),
            session_id=session_id,
            attachments=attachments,
        )
        await store.append(user_turn)
        await _broadcast_conversation(user_turn.as_dict())

        # Optional: inject environment context into the prompt so planning can
        # adapt to resource constraints (opt-in via env var).
        if _parse_bool(os.getenv("UI_INCLUDE_ENV_CONTEXT"), default=False):
            try:
                from modules.environment.environment_adapter import EnvironmentAdapter

                adapter = getattr(app.state, "ui_environment_adapter", None)
                if adapter is None:
                    adapter = EnvironmentAdapter(worker_id="forge-ui-env", event_bus=None)
                    app.state.ui_environment_adapter = adapter
                env_prompt = adapter.environment_prompt()
                if isinstance(env_prompt, str) and env_prompt.strip():
                    inserted = False
                    for msg in messages:
                        if (msg.get("role") or "").lower() == "system":
                            msg["content"] = (msg.get("content") or "") + "\n\n" + env_prompt
                            inserted = True
                            break
                    if not inserted:
                        messages.insert(0, {"role": "system", "content": env_prompt})
            except Exception:
                pass

        llm = _get_llm()
        response = llm.chat(
            messages,
            temperature=float(payload.get("temperature", 0.2)),
            max_tokens=int(payload.get("max_tokens", payload.get("maxTokens", 512))),
            response_format=payload.get("response_format") or payload.get("responseFormat"),
        )

        reply_text = response.text
        confidence = response.meta.get("confidence")
        enable_clarify = payload.get("enable_clarify", True)
        clarify_threshold = float(os.getenv("UI_CLARIFY_THRESHOLD", "0.4"))
        if enable_clarify and isinstance(confidence, (int, float)) and confidence < clarify_threshold:
            reply_text = (
                "我可能没完全理解你的意思。你希望我：\n"
                "1) 给出定义/解释\n"
                "2) 给出步骤/方案\n"
                "3) 直接给结论\n"
                "你更偏向哪一种？也可以补充目标或约束条件。"
            )
            response.meta = {**(response.meta or {}), "clarify": True, "clarify_threshold": clarify_threshold}

        agent_turn = ConversationTurn(
            id=str(uuid4()),
            role="agent",
            message=reply_text,
            timestamp=_utc_now_iso(),
            session_id=session_id,
        )
        await store.append(agent_turn)
        await _broadcast_conversation(agent_turn.as_dict())

        return {
            "session_id": session_id,
            "reply": reply_text,
            "meta": response.meta,
            "latency": response.latency,
            "attachments_received": bool(attachments),
        }

    @app.post("/api/memory/store")
    async def memory_store(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        kb = _get_knowledge_base()
        if kb is None:
            raise HTTPException(status_code=503, detail="KnowledgeBase unavailable")

        category = payload.get("category") or payload.get("type") or "general"
        content = payload.get("content") or payload.get("text") or ""
        if not str(content).strip():
            raise HTTPException(status_code=400, detail="content is required")

        tags = payload.get("tags")
        if not isinstance(tags, list):
            tags = []
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        entry_id = kb.save_memory(
            str(category),
            str(content),
            tags=[str(t) for t in tags if str(t).strip()],
            metadata=metadata,
            confidence=payload.get("confidence"),
            status=str(payload.get("status") or "active"),
        )
        return {"id": entry_id}

    @app.get("/api/memory/query")
    async def memory_query(
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> Dict[str, Any]:
        kb = _get_knowledge_base()
        if kb is None:
            raise HTTPException(status_code=503, detail="KnowledgeBase unavailable")
        items = kb.query_memory(query, top_k=max(1, int(top_k)), category=category)
        results: List[Dict[str, Any]] = []
        for item in items:
            if hasattr(item, "__dict__"):
                results.append(dict(item.__dict__))
            else:
                results.append(dict(item))
        return {"count": len(results), "items": results}

    @app.get("/api/memory/recent")
    async def memory_recent(limit: int = 20, category: str | None = None) -> Dict[str, Any]:
        kb = _get_knowledge_base()
        if kb is None:
            raise HTTPException(status_code=503, detail="KnowledgeBase unavailable")
        items = kb.recent(limit=max(0, int(limit)), category=category)
        results: List[Dict[str, Any]] = []
        for item in items:
            if hasattr(item, "__dict__"):
                results.append(dict(item.__dict__))
            else:
                results.append(dict(item))
        return {"count": len(results), "items": results}

    @app.post("/api/control/interrupt")
    async def control_interrupt(active: bool = True) -> Dict[str, Any]:
        interrupt_state.set(bool(active))
        return {"interrupt": interrupt_state.active()}

    @app.get("/api/control/interrupt")
    async def interrupt_status() -> Dict[str, Any]:
        return {"interrupt": interrupt_state.active()}

    @app.get("/api/logs/system")
    async def system_logs(level: str | None = None, limit: int = 200) -> List[Dict[str, Any]]:  # noqa: ARG001
        return []

    @app.websocket("/ws/logs/system")
    async def ws_logs_system(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.receive_text()
        except Exception:
            return

    @app.websocket("/ws/logs/conversation")
    async def ws_conversation(ws: WebSocket) -> None:
        await ws.accept()
        app.state.ui_conversation_clients.add(ws)
        try:
            while True:
                await ws.receive_text()
        except Exception:
            pass
        finally:
            app.state.ui_conversation_clients.discard(ws)

    # Optional dashboard endpoints (stubs) so the Flutter UI doesn't 404.
    @app.get("/api/agents/status")
    async def agents_status() -> List[Dict[str, Any]]:
        return []

    @app.websocket("/ws/agents/status")
    async def ws_agents_status(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.receive_text()
        except Exception:
            return

    @app.get("/api/learning/jobs")
    async def learning_jobs() -> List[Dict[str, Any]]:
        return []

    @app.get("/api/learning/jobs/{job_id}")
    async def learning_job(job_id: str) -> Dict[str, Any]:
        return {"id": job_id, "name": "unknown", "status": "unknown", "progress": 0.0, "iteration": 0, "totalIterations": 0}

    @app.get("/api/memory/entries")
    async def memory_entries(sort: str | None = None, search: str | None = None, limit: int = 200) -> List[Dict[str, Any]]:  # noqa: ARG001
        return []

    @app.delete("/api/memory/entries")
    async def memory_clear_all() -> Dict[str, Any]:
        return {"cleared": True}

    @app.delete("/api/memory/entries/{entry_id}")
    async def memory_delete(entry_id: str) -> Dict[str, Any]:
        return {"deleted": entry_id}

    @app.get("/api/memory/entries/stats")
    async def memory_stats() -> Dict[str, int]:
        return {"total": 0, "promoted": 0}

    @app.get("/api/knowledge/entries")
    async def knowledge_entries(search: str | None = None) -> List[Dict[str, Any]]:  # noqa: ARG001
        return []

    @app.get("/api/knowledge/entries/{entry_id}")
    async def knowledge_entry(entry_id: str) -> Dict[str, Any]:
        return {
            "id": entry_id,
            "title": "",
            "summary": "",
            "content": "",
            "tags": [],
            "source": "unknown",
            "updatedAt": _utc_now_iso(),
        }

    @app.post("/api/knowledge/entries")
    async def knowledge_create(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        entry_id = str(uuid4())
        return {
            "id": entry_id,
            "title": payload.get("title", ""),
            "summary": payload.get("summary", ""),
            "content": payload.get("content", ""),
            "tags": list(payload.get("tags") or []),
            "source": payload.get("source") or "user",
            "updatedAt": _utc_now_iso(),
        }

    @app.put("/api/knowledge/entries/{entry_id}")
    async def knowledge_update(entry_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        return {
            "id": entry_id,
            "title": payload.get("title", ""),
            "summary": payload.get("summary", ""),
            "content": payload.get("content", ""),
            "tags": list(payload.get("tags") or []),
            "source": payload.get("source") or "user",
            "updatedAt": _utc_now_iso(),
        }
