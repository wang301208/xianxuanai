from __future__ import annotations

"""Shared global workspace for broadcasting state between modules."""

import asyncio
import inspect
import threading
import heapq
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from uuid import uuid4


def _ensure_sequence(attention: Optional[Sequence[float] | float]) -> Optional[List[float]]:
    if attention is None:
        return None
    if isinstance(attention, Sequence) and not isinstance(attention, (str, bytes)):
        return [float(v) for v in attention]
    return [float(attention)]


def _normalise_tags(tags: Optional[Sequence[str] | str]) -> Tuple[str, ...]:
    if tags is None:
        return tuple()
    if isinstance(tags, (str, bytes)):
        return (str(tags),)
    return tuple(str(tag) for tag in tags if tag is not None)


@dataclass
class WorkspaceMessage:
    """Structured payload stored on the global workspace blackboard."""

    type: str
    source: str
    payload: Dict[str, Any]
    summary: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=lambda: time.time())
    id: str = field(default_factory=lambda: str(uuid4()))
    importance: float = 0.0
    attention: Optional[Tuple[float, ...]] = None
    sequence: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "payload": dict(self.payload),
            "timestamp": float(self.timestamp),
            "importance": float(self.importance),
        }
        if self.summary is not None:
            data["summary"] = self.summary
        if self.tags:
            data["tags"] = list(self.tags)
        if self.attention is not None:
            data["attention"] = list(self.attention)
        if self.sequence is not None:
            data["sequence"] = int(self.sequence)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceMessage":
        if "type" not in data or "source" not in data:
            raise ValueError("WorkspaceMessage requires 'type' and 'source'.")
        payload = data.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {"data": payload}
        attention = data.get("attention")
        attention_tuple: Optional[Tuple[float, ...]] = None
        if attention is not None:
            seq = _ensure_sequence(attention)
            attention_tuple = tuple(seq) if seq is not None else None
        return cls(
            type=str(data["type"]),
            source=str(data["source"]),
            payload=dict(payload),
            summary=str(data["summary"]) if data.get("summary") is not None else None,
            tags=_normalise_tags(data.get("tags")),
            timestamp=float(data.get("timestamp", time.time())),
            id=str(data.get("id", str(uuid4()))),
            importance=float(data.get("importance", 0.0)),
            attention=attention_tuple,
            sequence=int(data["sequence"]) if data.get("sequence") is not None else None,
        )


MessageFilter = Callable[[WorkspaceMessage], bool]


class GlobalWorkspace:
    """Registry that enables modules to share state and attention."""

    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}
        # Store per-module multi-head attention weights
        self._attention: Dict[str, List[float]] = {}
        self._state_subs: Dict[str, List[Callable[[Any], None]]] = {}
        # (module_a, module_b) -> cross-attention fusion hook
        self._cross_attn: Dict[Tuple[str, str], Callable[[Any, Any, Optional[List[float]], Optional[List[float]]], Tuple[Any, Optional[List[float]]]]] = {}
        # Synchronization lock for internal dictionaries
        self._lock = threading.RLock()
        # Candidate queue sorted by activation (negative for max-heap)
        self._candidates: List[Tuple[float, int, Tuple[str, Any, Optional[List[float]], str, Optional[List[str]], Optional[int], bool]]] = []
        self._candidate_counter = 0
        self._attention_threshold = 0.0
        self._messages: Deque[Tuple[int, WorkspaceMessage]] = deque(maxlen=512)
        self._message_seq = 0
        self._message_subs: List[Tuple[Callable[[WorkspaceMessage], Any], Optional[MessageFilter]]] = []

    # ------------------------------------------------------------------
    def register_module(self, name: str, module: Any) -> None:
        """Register *module* under *name* in the workspace."""
        with self._lock:
            self._modules[name] = module

    def unregister_module(self, name: str) -> None:
        """Remove *name* from the workspace and clear associated state."""

        with self._lock:
            self._modules.pop(name, None)
            self._state.pop(name, None)
            self._attention.pop(name, None)
            self._state_subs.pop(name, None)
            if self._cross_attn:
                self._cross_attn = {
                    key: handler
                    for key, handler in self._cross_attn.items()
                    if name not in key
                }

            if self._candidates:
                new_candidates: List[
                    Tuple[
                        float,
                        int,
                        Tuple[
                            str,
                            Any,
                            Optional[List[float]],
                            str,
                            Optional[List[str]],
                            Optional[int],
                            bool,
                        ],
                    ]
                ] = []
                for activation, counter, data in self._candidates:
                    sender, state, att_list, strategy, targets, k, allow_cross = data
                    if sender == name:
                        continue
                    filtered_targets = None
                    if targets is not None:
                        filtered_targets = [target for target in targets if target != name]
                        if not filtered_targets and strategy == "local":
                            continue
                    if filtered_targets is not None and filtered_targets is not targets:
                        data = (
                            sender,
                            state,
                            att_list,
                            strategy,
                            filtered_targets,
                            k,
                            allow_cross,
                        )
                    new_candidates.append((activation, counter, data))
                if len(new_candidates) != len(self._candidates):
                    self._candidates = new_candidates
                    heapq.heapify(self._candidates)
                else:
                    # Even if sizes match, targets might have changed; rebuild heap
                    self._candidates = new_candidates
                    heapq.heapify(self._candidates)

    def broadcast(
        self,
        sender: str,
        state: Any,
        attention: Optional[Sequence[float] | float] = None,
        *,
        strategy: str = "full",
        targets: Optional[List[str]] = None,
        k: Optional[int] = None,
        _allow_cross: bool = True,
    ) -> None:
        """Broadcast *state* from *sender* according to *strategy*.

        Parameters
        ----------
        sender:
            Name of the broadcasting module.
        state:
            Arbitrary payload to share.
        attention:
            Optional attention vector. A scalar will be promoted to a
            single-element list.
        strategy:
            ``"full"`` (default) broadcasts to all modules, ``"local"`` only to
            provided ``targets`` and ``"sparse"`` routes to the top ``k`` modules
            ranked by their current attention weights.
        targets:
            Optional explicit list of recipient modules. Required for
            ``strategy="local"``.
        k:
            Number of modules to target when ``strategy="sparse"``.
        _allow_cross:
            Internal flag to prevent recursive cross attention broadcasting.
        """

        with self._lock:
            self._state[sender] = state

            att_list: Optional[List[float]] = None
            if attention is not None:
                if isinstance(attention, Sequence) and not isinstance(attention, (str, bytes)):
                    att_list = [float(a) for a in attention]
                else:
                    att_list = [float(attention)]
                self._attention[sender] = att_list

            activation = sum(att_list) if att_list else 0.0
            data = (sender, state, att_list, strategy, targets, k, _allow_cross)
            heapq.heappush(self._candidates, (-activation, self._candidate_counter, data))
            self._candidate_counter += 1

        self._process_queue()

    def subscribe_state(self, name: str, handler: Callable[[Any], None]) -> None:
        """Invoke *handler* whenever *name* publishes new state."""

        with self._lock:
            self._state_subs.setdefault(name, []).append(handler)

    def subscribe_messages(
        self,
        handler: Callable[[WorkspaceMessage], Any],
        *,
        message_filter: Optional[MessageFilter] = None,
    ) -> Callable[[], None]:
        """Invoke *handler* whenever a matching workspace message is published."""

        with self._lock:
            entry = (handler, message_filter)
            self._message_subs.append(entry)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._message_subs.remove(entry)
                except ValueError:
                    pass

        return _unsubscribe

    def register_cross_attention(
        self,
        module_a: str,
        module_b: str,
        handler: Callable[[Any, Any, Optional[List[float]], Optional[List[float]]], Tuple[Any, Optional[List[float]]]],
    ) -> None:
        """Register a cross-attention fusion hook between two modules.

        When both modules have published state, *handler* is invoked with
        ``(state_a, state_b, attn_a, attn_b)`` and should return a tuple of
        ``(fused_state, fused_attention)`` which is then broadcast under the
        sender name ``"module_a|module_b"``.
        """

        key = tuple(sorted((module_a, module_b)))
        with self._lock:
            self._cross_attn[key] = handler

    def _trigger_cross_attention(self, sender: str) -> None:
        with self._lock:
            items = list(self._cross_attn.items())
            states = dict(self._state)
            attns = dict(self._attention)

        for (a, b), handler in items:
            if sender not in (a, b):
                continue
            other = b if sender == a else a
            if other not in states:
                continue
            state_a = states[a]
            state_b = states[b]
            att_a = attns.get(a)
            att_b = attns.get(b)
            fused_state, fused_attn = handler(state_a, state_b, att_a, att_b)
            self.broadcast(f"{a}|{b}", fused_state, fused_attn, _allow_cross=False)

    # ------------------------------------------------------------------
    def state(self, name: str) -> Any:
        """Return the last state published by *name*."""
        with self._lock:
            return self._state.get(name)

    def attention(self, name: str) -> Optional[List[float]]:
        """Return the last attention vector published by *name*."""
        with self._lock:
            return self._attention.get(name)

    def attention_threshold(self) -> float:
        """Return the current activation threshold for broadcasts."""

        with self._lock:
            return self._attention_threshold

    # ------------------------------------------------------------------
    def set_attention_threshold(self, threshold: float) -> None:
        """Set the activation threshold required to enter consciousness."""

        with self._lock:
            self._attention_threshold = float(threshold)

    # ------------------------------------------------------------------
    # Blackboard API
    # ------------------------------------------------------------------
    def publish_message(
        self,
        message: WorkspaceMessage | Dict[str, Any],
        *,
        attention: Optional[Sequence[float] | float] = None,
        strategy: str = "full",
        targets: Optional[List[str]] = None,
        k: Optional[int] = None,
        propagate: bool = False,
        sender: Optional[str] = None,
    ) -> WorkspaceMessage:
        """Store *message* on the workspace blackboard and optionally broadcast it."""

        if not isinstance(message, WorkspaceMessage):
            message = WorkspaceMessage.from_dict(dict(message))

        tags = _normalise_tags(message.tags)
        message.tags = tags

        att_override = _ensure_sequence(attention)
        if att_override is not None:
            message.attention = tuple(att_override)
        elif message.attention is not None:
            message.attention = tuple(message.attention)

        with self._lock:
            seq = self._message_seq
            self._message_seq += 1
            message.sequence = seq
            self._messages.append((seq, message))
            subscribers = list(self._message_subs)

        for callback, filter_fn in subscribers:
            try:
                if filter_fn is not None and not filter_fn(message):
                    continue
                if inspect.iscoroutinefunction(callback):
                    asyncio.create_task(callback(message))
                else:
                    result = callback(message)
                    if inspect.isawaitable(result):
                        asyncio.create_task(result)
            except Exception:
                # Defensive: keep workspace resilient to subscriber failures
                pass

        if propagate:
            broadcast_sender = sender or f"blackboard:{message.source}"
            self.broadcast(
                broadcast_sender,
                message.to_dict(),
                attention=message.attention if message.attention is not None else None,
                strategy=strategy,
                targets=list(targets) if targets is not None else None,
                k=k,
                _allow_cross=False,
            )

        return message

    def get_updates(
        self,
        *,
        cursor: Optional[int] = None,
        since: Optional[float] = None,
        types: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        exclude_sources: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        min_importance: Optional[float] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> Tuple[List[WorkspaceMessage], Optional[int]]:
        """Return messages filtered by the provided constraints.

        The *cursor* denotes the last processed message sequence. Only messages
        with a higher sequence value are returned. The second element of the
        tuple contains the sequence of the newest returned message (or ``None``
        when no messages match).
        """

        with self._lock:
            entries = list(self._messages)

        if newest_first:
            iterable: Iterable[Tuple[int, WorkspaceMessage]] = reversed(entries)
        else:
            iterable = entries

        type_set = set(types) if types else None
        source_set = set(sources) if sources else None
        exclude_set = set(exclude_sources) if exclude_sources else set()
        tag_set = set(tags) if tags else None
        importance_min = float(min_importance) if min_importance is not None else None

        collected: List[WorkspaceMessage] = []
        last_cursor: Optional[int] = None

        for seq, message in iterable:
            if cursor is not None and seq <= cursor:
                continue
            if since is not None and message.timestamp <= since:
                continue
            if type_set is not None and message.type not in type_set:
                continue
            if source_set is not None and message.source not in source_set:
                continue
            if exclude_set and message.source in exclude_set:
                continue
            if tag_set and not tag_set.intersection(message.tags):
                continue
            if importance_min is not None and message.importance < importance_min:
                continue
            collected.append(replace(message))
            last_cursor = seq if last_cursor is None else max(last_cursor, seq)
            if limit is not None and len(collected) >= limit:
                break

        if newest_first:
            collected.reverse()

        return collected, last_cursor

    # Internal methods --------------------------------------------------
    def _process_queue(self) -> None:
        while True:
            with self._lock:
                if not self._candidates:
                    return
                activation_neg, _, data = self._candidates[0]
                activation = -activation_neg
                if activation < self._attention_threshold:
                    return
                heapq.heappop(self._candidates)
            sender, state, att_list, strategy, targets, k, allow_cross = data
            recipients = self._deliver_broadcast(sender, state, att_list, strategy, targets, k, allow_cross)
            self._push_to_all(sender, state, att_list, recipients)

    def _deliver_broadcast(
        self,
        sender: str,
        state: Any,
        att_list: Optional[List[float]],
        strategy: str,
        targets: Optional[List[str]],
        k: Optional[int],
        allow_cross: bool,
    ) -> Set[str]:
        """Deliver broadcast to recipients based on strategy.

        Returns the set of recipient module names.
        """

        with self._lock:
            recipients = [name for name in self._modules if name != sender]
            if strategy == "local":
                if targets is None:
                    raise ValueError("targets must be provided for local strategy")
                recipients = [n for n in targets if n in self._modules and n != sender]
            elif strategy == "sparse":
                if k is None:
                    raise ValueError("k must be provided for sparse strategy")
                scores = {n: sum(self._attention.get(n, [])) for n in recipients}
                recipients = [n for n, _ in sorted(scores.items(), key=lambda i: i[1], reverse=True)[:k]]
            elif targets is not None:
                recipients = [n for n in targets if n in self._modules and n != sender]

            modules = {name: self._modules[name] for name in recipients}
            subs = list(self._state_subs.get(sender, []))

        for name, module in modules.items():
            handler = getattr(module, "receive_broadcast", None)
            if callable(handler):
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(sender, state, att_list))
                else:
                    handler(sender, state, att_list)

        for handler in subs:
            handler(state)

        if allow_cross:
            self._trigger_cross_attention(sender)

        return set(modules.keys())

    def _push_to_all(
        self,
        sender: str,
        state: Any,
        att_list: Optional[List[float]],
        already_sent: Set[str],
    ) -> None:
        """After broadcast, push the state to all remaining modules."""

        with self._lock:
            recipients = [n for n in self._modules if n not in already_sent and n != sender]
            modules = {name: self._modules[name] for name in recipients}

        for module in modules.values():
            handler = getattr(module, "receive_broadcast", None)
            if callable(handler):
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(sender, state, att_list))
                else:
                    handler(sender, state, att_list)


# Global workspace instance

global_workspace = GlobalWorkspace()
