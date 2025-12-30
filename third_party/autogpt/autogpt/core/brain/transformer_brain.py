from __future__ import annotations

import logging
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn, optim
    from torch.nn import functional as F
    from torch.nn.utils import clip_grad_norm_
except Exception:  # pragma: no cover - optional dependency absent or incomplete
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    optim = None  # type: ignore
    clip_grad_norm_ = None  # type: ignore

from .config import TransformerBrainConfig
from .memory_manager import HierarchicalMemorySystem


logger = logging.getLogger(__name__)


@dataclass
class ThoughtStep:
    index: int
    summary: str
    vector: list[float]
    score: float
    memory_alignment: float

    def as_serializable(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "summary": self.summary,
            "score": self.score,
            "memory_alignment": self.memory_alignment,
            "vector": self.vector,
        }


@dataclass
class OnlineExperience:
    observation: Any
    memory: Any
    action_index: int
    logits: Any
    reward: float
    metadata: dict[str, Any]
    timestamp: float


class OnlineExperienceBuffer:
    """Fixed-size buffer that retains recent experiences for replay."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._buffer: deque[OnlineExperience] = deque(maxlen=self.capacity)

    def add(self, experience: OnlineExperience) -> None:
        self._buffer.append(experience)

    def sample(self, size: int) -> list[OnlineExperience]:
        if size <= 0 or not self._buffer:
            return []
        size = min(size, len(self._buffer))
        indices = random.sample(range(len(self._buffer)), k=size)
        return [self._buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


def serialize_observation(observation: Any) -> Any:
    if torch is not None and isinstance(observation, torch.Tensor):
        return observation.detach().cpu().tolist()
    if isinstance(observation, dict):
        return {key: serialize_observation(value) for key, value in observation.items()}
    if isinstance(observation, (list, tuple)):
        return [serialize_observation(value) for value in observation]
    if isinstance(observation, (str, int, float, bool)) or observation is None:
        return observation
    return repr(observation)


class SimpleTaskDecomposer:
    """Simple hierarchical task decomposer based on heuristic text splitting."""

    def __init__(self, max_depth: int, max_children: int) -> None:
        self.max_depth = max_depth
        self.max_children = max_children

    def decompose(
        self,
        goal: str | None,
        context: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not goal:
            return []

        segments = self._split_segments(goal)
        context_snippets = list(context or [])[: self.max_children]
        plan: list[dict[str, Any]] = []

        for idx, segment in enumerate(segments[: self.max_children], start=1):
            subtasks = self._build_subtasks(segment, level=2)
            plan.append(
                {
                    "id": f"{idx}",
                    "level": 1,
                    "task": segment,
                    "context": context_snippets,
                    "subtasks": subtasks,
                }
            )
        return plan

    def _build_subtasks(self, text: str, level: int) -> list[dict[str, Any]]:
        if level > self.max_depth:
            return []

        fragments = self._split_phrases(text)
        subtasks: list[dict[str, Any]] = []

        for idx, fragment in enumerate(fragments[: self.max_children], start=1):
            item: dict[str, Any] = {
                "id": f"{level}.{idx}",
                "level": level,
                "task": fragment,
            }
            if level + 1 <= self.max_depth:
                nested = self._build_subtasks(fragment, level + 1)
                if nested:
                    item["subtasks"] = nested
            subtasks.append(item)

        return subtasks

    def _split_segments(self, text: str) -> list[str]:
        segments = [seg.strip() for seg in re.split(r"[.;\n]+", text) if seg.strip()]
        return segments or [text.strip()]

    def _split_phrases(self, text: str) -> list[str]:
        phrases = [
            seg.strip()
            for seg in re.split(r"\b(?:then|and|after|next|->)\b", text, flags=re.IGNORECASE)
            if seg.strip()
        ]
        return phrases or [text.strip()]


if torch is None:  # pragma: no cover - optional dependency absent

    class TransformerBrain:  # type: ignore[no-redef]
        """Stub implementation when PyTorch is unavailable."""

        def __init__(self, config: TransformerBrainConfig | None = None):
            raise RuntimeError("PyTorch is required to use TransformerBrain.")


else:

    class TransformerBrain(nn.Module):
        """Transformer-based cognitive core with chain-of-thought and planning utilities."""

        def __init__(self, config: TransformerBrainConfig | None = None):
            super().__init__()
            self.config = config or TransformerBrainConfig()

            self.embedding = nn.Linear(self.config.dim, self.config.dim)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=self.config.dim,
                        nhead=self.config.heads,
                        dropout=self.config.dropout,
                        batch_first=True,
                    )
                    for _ in range(self.config.layers)
                ]
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.config.dim,
                num_heads=self.config.heads,
                dropout=self.config.dropout,
                batch_first=True,
            )
            self.action_head = nn.Linear(self.config.dim, self.config.dim)
            self.task_decomposer = SimpleTaskDecomposer(
                max_depth=self.config.task_decomposition_depth,
                max_children=self.config.task_decomposition_branch_factor,
            )

            self.memory_system = (
                HierarchicalMemorySystem(
                    working_capacity=self.config.working_memory_capacity,
                    episodic_limit=self.config.episodic_memory_limit,
                    consolidation_importance=self.config.consolidation_importance_threshold,
                    consolidation_window=self.config.consolidation_time_window,
                    consolidation_batch_size=self.config.consolidation_batch_size,
                    long_term_path=self.config.long_term_memory_path,
                    long_term_max_entries=self.config.long_term_memory_max_entries,
                    decay_half_life=self.config.memory_decay_half_life,
                    interference_penalty=self.config.memory_interference_penalty,
                    semantic_limit=self.config.semantic_memory_limit,
                )
                if self.config.enable_hierarchical_memory
                else None
            )

            self._online_buffer: OnlineExperienceBuffer | None = None
            self._online_optimizer: optim.Optimizer | None = None
            self._ewc_reference: list[Tensor] | None = None
            self._ewc_importance: list[Tensor] | None = None
            self._online_interactions: int = 0
            self._online_reward_baseline: float = 0.0
            self._last_online_loss: float | None = None
            self._init_online_learning()

            self._last_trace: dict[str, Any] = {
                "chain_of_thought": [],
                "task_plan": [],
                "attention": None,
                "goal": None,
                "derived_goal": None,
                "react_trace": [],
                "memory_hits": [],
            }
            self._last_thought: Tensor | None = None

            self._load_weights_if_available()
            self.eval()

        def _load_weights_if_available(self) -> None:
            weights = self.config.weights_path
            if not weights:
                return

            path = Path(weights)
            try:
                state = torch.load(path, map_location=torch.device("cpu"))
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.load_state_dict(state, strict=False)
            except FileNotFoundError:
                logger.warning("Transformer brain weights not found at %s", path)
            except Exception:  # pragma: no cover - informative logging only
                logger.exception("Failed to load transformer brain weights from %s", path)

        def _init_online_learning(self) -> None:
            if torch is None or not self.config.online_learning_enabled:
                self._online_buffer = None
                self._online_optimizer = None
                self._ewc_reference = None
                self._ewc_importance = None
                return

            if optim is None:
                logger.warning(
                    "torch.optim is unavailable; disabling transformer brain online learning."
                )
                self.config.online_learning_enabled = False
                self._online_buffer = None
                self._online_optimizer = None
                self._ewc_reference = None
                self._ewc_importance = None
                return

            self._online_buffer = OnlineExperienceBuffer(self.config.online_buffer_size)
            lr = self.config.online_learning_rate or self.config.learning_rate
            self._online_optimizer = optim.Adam(self.parameters(), lr=lr)
            self._init_ewc_buffers()

        def _init_ewc_buffers(self) -> None:
            if torch is None or not self.config.online_learning_enabled:
                self._ewc_reference = None
                self._ewc_importance = None
                return

            self._ewc_reference = [param.detach().clone() for param in self.parameters()]
            self._ewc_importance = [torch.zeros_like(param) for param in self.parameters()]

        @property
        def supports_online_learning(self) -> bool:
            return (
                self.config.online_learning_enabled
                and torch is not None
                and self._online_buffer is not None
                and self._online_optimizer is not None
                and self._ewc_reference is not None
                and self._ewc_importance is not None
            )

        def think(
            self,
            observation,
            memory_ctx=None,
            *,
            goal: str | None = None,
            task_context: Sequence[str] | None = None,
        ) -> Tensor:
            """Encode observation and optional memory into a refined thought vector."""

            obs = self._ensure_sequence(observation)
            x = self.embedding(obs)

            mem_embed = None
            if memory_ctx is not None:
                mem_embed = self.embedding(self._ensure_sequence(memory_ctx))

            chain_steps: list[ThoughtStep] = []
            representation = x
            for idx, layer in enumerate(self.layers, start=1):
                representation = layer(representation)
                if self.config.enable_chain_of_thought and idx <= self.config.chain_of_thought_steps:
                    chain_steps.append(self._build_thought_step(idx, representation, mem_embed))

            attn_weights = None
            if mem_embed is not None:
                representation, attn_weights = self.cross_attn(representation, mem_embed, mem_embed)

            thought = representation.mean(dim=1).squeeze(0)
            self._last_thought = thought.detach()

            context_for_tasks = (
                list(task_context) if task_context is not None else [step.summary for step in chain_steps]
            )

            memory_hits: list[dict[str, Any]] = []
            if self.memory_system is not None:
                query = goal or (context_for_tasks[0] if context_for_tasks else None)
                memory_hits = self.memory_system.retrieve(query, limit=3)
                if memory_hits:
                    context_for_tasks.extend(hit["content"] for hit in memory_hits if "content" in hit)

            planning_goal = goal or (context_for_tasks[0] if context_for_tasks else None)
            task_plan: list[dict[str, Any]] = []
            if self.config.enable_task_decomposition and planning_goal:
                task_plan = self.task_decomposer.decompose(planning_goal, context=context_for_tasks)

            chain_serialized = [step.as_serializable() for step in chain_steps]
            self._last_trace = {
                "chain_of_thought": chain_serialized,
                "task_plan": task_plan,
                "attention": self._serialize_attention(attn_weights),
                "goal": goal,
                "derived_goal": planning_goal,
                "react_trace": [],
                "memory_hits": memory_hits,
            }

            if self.memory_system is not None:
                importance = self._estimate_importance(thought)
                tags = [planning_goal] if planning_goal else []
                self.memory_system.encode_experience(
                    {
                        "observation": serialize_observation(observation),
                        "memory": serialize_observation(memory_ctx),
                        "goal": planning_goal,
                        "chain_depth": len(chain_serialized),
                    },
                    modality="observation",
                    importance=importance,
                    tags=tags,
                    metadata={"source": "think"},
                )
                self.memory_system.apply_decay()

            return thought

        def propose_action(
            self,
            thought,
            *,
            tools: Iterable[Callable[[dict[str, Any]], Any]] | None = None,
            goal: str | None = None,
        ):
            """Generate an action proposal enriched with reasoning traces."""

            t = torch.as_tensor(thought, dtype=torch.float32)
            if t.dim() > 1:
                t = t.squeeze(0)
            if t.dim() == 0:
                t = t.unsqueeze(0)

            react_trace, refined_thought = self._react_loop(t, tools or [])
            action_vec = self.action_head(refined_thought.unsqueeze(0)).squeeze(0)

            self._last_trace["react_trace"] = react_trace
            self._last_trace["final_thought"] = refined_thought.detach().cpu().tolist()

            info: dict[str, Any] = {
                "thought": refined_thought.detach().cpu().tolist(),
                "action": action_vec.detach().cpu().tolist(),
            }

            if self._last_trace.get("chain_of_thought"):
                info["chain_of_thought"] = self._last_trace["chain_of_thought"]
            if self._last_trace.get("task_plan"):
                info["task_plan"] = self._last_trace["task_plan"]
            if self._last_trace.get("attention") is not None:
                info["attention_map"] = self._last_trace["attention"]
            if react_trace:
                info["react_trace"] = react_trace
            if self._last_trace.get("memory_hits"):
                info["memory_hits"] = self._last_trace["memory_hits"]

            goal_value = (
                goal if goal is not None else self._last_trace.get("goal") or self._last_trace.get("derived_goal")
            )
            if goal_value is not None:
                info["goal"] = goal_value

            if self.memory_system is not None:
                tags = [goal_value] if goal_value else []
                self.memory_system.encode_experience(
                    {
                        "thought": info["thought"],
                        "action": info["action"],
                        "goal": goal_value,
                        "react_trace": react_trace,
                    },
                    modality="action_plan",
                    importance=self._estimate_importance(refined_thought),
                    tags=tags,
                    metadata={"source": "propose_action"},
                )
                self.memory_system.consolidate()
                self.memory_system.apply_decay()

            return "internal_brain_action", {}, info

        def _ensure_sequence(self, data) -> Tensor:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).unsqueeze(1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            return tensor

        def _build_thought_step(
            self,
            index: int,
            representation: Tensor,
            memory_embed: Tensor | None,
        ) -> ThoughtStep:
            pooled = representation.mean(dim=1).squeeze(0)
            score = pooled.norm(p=2).item()
            alignment = 0.0
            if memory_embed is not None:
                memory_vector = memory_embed.mean(dim=1).squeeze(0)
                if memory_vector.norm(p=2).item() > 0 and pooled.norm(p=2).item() > 0:
                    alignment = F.cosine_similarity(pooled, memory_vector, dim=0).item()
            summary = f"Layer {index} refinement (score={score:.3f}, alignment={alignment:.3f})"
            vector_list = pooled.detach().cpu().tolist()
            return ThoughtStep(
                index=index,
                summary=summary,
                vector=vector_list,
                score=score,
                memory_alignment=alignment,
            )

        def _serialize_attention(self, attn_weights: Tensor | None) -> Any:
            if attn_weights is None:
                return None
            return attn_weights.detach().cpu().tolist()

        def recall_memories(
            self,
            query: str | None = None,
            *,
            limit: int = 5,
            sources: Sequence[str] | None = None,
            tags: Sequence[str] | None = None,
        ) -> list[dict[str, Any]]:
            if self.memory_system is None:
                return []
            return self.memory_system.retrieve(query, limit=limit, sources=sources, tags=tags)

        def consolidate_memory(self, *, force: bool = False) -> list[str]:
            if self.memory_system is None:
                return []
            return self.memory_system.consolidate(force=force)

        def apply_memory_decay(self) -> None:
            if self.memory_system is not None:
                self.memory_system.apply_decay()

        def shutdown(self) -> None:
            if self.memory_system is not None:
                self.memory_system.shutdown()

        def _estimate_importance(self, vector: Tensor) -> float:
            if vector.numel() == 0:
                return 0.0
            norm = vector.norm(p=2).item()
            scale = math.tanh(norm / max(1.0, math.sqrt(vector.numel())))
            return float(max(0.0, min(1.0, scale)))

        def _react_loop(
            self,
            thought: Tensor,
            tools: Iterable[Callable[[dict[str, Any]], Any]],
        ) -> tuple[list[dict[str, Any]], Tensor]:
            if not tools or not self.config.enable_react:
                return [], thought

            tools_list = list(tools)
            if not tools_list:
                return [], thought

            react_trace: list[dict[str, Any]] = []
            current_state = thought

            chain = self._last_trace.get("chain_of_thought") or []

            for iteration in range(1, self.config.react_iterations + 1):
                tool = tools_list[(iteration - 1) % len(tools_list)]
                summary = (
                    chain[min(iteration - 1, len(chain) - 1)]["summary"]
                    if chain
                    else f"Iteration {iteration} refinement"
                )
                payload = {
                    "iteration": iteration,
                    "thought": current_state.detach().cpu().tolist(),
                    "summary": summary,
                    "task_plan": self._last_trace.get("task_plan", []),
                }
                try:
                    observation = tool(payload)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "ReAct tool %s raised %s",
                        getattr(tool, "__name__", repr(tool)),
                        exc,
                        exc_info=True,
                    )
                    observation = {"error": str(exc)}

                observation_serialized = serialize_observation(observation)
                react_trace.append(
                    {
                        "iteration": iteration,
                        "tool": getattr(tool, "__name__", tool.__class__.__name__),
                        "input_summary": summary,
                        "observation": observation_serialized,
                    }
                )

                observation_tensor = self._observation_to_tensor(observation, like=current_state)
                if observation_tensor is not None:
                    current_state = torch.lerp(current_state, observation_tensor, 0.5)

            return react_trace, current_state

        def _observation_to_tensor(self, observation: Any, like: Tensor) -> Tensor | None:
            if isinstance(observation, torch.Tensor):
                obs = observation.detach().to(dtype=like.dtype, device=like.device)
                if obs.shape == like.shape:
                    return obs
                if obs.numel() == like.numel():
                    return obs.reshape_as(like)
                return None
            if isinstance(observation, dict):
                if "vector" in observation:
                    return self._observation_to_tensor(observation["vector"], like)
                return None
            if isinstance(observation, (list, tuple)):
                flat = list(observation)
                if len(flat) == like.numel():
                    return torch.tensor(flat, dtype=like.dtype, device=like.device).reshape_as(like)
                if len(flat) == like.shape[0]:
                    return torch.tensor(flat, dtype=like.dtype, device=like.device)
                return None
            if isinstance(observation, (int, float)):
                return torch.full_like(like, float(observation))
            return None

        def _prepare_feature_tensor(self, data: Any) -> Tensor:
            if isinstance(data, torch.Tensor):
                tensor = data.detach().clone()
            else:
                tensor = torch.as_tensor(data, dtype=torch.float32)
            if tensor.dim() > 1:
                tensor = tensor.view(-1)
            return tensor.to(dtype=torch.float32)

        def _forward_pass(self, observation: Tensor, memory: Tensor | None = None) -> Tensor:
            obs_seq = observation
            if obs_seq.dim() == 1:
                obs_seq = obs_seq.unsqueeze(0).unsqueeze(1)
            elif obs_seq.dim() == 2:
                obs_seq = obs_seq.unsqueeze(1)
            representation = self.embedding(obs_seq)

            mem_embed = None
            if memory is not None:
                mem_seq = memory
                if mem_seq.dim() == 1:
                    mem_seq = mem_seq.unsqueeze(0).unsqueeze(1)
                elif mem_seq.dim() == 2:
                    mem_seq = mem_seq.unsqueeze(1)
                mem_embed = self.embedding(mem_seq)

            for layer in self.layers:
                representation = layer(representation)
            if mem_embed is not None:
                representation, _ = self.cross_attn(representation, mem_embed, mem_embed)

            return representation.mean(dim=1).squeeze(0)

        def _derive_reward(self, outcome: Any, metadata: Optional[dict[str, Any]] = None) -> float:
            metadata = metadata or {}
            if "reward" in metadata:
                try:
                    return float(metadata["reward"])
                except (TypeError, ValueError):
                    pass

            status = getattr(outcome, "status", None)
            if status == "success":
                return 1.0
            if status == "error":
                return -1.0
            if status == "interrupted_by_human":
                return 0.0
            return 0.0

        def _reward_weight(self, reward: float) -> float:
            centered = reward - self._online_reward_baseline
            return float(max(0.1, 1.0 + centered))

        def _should_online_update(self) -> bool:
            if not self.supports_online_learning or self._online_buffer is None:
                return False
            if len(self._online_buffer) < max(1, self.config.online_min_batch):
                return False
            interval = max(1, self.config.online_update_interval)
            return self._online_interactions % interval == 0

        def _ewc_penalty(self, device: torch.device) -> Tensor:
            if (
                not self.supports_online_learning
                or self._ewc_reference is None
                or self._ewc_importance is None
                or self.config.online_ewc_lambda <= 0
            ):
                return torch.zeros((), dtype=torch.float32, device=device)

            penalty = torch.zeros((), dtype=torch.float32, device=device)
            for param, importance, reference in zip(
                self.parameters(), self._ewc_importance, self._ewc_reference
            ):
                diff = param - reference
                penalty = penalty + (importance.to(device) * diff.pow(2)).sum()
            return 0.5 * self.config.online_ewc_lambda * penalty

        def _update_ewc_importance(self, grad_squares: Sequence[Tensor]) -> None:
            if (
                not self.supports_online_learning
                or self._ewc_reference is None
                or self._ewc_importance is None
            ):
                return

            decay = float(self.config.online_ewc_decay)
            for param, importance, reference, grad_sq in zip(
                self.parameters(), self._ewc_importance, self._ewc_reference, grad_squares
            ):
                importance.mul_(decay).add_((1.0 - decay) * grad_sq.to(importance.device))
                reference.data.copy_(param.detach())

        def _perform_online_update(self) -> None:
            if (
                not self.supports_online_learning
                or self._online_buffer is None
                or self._online_optimizer is None
            ):
                return

            batch_size = min(self.config.online_batch_size, len(self._online_buffer))
            batch = self._online_buffer.sample(batch_size)
            if not batch:
                return

            device = next(self.parameters()).device
            was_training = self.training
            self.train()

            logits_list: list[Tensor] = []
            targets: list[int] = []
            weights: list[float] = []

            for exp in batch:
                obs = exp.observation.to(device=device, dtype=torch.float32)
                mem = (
                    exp.memory.to(device=device, dtype=torch.float32)
                    if isinstance(exp.memory, torch.Tensor)
                    else None
                )
                thought = self._forward_pass(obs, mem)
                logits = self.action_head(thought)
                logits_list.append(logits)
                targets.append(exp.action_index)
                weights.append(self._reward_weight(exp.reward))

            target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
            stacked_logits = torch.stack(logits_list)
            ce_losses = F.cross_entropy(stacked_logits, target_tensor, reduction="none")

            weight_tensor = torch.tensor(weights, dtype=stacked_logits.dtype, device=device)
            weight_sum = weight_tensor.sum().clamp(min=1.0)
            ce_loss = (ce_losses * weight_tensor).sum() / weight_sum
            penalty = self._ewc_penalty(device)
            total_loss = ce_loss + penalty

            self._online_optimizer.zero_grad()
            total_loss.backward()

            if self.config.online_gradient_clip > 0 and clip_grad_norm_ is not None:
                clip_grad_norm_(self.parameters(), self.config.online_gradient_clip)

            grad_squares = []
            for param in self.parameters():
                if param.grad is None:
                    grad_squares.append(torch.zeros_like(param))
                else:
                    grad_squares.append(param.grad.detach().clone().pow(2))

            self._online_optimizer.step()
            self._update_ewc_importance(grad_squares)
            self._last_online_loss = float(total_loss.detach().cpu().item())

            if not was_training:
                self.eval()

            try:
                logger.debug(
                    "Transformer brain online update completed (batch=%d, loss=%.4f)",
                    len(batch),
                    float(total_loss.detach().cpu().item()),
                )
            except Exception:  # pragma: no cover - logging only
                pass

        def complete_interaction(
            self,
            *,
            observation: Any,
            memory: Any,
            brain_result: tuple[str, dict[str, Any], dict[str, Any]],
            outcome: Any,
            metadata: Optional[dict[str, Any]] = None,
        ) -> None:
            if not self.supports_online_learning or self._online_buffer is None:
                return

            metadata = metadata or {}
            _, _, info = brain_result
            action_vec = info.get("action")
            if action_vec is None:
                return

            action_logits = torch.as_tensor(action_vec, dtype=torch.float32)
            action_index = int(metadata.get("action_index", action_logits.argmax().item()))
            reward = self._derive_reward(outcome, metadata=metadata)

            observation_tensor = self._prepare_feature_tensor(observation).cpu()
            memory_tensor = (
                self._prepare_feature_tensor(memory).cpu() if memory is not None else None
            )

            experience = OnlineExperience(
                observation=observation_tensor,
                memory=memory_tensor,
                action_index=action_index,
                logits=action_logits.detach().clone().cpu(),
                reward=reward,
                metadata={
                    **metadata,
                    "status": getattr(outcome, "status", None),
                },
                timestamp=time.time(),
            )
            self._online_buffer.add(experience)

            # Update reward baseline for adaptive weighting
            self._online_reward_baseline = (0.95 * self._online_reward_baseline) + (0.05 * reward)

            self._online_interactions += 1
            if self._should_online_update():
                self._perform_online_update()
