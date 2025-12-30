"""Meta-learning retrieval policy for external knowledge acquisition.

This module provides a lightweight policy that learns which retrieval channels
work best for different task "domains" (e.g., algorithm vs API usage), and can
suggest a retrieval configuration for the agent.

Design goals:
- Dependency-light (stdlib only) and deterministic by default.
- Persist a small amount of state (success/failure counts) for online learning.
- Safe: only suggests tools that are enabled/available to the agent.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_state_path() -> Path:
    raw = os.environ.get("BSS_META_RETRIEVAL_STATE_PATH") or os.environ.get("BSS_META_RETRIEVAL_STATE")
    if raw:
        try:
            return Path(str(raw)).expanduser()
        except Exception:
            pass
    return _repo_root() / "data" / "meta_retrieval_policy.json"


def _normalize_enabled_actions(enabled_actions: Mapping[str, Any] | None) -> Dict[str, bool]:
    if not isinstance(enabled_actions, Mapping):
        return {}
    out: Dict[str, bool] = {}
    for key, value in enabled_actions.items():
        if not key:
            continue
        out[str(key)] = bool(value)
    return out


def _domain_from_text(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t:
        return "general"

    api_re = re.compile(
        r"\b(api|sdk|grpc|http|endpoint|request|response|parameter|auth|token|oauth|docs|documentation|swagger|openapi)\b"
    )
    api_zh_re = re.compile(r"(接口|文档|参数|调用|鉴权|授权|请求|响应|端点)")
    if api_re.search(t) or api_zh_re.search(t):
        return "api"

    debug_re = re.compile(r"\b(traceback|exception|error|bug|stack trace|segfault|importerror|typeerror|attributeerror)\b")
    debug_zh_re = re.compile(r"(报错|异常|错误|崩溃|堆栈|回溯|importerror|typeerror|attributeerror)")
    if debug_re.search(t) or debug_zh_re.search(t):
        return "debug"

    algo_re = re.compile(r"\b(algorithm|complexity|optimi[sz]e|dynamic programming|dp|mcts|a\\*|dijkstra|rl|nas)\b")
    algo_zh_re = re.compile(r"(算法|复杂度|优化|动态规划|强化学习|神经结构搜索|NAS|遗传算法|元学习)")
    if algo_re.search(t) or algo_zh_re.search(t):
        return "algorithm"

    code_re = re.compile(r"\b(refactor|implement|implementation|class|function|module|library)\b")
    code_zh_re = re.compile(r"(实现|代码|函数|类|模块|库)")
    if code_re.search(t) or code_zh_re.search(t):
        return "code"

    return "general"


def _beta_mean(successes: float, failures: float, *, prior_success: float, prior_failure: float) -> float:
    alpha = max(0.0, float(prior_success)) + max(0.0, float(successes))
    beta = max(0.0, float(prior_failure)) + max(0.0, float(failures))
    denom = alpha + beta
    if denom <= 0:
        return 0.5
    return float(alpha / denom)


def _normalize_unit_reward(value: Any) -> float | None:
    """Normalise a feedback score into [0, 1].

    Accepts already-normalised values in [0, 1], and common rating scales like
    1-5 / 1-10 / 0-100.
    """

    if value is None:
        return None
    try:
        raw = float(value)
    except Exception:
        return None
    if raw != raw:  # NaN
        return None
    if raw <= 1.0:
        return max(0.0, min(1.0, raw))
    if raw <= 5.0:
        return max(0.0, min(1.0, raw / 5.0))
    if raw <= 10.0:
        return max(0.0, min(1.0, raw / 10.0))
    if raw <= 100.0:
        return max(0.0, min(1.0, raw / 100.0))
    return max(0.0, min(1.0, raw))


@dataclass(frozen=True)
class MetaRetrievalPolicyConfig:
    state_path: Path
    save_on_update: bool = True
    top_channels: int = 2
    prior_success: float = 1.0
    prior_failure: float = 1.0
    allow_github_code_search: bool = False
    reward_weight: float = 1.0


class MetaRetrievalPolicy:
    """Learn a per-domain preference over retrieval channels.

    State is a dict of:
      domain -> channel -> {"successes": float, "failures": float}
    """

    CHANNELS: Tuple[str, ...] = ("code_index", "web_search", "documentation_tool", "github_code_search")

    def __init__(self, config: MetaRetrievalPolicyConfig | None = None) -> None:
        cfg = config or MetaRetrievalPolicyConfig(state_path=_default_state_path())
        self.config = cfg
        self._state: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = Path(self.config.state_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._state = {}
            return
        domains = raw.get("domains") if isinstance(raw, dict) else None
        if not isinstance(domains, dict):
            self._state = {}
            return
        parsed: Dict[str, Dict[str, Dict[str, float]]] = {}
        for domain, channel_map in domains.items():
            if not isinstance(channel_map, dict):
                continue
            d = str(domain or "").strip() or "general"
            parsed[d] = {}
            for channel, counts in channel_map.items():
                if channel not in self.CHANNELS:
                    continue
                if not isinstance(counts, dict):
                    continue
                parsed[d][channel] = {
                    "successes": max(0.0, _safe_float(counts.get("successes"), 0.0)),
                    "failures": max(0.0, _safe_float(counts.get("failures"), 0.0)),
                }
        self._state = parsed

    def save(self) -> None:
        path = Path(self.config.state_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "domains": self._state,
        }
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        except Exception:
            return

    def classify(self, task_text: str) -> str:
        return _domain_from_text(task_text)

    def _scores_for_domain(self, domain: str) -> Dict[str, float]:
        self.load()
        d = str(domain or "").strip() or "general"
        channel_map = self._state.get(d, {})
        scores: Dict[str, float] = {}
        for channel in self.CHANNELS:
            counts = channel_map.get(channel, {})
            scores[channel] = _beta_mean(
                _safe_float(counts.get("successes"), 0.0),
                _safe_float(counts.get("failures"), 0.0),
                prior_success=self.config.prior_success,
                prior_failure=self.config.prior_failure,
            )
        return scores

    def suggest(
        self,
        *,
        task_text: str,
        enabled_actions: Mapping[str, Any] | None = None,
        has_local_roots: bool = True,
    ) -> Dict[str, Any]:
        """Return a knowledge-acquisition config patch + meta explanation.

        Returns a dict containing:
          - domain: str
          - channels: List[str]
          - scores: Dict[str, float]
          - config_patch: Dict[str, Any]
        """

        domain = self.classify(task_text)
        enabled = _normalize_enabled_actions(enabled_actions)

        base_order: List[str]
        if domain == "api":
            base_order = ["documentation_tool", "web_search", "code_index"]
        elif domain == "algorithm":
            base_order = ["web_search", "code_index", "documentation_tool"]
        elif domain in {"debug", "code"}:
            base_order = ["code_index", "documentation_tool", "web_search"]
        else:
            base_order = ["code_index", "web_search", "documentation_tool"]

        allowed: List[str] = []
        if enabled.get("code_index_search", True) and has_local_roots:
            allowed.append("code_index")
        if enabled.get("web_search", False):
            allowed.append("web_search")
        if enabled.get("documentation_tool", False):
            allowed.append("documentation_tool")
        if self.config.allow_github_code_search and enabled.get("github_code_search", False):
            allowed.append("github_code_search")

        scores = self._scores_for_domain(domain)
        order_index = {name: idx for idx, name in enumerate(base_order)}

        def _sort_key(channel: str) -> Tuple[float, int]:
            return (float(scores.get(channel, 0.5)), -order_index.get(channel, 999))

        chosen: List[str] = []
        if "code_index" in allowed:
            chosen.append("code_index")

        remaining = [c for c in allowed if c not in chosen]
        remaining.sort(key=_sort_key, reverse=True)
        limit = max(1, int(self.config.top_channels))
        for channel in remaining:
            if len(chosen) >= limit:
                break
            chosen.append(channel)

        config_patch: Dict[str, Any] = {}
        if "web_search" in chosen:
            config_patch["web_search"] = True
        if "documentation_tool" in chosen:
            config_patch["documentation"] = True
        if "github_code_search" in chosen:
            config_patch["github_code_search"] = True

        return {
            "domain": domain,
            "channels": chosen,
            "scores": {k: float(v) for k, v in scores.items()},
            "config_patch": config_patch,
        }

    def observe(
        self,
        *,
        domain: str,
        channels: Sequence[str],
        success: bool | None = None,
        reward: float | None = None,
    ) -> None:
        """Update success/failure counts for a (domain, channels) outcome.

        When ``reward`` is provided, it is treated as an RLHF-style feedback
        score and normalised into ``[0, 1]`` (supporting common scales like
        1-5 / 1-10 / 0-100). When absent, ``success`` is used as a binary
        signal.
        """

        self.load()
        dom = str(domain or "").strip() or "general"
        ch_list = [str(c or "").strip() for c in channels if c]
        if not ch_list:
            return
        r = _normalize_unit_reward(reward)
        if r is None:
            if success is None:
                return
            r = 1.0 if bool(success) else 0.0
        r = max(0.0, min(1.0, float(r)))
        weight = max(0.0, float(getattr(self.config, "reward_weight", 1.0)))
        succ_inc = weight * r
        fail_inc = weight * (1.0 - r)

        dom_map = self._state.setdefault(dom, {})
        for channel in ch_list:
            if channel not in self.CHANNELS:
                continue
            counts = dom_map.setdefault(channel, {"successes": 0.0, "failures": 0.0})
            counts["successes"] = max(0.0, _safe_float(counts.get("successes"), 0.0)) + succ_inc
            counts["failures"] = max(0.0, _safe_float(counts.get("failures"), 0.0)) + fail_inc

        if self.config.save_on_update:
            self.save()

    def snapshot(self) -> Dict[str, Any]:
        self.load()
        return {"config": {"state_path": str(self.config.state_path)}, "domains": json.loads(json.dumps(self._state))}
