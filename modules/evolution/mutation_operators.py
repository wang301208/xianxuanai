"""Mutation operator library for NAS / evolutionary search.

The project already contains simple genetic algorithms that use Gaussian noise
over dict-valued genomes. This module adds a *library* of mutation operators so
search can:
- mix discrete + continuous mutations (flags, categorical prompt variants)
- be orchestrated by a higher-level controller (e.g., bandit/meta-learning)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _is_flag_key(key: str) -> bool:
    key = str(key or "")
    return key.endswith("_flag") or key.endswith("_enabled")


def _flag(value: Any, default: float = 0.0) -> float:
    return 1.0 if _safe_float(value, default) >= 0.5 else 0.0


def _pick_key(keys: Sequence[str], rng: random.Random) -> str | None:
    if not keys:
        return None
    return keys[rng.randrange(len(keys))]


@dataclass(frozen=True)
class MutationContext:
    """Optional context passed into mutation operators."""

    issues: Sequence[Any] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)
    score_hint: float | None = None


class MutationOperator:
    """Base class for single-parent genome mutation operators."""

    name: str = "mutation"

    def mutate(
        self,
        genome: Mapping[str, float],
        *,
        rng: random.Random,
        context: MutationContext | None = None,
    ) -> Dict[str, float]:
        raise NotImplementedError


class GaussianNoiseOperator(MutationOperator):
    """Add Gaussian noise to a random subset of numeric genes."""

    name = "gaussian_noise"

    def __init__(
        self,
        *,
        mutation_rate: float = 0.35,
        sigma: float = 0.12,
        key_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.mutation_rate = float(mutation_rate)
        self.sigma = float(sigma)
        self.key_filter = key_filter

    def mutate(
        self,
        genome: Mapping[str, float],
        *,
        rng: random.Random,
        context: MutationContext | None = None,
    ) -> Dict[str, float]:
        out = dict(genome)
        for key, value in list(out.items()):
            if self.key_filter is not None and not self.key_filter(str(key)):
                continue
            if rng.random() >= self.mutation_rate:
                continue
            out[str(key)] = float(value) + rng.gauss(0.0, self.sigma)
        return out


class ScaleOperator(MutationOperator):
    """Scale a random continuous gene by a multiplicative factor."""

    name = "scale"

    def __init__(
        self,
        *,
        scale_sigma: float = 0.2,
        key_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.scale_sigma = float(scale_sigma)
        self.key_filter = key_filter

    def mutate(
        self,
        genome: Mapping[str, float],
        *,
        rng: random.Random,
        context: MutationContext | None = None,
    ) -> Dict[str, float]:
        keys = [
            str(k)
            for k in genome.keys()
            if not _is_flag_key(str(k)) and (self.key_filter(str(k)) if self.key_filter else True)
        ]
        target = _pick_key(keys, rng)
        if target is None:
            return dict(genome)
        out = dict(genome)
        value = _safe_float(out.get(target), 0.0)
        factor = max(0.0, 1.0 + rng.gauss(0.0, self.scale_sigma))
        out[target] = value * factor
        return out


class FlipFlagOperator(MutationOperator):
    """Flip a random *_flag gene."""

    name = "flip_flag"

    def __init__(self, *, keys: Optional[Iterable[str]] = None) -> None:
        self._keys = [str(k) for k in keys] if keys is not None else None

    def mutate(
        self,
        genome: Mapping[str, float],
        *,
        rng: random.Random,
        context: MutationContext | None = None,
    ) -> Dict[str, float]:
        keys = self._keys
        if keys is None:
            keys = [str(k) for k in genome.keys() if _is_flag_key(str(k))]
        target = _pick_key(list(keys), rng)
        if target is None:
            return dict(genome)
        out = dict(genome)
        out[target] = 1.0 - _flag(out.get(target), 0.0)
        return out


class StepIntegerOperator(MutationOperator):
    """Increment/decrement a random integer-coded gene."""

    name = "step_int"

    def __init__(
        self,
        *,
        keys: Optional[Iterable[str]] = None,
        step: int = 1,
        lo: int = -2**31,
        hi: int = 2**31 - 1,
    ) -> None:
        self._keys = [str(k) for k in keys] if keys is not None else None
        self.step = int(step) if step else 1
        self.lo = int(lo)
        self.hi = int(hi)

    def mutate(
        self,
        genome: Mapping[str, float],
        *,
        rng: random.Random,
        context: MutationContext | None = None,
    ) -> Dict[str, float]:
        keys = self._keys or [str(k) for k in genome.keys() if str(k).endswith("_count") or str(k).endswith("_variant")]
        target = _pick_key(list(keys), rng)
        if target is None:
            return dict(genome)
        out = dict(genome)
        value = int(round(_safe_float(out.get(target), 0.0)))
        direction = -1 if rng.random() < 0.5 else 1
        value = int(_clamp(value + direction * self.step, self.lo, self.hi))
        out[target] = float(value)
        return out


class PromptPerturbationOperator(MutationOperator):
    """Targeted mutation for prompt-strategy genes used by agent planners."""

    name = "prompt_perturb"

    def __init__(
        self,
        *,
        json_step: float = 0.1,
        safety_step: float = 0.1,
        variant_step: int = 1,
        max_variant: int = 2,
    ) -> None:
        self.json_step = float(json_step)
        self.safety_step = float(safety_step)
        self.variant_step = int(variant_step)
        self.max_variant = int(max_variant)

    def mutate(
        self,
        genome: Mapping[str, float],
        *,
        rng: random.Random,
        context: MutationContext | None = None,
    ) -> Dict[str, float]:
        out = dict(genome)
        # Select which prompt knob to tweak.
        choice = rng.choice(["json", "safety", "variant"])
        if choice == "json":
            key = "llm_prompt_json_strictness"
            value = _safe_float(out.get(key), 0.6)
            delta = self.json_step if rng.random() < 0.5 else -self.json_step
            out[key] = _clamp(value + delta, 0.0, 1.0)
        elif choice == "safety":
            key = "llm_prompt_safety_bias"
            value = _safe_float(out.get(key), 0.8)
            delta = self.safety_step if rng.random() < 0.5 else -self.safety_step
            out[key] = _clamp(value + delta, 0.0, 1.0)
        else:
            key = "llm_prompt_variant"
            value = int(round(_safe_float(out.get(key), 0.0)))
            delta = self.variant_step if rng.random() < 0.5 else -self.variant_step
            value = int(_clamp(value + delta, 0, self.max_variant))
            out[key] = float(value)
        return out


class MutationOperatorLibrary:
    """A registry of mutation operators that can be sampled by name."""

    def __init__(self, operators: Optional[Iterable[MutationOperator]] = None) -> None:
        self._ops: Dict[str, MutationOperator] = {}
        for op in operators or []:
            self.register(op)

    def register(self, operator: MutationOperator) -> None:
        self._ops[str(operator.name)] = operator

    def get(self, name: str) -> MutationOperator | None:
        return self._ops.get(str(name))

    def names(self) -> List[str]:
        return sorted(self._ops.keys())

    def mutate(
        self,
        genome: Mapping[str, float],
        *,
        name: str,
        rng: random.Random,
        context: MutationContext | None = None,
    ) -> Dict[str, float]:
        op = self.get(name)
        if op is None:
            raise KeyError(f"Unknown mutation operator '{name}'")
        return op.mutate(genome, rng=rng, context=context)


def default_operator_library() -> MutationOperatorLibrary:
    """Return a default operator library suitable for mixed strategy genomes."""

    return MutationOperatorLibrary(
        [
            GaussianNoiseOperator(),
            ScaleOperator(),
            FlipFlagOperator(),
            StepIntegerOperator(keys=["llm_prompt_variant"], lo=0, hi=2),
            PromptPerturbationOperator(),
        ]
    )


__all__ = [
    "MutationContext",
    "MutationOperator",
    "GaussianNoiseOperator",
    "ScaleOperator",
    "FlipFlagOperator",
    "StepIntegerOperator",
    "PromptPerturbationOperator",
    "MutationOperatorLibrary",
    "default_operator_library",
]

