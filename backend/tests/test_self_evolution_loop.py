import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

BACKEND_EXECUTION = Path(ROOT, "backend", "execution")


def _load():
    module_path = BACKEND_EXECUTION / "adaptive_controller.py"
    spec = importlib.util.spec_from_file_location("backend.execution.adaptive_controller", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module  # type: ignore[attr-defined]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


module = _load()
EvolutionCandidate = module.EvolutionCandidate  # type: ignore[attr-defined]
KnowledgeConstraint = module.KnowledgeConstraint  # type: ignore[attr-defined]
EvolutionGuard = module.EvolutionGuard  # type: ignore[attr-defined]
SelfEvolutionLoop = module.SelfEvolutionLoop  # type: ignore[attr-defined]
EvolutionConstraintViolation = module.EvolutionConstraintViolation  # type: ignore[attr-defined]
immutable_components_constraint = module.immutable_components_constraint  # type: ignore[attr-defined]
DiversityArchive = module.DiversityArchive  # type: ignore[attr-defined]


def test_self_evolution_loop_selects_best_candidate():
    history: list[str] = []

    def observer():
        return {"reward": 0.1}

    def generator(metrics):
        yield EvolutionCandidate(
            name="baseline",
            payload={"delta": 0.1, "safety": True},
            metadata={"safety": True},
            score=None,
        )
        yield EvolutionCandidate(
            name="improved",
            payload={"delta": 0.8, "safety": True},
            metadata={"safety": True},
            score=None,
        )

    def evaluator(candidate, metrics):
        return float(candidate.payload["delta"])

    def replacer(candidate, metrics):
        history.append(candidate.name)
        return True

    guard = EvolutionGuard(
        [
            KnowledgeConstraint(
                name="safety",
                validator=lambda state: bool(state.get("safety", False)),
                message="safety flag must remain enabled",
            ),
            immutable_components_constraint(["core_safety"]),
        ]
    )

    loop = SelfEvolutionLoop(
        observer=observer,
        generator=generator,
        evaluator=evaluator,
        replacer=replacer,
        guard=guard,
        regression_runner=lambda: True,
    )

    result = loop.run_cycle()
    assert result is not None
    assert result.name == "improved"
    assert history == ["improved"]


def test_self_evolution_loop_enforces_constraints():
    def observer():
        return {}

    def generator(metrics):
        yield EvolutionCandidate(
            name="unsafe",
            payload={"safety": False, "modified_components": ["core_safety"]},
            metadata={"safety": False},
            score=None,
        )

    def evaluator(candidate, metrics):
        return 1.0

    def replacer(candidate, metrics):
        raise AssertionError("Unsafe candidate should not apply")

    guard = EvolutionGuard(
        [
            KnowledgeConstraint(
                name="safety",
                validator=lambda state: bool(state.get("safety", False)),
                message="safety module must stay enabled",
            )
        ]
    )

    loop = SelfEvolutionLoop(
        observer=observer,
        generator=generator,
        evaluator=evaluator,
        replacer=replacer,
        guard=guard,
    )

    result = loop.run_cycle()
    assert result is None


def test_diversity_archive_preserves_multiple_styles():
    archive = DiversityArchive(
        max_size=2,
        distance_fn=lambda a, b: 0.0 if a.metadata.get("style") == b.metadata.get("style") else 1.0,
        min_distance=0.5,
    )

    def observer():
        return {}

    call_count = {"value": 0}

    def generator(metrics):
        call_count["value"] += 1
        if call_count["value"] == 1:
            yield EvolutionCandidate(
                name="candidate_a",
                payload={"delta": 0.65, "safety": True},
                metadata={"safety": True, "style": "analytical", "modified_components": []},
            )
            yield EvolutionCandidate(
                name="candidate_b",
                payload={"delta": 0.6, "safety": True},
                metadata={"safety": True, "style": "creative", "modified_components": []},
            )
        else:
            yield EvolutionCandidate(
                name="candidate_b2",
                payload={"delta": 0.75, "safety": True},
                metadata={"safety": True, "style": "creative", "modified_components": []},
            )
            yield EvolutionCandidate(
                name="candidate_a2",
                payload={"delta": 0.7, "safety": True},
                metadata={"safety": True, "style": "analytical", "modified_components": []},
            )

    def evaluator(candidate, metrics):
        return float(candidate.payload["delta"])

    accepted: list[str] = []

    def replacer(candidate, metrics):
        accepted.append(candidate.name)
        return True

    guard = EvolutionGuard(
        [
            KnowledgeConstraint(
                name="safety",
                validator=lambda state: bool(state.get("safety", False)),
                message="safety must remain true",
            )
        ]
    )

    loop = SelfEvolutionLoop(
        observer=observer,
        generator=generator,
        evaluator=evaluator,
        replacer=replacer,
        guard=guard,
        diversity_archive=archive,
    )

    loop.run_cycle()
    loop.run_cycle()
    entries = archive.entries()
    assert len(entries) <= 2
    styles = {entry.metadata.get("style") for entry in entries}
    assert "analytical" in styles and "creative" in styles

