from __future__ import annotations

"""Sample Ray Serve deployment that emulates a remote brain inference service."""

import random
import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from ray import serve

app = FastAPI(title="Brain Serving Demo", version="0.1.0")


def _fake_brain_cycle(inputs: Dict[str, Any]) -> Dict[str, Any]:
    goal = inputs.get("goals", ["clarify_objective"])[0] if inputs else "clarify_objective"
    attention = random.random()
    plan = [
        "scan_context",
        "gather_signals",
        "synthesise_plan",
    ]
    return {
        "perception": {
            "modalities": inputs.get("modalities", {}),
            "semantic": inputs.get("semantic", {}),
            "knowledge_facts": [],
        },
        "emotion": {
            "primary": "neutral",
            "intensity": 0.2 + attention * 0.1,
            "mood": 0.1,
            "dimensions": {"valence": 0.2, "arousal": attention},
            "context": {"focus": attention},
            "decay": 0.05,
            "intent_bias": {"explore": 0.4, "observe": 0.6},
        },
        "intent": {
            "intention": "explore",
            "salience": True,
            "plan": plan,
            "confidence": 0.65 + attention * 0.1,
            "weights": {"explore": 0.55, "observe": 0.3, "approach": 0.1, "withdraw": 0.05},
            "tags": ["demo"],
        },
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.45,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.2,
        },
        "curiosity": {
            "drive": 0.5 + attention * 0.3,
            "novelty_preference": 0.6,
            "fatigue": 0.2,
            "last_novelty": attention,
        },
        "energy_used": 18,
        "idle_skipped": 0,
        "thoughts": {
            "focus": goal,
            "summary": f"Evaluate options for {goal}",
            "plan": plan,
            "confidence": 0.6,
            "memory_refs": [],
            "tags": ["demo"],
        },
        "feeling": {
            "descriptor": "engaged",
            "valence": 0.3,
            "arousal": 0.4,
            "mood": 0.2,
            "confidence": 0.55,
            "context_tags": ["demo"],
        },
        "metrics": {"latency_hint_ms": 12.5},
        "metadata": {"provider": "ray_serve_demo"},
    }


@serve.deployment(
    route_prefix="/infer",
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class TransformerBrainDeployment:
    async def __call__(self, request: Request) -> Dict[str, Any]:
        body = await request.json()
        start = time.perf_counter()
        inputs = body.get("inputs", {})
        result = _fake_brain_cycle(inputs if isinstance(inputs, dict) else {})
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "status": "ok",
            "metrics": {
                "latency_ms": latency_ms,
                "gpu_utilization": body.get("metadata", {}).get("gpu_utilization", 0.0),
                "qps": 1.0,
            },
            "result": result,
        }


brain_app = TransformerBrainDeployment.bind()
