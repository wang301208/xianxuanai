from __future__ import annotations

"""Evaluation utilities for assessing creative multimodal outputs."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - circular import safety
    from modules.optimization.meta_learner import MetaLearner


@dataclass
class CreativeScore:
    """Holds novelty and usefulness scores for a modality output."""

    novelty: float
    usefulness: float


class CreativeEvaluator:
    """Evaluate outputs and provide feedback for self-optimization."""

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, CreativeScore]:
        """Return novelty and usefulness scores for each modality.

        Parameters
        ----------
        results: Mapping from modality to its generation result. Each result
            is expected to contain an ``"output"`` entry with the produced
            content.
        """
        scores: Dict[str, CreativeScore] = {}
        for modality, data in results.items():
            output = data.get("output")
            if isinstance(output, str):
                words = output.split()
                novelty = len(set(words)) / (len(words) or 1)
                usefulness = 1.0 if output else 0.0
            else:
                novelty = 0.5
                usefulness = 1.0 if output is not None else 0.0
            scores[modality] = CreativeScore(novelty, usefulness)
        return scores

    def feedback(
        self, meta_learner: Optional["MetaLearner"], scores: Dict[str, CreativeScore]
    ) -> Dict[str, float]:
        """Provide feedback to the generation process via ``meta_learner``.

        Returns a mapping from modality to the aggregated score used for
        learning. If ``meta_learner`` is provided its weights are updated.
        """
        aggregate = {
            modality: (score.novelty + score.usefulness) / 2
            for modality, score in scores.items()
        }
        if meta_learner is not None:
            meta_learner.update(aggregate)
        return aggregate


__all__ = ["CreativeEvaluator", "CreativeScore"]
