"""Post-generation reflection backed by an LLM."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Tuple

from backend.memory import MemoryProtocol


@dataclass
class ReflectionResult:
    """Structured output of a reflection evaluation."""

    confidence: float
    sentiment: str
    raw: str = ""


EvaluationFn = Callable[[str], ReflectionResult]
RewriteFn = Callable[[str], str]

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    openai = None

logger = logging.getLogger(__name__)


@dataclass
class ReflectionModule:
    """Evaluate and rewrite an initial response using an LLM.

    Parameters
    ----------
    evaluate
        Optional custom evaluation function. If not provided a call to
        :func:`openai.ChatCompletion.create` is attempted and the result is
        interpreted as a ``score`` between 0 and 1.
    rewrite
        Optional custom rewrite function. If not provided the LLM is asked to
        improve the text while keeping the meaning.
    max_passes
        Maximum number of reflection passes to attempt.
    quality_threshold
        Minimum acceptable score before stopping further reflection passes.
    model
        Name of the LLM model to use when calling the OpenAI API.
    history
        List of tuples containing ``(evaluation, revised_text)`` for each pass.
        This acts as a log for later analysis or fine-tuning.
    """

    evaluate: EvaluationFn | None = None
    rewrite: RewriteFn | None = None
    max_passes: int = 1
    quality_threshold: float = 0.0
    model: str = "gpt-3.5-turbo"
    history: list[Tuple[ReflectionResult, str]] = field(default_factory=list)
    logger: logging.Logger | None = None
    callback: Callable[[ReflectionResult, str], None] | None = None

    def __post_init__(self) -> None:
        if self.evaluate is None:
            self.evaluate = self._llm_evaluate
        if self.rewrite is None:
            self.rewrite = self._llm_rewrite
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    # --- Default LLM-backed implementations ---------------------------------

    def _llm_evaluate(self, text: str) -> ReflectionResult:
        """Return a structured assessment for ``text`` using an LLM.

        Falls back to a trivial heuristic if the API is unavailable.
        """

        if openai is not None and os.getenv("OPENAI_API_KEY"):
            try:  # pragma: no cover - network call
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Provide JSON with keys confidence (0-1) and "
                                "sentiment ('positive','negative','neutral') "
                                "evaluating the quality of: \n" + text
                            ),
                        }
                    ],
                )
                content = response["choices"][0]["message"]["content"].strip()
                try:
                    data = json.loads(content)
                    return ReflectionResult(
                        confidence=float(data.get("confidence", 0.0)),
                        sentiment=data.get("sentiment", "neutral"),
                        raw=content,
                    )
                except Exception:
                    return ReflectionResult(0.0, "neutral", raw=content)
            except Exception as exc:  # pragma: no cover - network failure
                logger.warning("LLM evaluation failed: %s", exc)

        # Fallback heuristic
        sentiment = "negative" if re.search(r"fail|error", text, re.I) else (
            "positive" if re.search(r"success|done|complete", text, re.I) else "neutral"
        )
        length = len(text.split())
        return ReflectionResult(0.0, sentiment, raw=f"unavailable length={length}")

    def _llm_rewrite(self, text: str) -> str:
        """Improve ``text`` using an LLM.

        Falls back to appending a tag if the API is unavailable.
        """

        if openai is not None and os.getenv("OPENAI_API_KEY"):
            try:  # pragma: no cover - network call
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Rewrite the following response to improve "
                                "clarity and quality while preserving meaning:\n"
                                + text
                            ),
                        }
                    ],
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # pragma: no cover - network failure
                logger.warning("LLM rewrite failed: %s", exc)
        return text + " [revised]"

    # ------------------------------------------------------------------------

    def reflect(self, text: str) -> Tuple[ReflectionResult, str]:
        """Evaluate ``text`` and return ``(evaluation, revised_text)``.

        The method performs up to ``max_passes`` reflection cycles. After each
        evaluation, the score is checked against ``quality_threshold``. If the
        threshold is not met, a rewrite is attempted. Each pass is stored in
        :attr:`history` and logged for later analysis.
        """

        self.history.clear()
        revised = text

        for i in range(self.max_passes):
            evaluation = self.evaluate(revised)
            if self.logger:
                self.logger.info("reflection_evaluation_pass_%d: %s", i, evaluation)
            score = evaluation.confidence
            if score >= self.quality_threshold or i == self.max_passes - 1:
                self.history.append((evaluation, revised))
                if self.callback:
                    self.callback(evaluation, revised)
                return evaluation, revised
            revised = self.rewrite(revised)
            if self.logger:
                self.logger.info("reflection_revision_pass_%d: %s", i, revised)
            self.history.append((evaluation, revised))
            if self.callback:
                self.callback(evaluation, revised)

        return evaluation, revised  # pragma: no cover - loop always returns


def history_to_json(history: list[Tuple[ReflectionResult, str]]) -> str:
    """Serialize a reflection history to JSON."""

    return json.dumps(
        [{"evaluation": asdict(ev), "revision": rev} for ev, rev in history]
    )


def save_history(
    memory: MemoryProtocol,
    history: list[Tuple[ReflectionResult, str]],
    *,
    category: str = "reflection",
) -> None:
    """Persist ``history`` to ``memory`` under ``category``."""

    memory.store(history_to_json(history), metadata={"category": category})


def load_histories(
    memory: MemoryProtocol, *, category: str = "reflection"
) -> Iterable[list[Tuple[ReflectionResult, str]]]:
    """Yield histories stored in ``memory`` under ``category``."""

    for item in memory.retrieve({"category": category}):
        data = json.loads(item)
        yield [
            (ReflectionResult(**entry["evaluation"]), entry["revision"])
            for entry in data
        ]
