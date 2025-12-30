"""
Training utilities for the internal language system.

This module provides lightweight dataset loaders and training helpers so the
language cortex, intent recogniser, and affect analyser can be improved without
relying on external LLMs.  It focuses on simple, reproducible routines that
operate on CSV/JSONL corpora and export results in formats directly consumable
by the runtime modules.
"""

from __future__ import annotations

import csv
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from BrainSimulationSystem.models.intent_recognizer import IntentRecognizer
from BrainSimulationSystem.models.language_cortex import LanguageCortex
from BrainSimulationSystem.models.language_processing import SemanticNetwork, WordRecognizer


# --------------------------------------------------------------------------- #
# Data structures                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class IntentSample:
    """Structured representation of a single labelled utterance."""

    text: str
    intent: str
    emotion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentTrainingResult:
    """Summary returned after intent prototype training."""

    label_counts: Dict[str, int]
    prototypes: Dict[str, List[float]]
    cortex_targets: Dict[str, List[float]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "label_counts": self.label_counts,
                "prototypes": self.prototypes,
                "cortex_targets": self.cortex_targets,
            },
            ensure_ascii=False,
            indent=2,
        )


@dataclass
class AffectLexicon:
    """Word-level statistics computed during affect training."""

    word_counts: Dict[str, Dict[str, int]]
    total_counts: Dict[str, int]

    def merge(self, other: "AffectLexicon") -> None:
        for emotion, words in other.word_counts.items():
            target = self.word_counts.setdefault(emotion, {})
            for word, count in words.items():
                target[word] = target.get(word, 0) + count
            self.total_counts[emotion] = self.total_counts.get(emotion, 0) + other.total_counts.get(emotion, 0)

    def to_json(self) -> str:
        return json.dumps(
            {
                "word_counts": self.word_counts,
                "total_counts": self.total_counts,
            },
            ensure_ascii=False,
            indent=2,
        )


# --------------------------------------------------------------------------- #
# Dataset loading                                                             #
# --------------------------------------------------------------------------- #


def load_intent_dataset(path: Path) -> List[IntentSample]:
    """
    Load intent training data from CSV or JSONL.

    Expected columns/keys: ``text`` (str), ``intent`` (str), optional ``emotion``.
    Additional fields are stored under ``metadata`` for downstream use.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    samples: List[IntentSample] = []

    if suffix in {".csv"}:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample = _row_to_sample(row)
                if sample:
                    samples.append(sample)
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                sample = _row_to_sample(data)
                if sample:
                    samples.append(sample)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    return samples


def _row_to_sample(row: Dict[str, Any]) -> Optional[IntentSample]:
    text = (row.get("text") or "").strip()
    intent = (row.get("intent") or "").strip()
    if not text or not intent:
        return None
    emotion = (row.get("emotion") or row.get("affect") or "").strip() or None
    metadata = {k: v for k, v in row.items() if k not in {"text", "intent", "emotion", "affect"}}
    return IntentSample(text=text, intent=intent, emotion=emotion, metadata=metadata)


# --------------------------------------------------------------------------- #
# Token utilities                                                             #
# --------------------------------------------------------------------------- #


def simple_tokenize(text: str) -> List[str]:
    """Lowercase tokenisation with punctuation stripping."""
    tokens: List[str] = []
    for raw in text.split():
        cleaned = raw.strip(".,!?\"'()[]{}")
        if cleaned:
            tokens.append(cleaned.lower())
    return tokens


# --------------------------------------------------------------------------- #
# Training helpers                                                            #
# --------------------------------------------------------------------------- #


class IntentTrainer:
    """
    Train intent prototypes and fine-tune the language cortex using labelled data.
    """

    def __init__(
        self,
        intent_config: Optional[Dict[str, Any]] = None,
        cortex_config: Optional[Dict[str, Any]] = None,
        seed: int = 37,
    ) -> None:
        self.intent_recognizer = IntentRecognizer(intent_config or {})
        self.language_cortex = LanguageCortex(cortex_config or {})
        self.word_recognizer = WordRecognizer({}, SemanticNetwork({}))
        self.rng = np.random.default_rng(seed)
        self._intent_targets: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def fit(self, samples: Sequence[IntentSample]) -> IntentTrainingResult:
        """
        Update cortex weights and compute prototype vectors for each intent label.
        """
        if not samples:
            raise ValueError("No samples supplied for intent training.")

        label_counts: Dict[str, int] = {}
        prototype_sums: Dict[str, np.ndarray] = {}
        affect_counts: Dict[str, Dict[str, int]] = {}

        for sample in samples:
            tokens = simple_tokenize(sample.text)
            if not tokens:
                continue

            self._ingest_tokens(tokens)

            context_vector = self.language_cortex.encode_tokens(tokens)
            label = sample.intent
            target_vec = self._intent_target(label, context_vector.shape[0])
            self.language_cortex.update_from_example(tokens, target_vec)

            label_counts[label] = label_counts.get(label, 0) + 1
            proto_sum = prototype_sums.get(label)
            if proto_sum is None:
                prototype_sums[label] = context_vector.copy()
            else:
                prototype_sums[label] = proto_sum + context_vector

            if sample.emotion:
                affect_bucket = affect_counts.setdefault(sample.emotion, {})
                for token in tokens:
                    affect_bucket[token] = affect_bucket.get(token, 0) + 1

        prototypes: Dict[str, List[float]] = {}
        for label, summed in prototype_sums.items():
            count = label_counts.get(label, 1)
            averaged = summed / max(count, 1)
            prototypes[label] = averaged.astype(float).tolist()

        self.intent_recognizer.load_prototypes(prototypes)

        cortex_targets = {
            label: vector.astype(float).tolist() for label, vector in self._intent_targets.items()
        }

        return IntentTrainingResult(
            label_counts=label_counts,
            prototypes=prototypes,
            cortex_targets=cortex_targets,
        )

    # ------------------------------------------------------------------ #
    def export_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the trainer state."""
        return {
            "intent_prototypes": self.intent_recognizer.export_prototypes(),
            "intent_targets": {label: vector.astype(float).tolist() for label, vector in self._intent_targets.items()},
            "language_cortex": _serialize_cortex(self.language_cortex),
        }

    def save_state(self, path: Path) -> None:
        path = Path(path)
        data = self.export_state()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    def _intent_target(self, label: str, dimension: int) -> np.ndarray:
        vector = self._intent_targets.get(label)
        if vector is not None and vector.shape[0] == dimension:
            return vector
        vector = self.rng.normal(0.0, 1.0, dimension)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            vector /= norm
        self._intent_targets[label] = vector
        return vector

    def _ingest_tokens(self, tokens: Sequence[str]) -> None:
        for token in tokens:
            self.word_recognizer.add_word(token, list(token), concept=token)
            self.word_recognizer.activate_word(token, amount=0.1)


class AffectTrainer:
    """
    Build a simple emotion lexicon by counting word occurrences per label.
    """

    def fit(self, samples: Sequence[IntentSample]) -> AffectLexicon:
        word_counts: Dict[str, Dict[str, int]] = {}
        totals: Dict[str, int] = {}

        for sample in samples:
            if not sample.emotion:
                continue
            tokens = simple_tokenize(sample.text)
            if not tokens:
                continue
            bucket = word_counts.setdefault(sample.emotion, {})
            for token in tokens:
                bucket[token] = bucket.get(token, 0) + 1
            totals[sample.emotion] = totals.get(sample.emotion, 0) + len(tokens)

        return AffectLexicon(word_counts=word_counts, total_counts=totals)


class LanguageCortexTrainer:
    """
    Unsupervised fine-tuning for the language cortex on raw text corpora.
    """

    def __init__(self, cortex: Optional[LanguageCortex] = None) -> None:
        self.language_cortex = cortex or LanguageCortex()

    def fit_unsupervised(self, texts: Iterable[str], epochs: int = 1) -> None:
        """
        Perform a lightweight self-distillation pass: encode tokens and reinforce
        the cortex to reproduce its own context vector.
        """
        for _ in range(max(1, epochs)):
            for text in texts:
                tokens = simple_tokenize(text)
                if not tokens:
                    continue
                context = self.language_cortex.encode_tokens(tokens)
                if not context.size:
                    continue
                self.language_cortex.update_from_example(tokens, context)

    def export_state(self) -> Dict[str, Any]:
        return _serialize_cortex(self.language_cortex)

    def save_state(self, path: Path) -> None:
        path = Path(path)
        data = self.export_state()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Helper serialisation                                                        #
# --------------------------------------------------------------------------- #


def _serialize_cortex(cortex: LanguageCortex) -> Dict[str, Any]:
    """Convert LanguageCortex parameters to JSON-friendly lists."""
    return {
        "embedding_dim": cortex.embedding_dim,
        "hidden_dim": cortex.hidden_dim,
        "learning_rate": cortex.learning_rate,
        "word_embeddings": {word: vec.astype(float).tolist() for word, vec in cortex.word_embeddings.items()},
        "W_enc": cortex.W_enc.astype(float).tolist(),
        "W_proj": cortex.W_proj.astype(float).tolist(),
        "W_out": cortex.W_out.astype(float).tolist(),
        "context_state": cortex.context_state.astype(float).tolist(),
    }


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train internal language components without external LLMs.")
    parser.add_argument("--dataset", type=Path, required=True, help="CSV or JSONL file with text,intent[,emotion] rows.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write intent training summary (JSON).",
    )
    parser.add_argument(
        "--state-output",
        type=Path,
        help="Optional path to save the full trainer state (intent prototypes + cortex weights).",
    )
    parser.add_argument(
        "--affect-output",
        type=Path,
        help="Optional path to export the affect lexicon (JSON).",
    )
    parser.add_argument(
        "--unsupervised-corpus",
        type=Path,
        help="Optional newline-delimited text file for additional unsupervised cortex tuning.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of unsupervised epochs when --unsupervised-corpus is provided.")

    args = parser.parse_args(argv)

    samples = load_intent_dataset(args.dataset)
    trainer = IntentTrainer()
    training_result = trainer.fit(samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(training_result.to_json(), encoding="utf-8")

    if args.state_output:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_state(args.state_output)

    if args.affect_output:
        affect = AffectTrainer().fit(samples)
        args.affect_output.parent.mkdir(parents=True, exist_ok=True)
        args.affect_output.write_text(affect.to_json(), encoding="utf-8")

    if args.unsupervised_corpus and args.unsupervised_corpus.exists():
        texts = [
            line.strip()
            for line in args.unsupervised_corpus.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if texts:
            cortex_trainer = LanguageCortexTrainer(trainer.language_cortex)
            cortex_trainer.fit_unsupervised(texts, epochs=max(1, args.epochs))
            # Save updated cortex state if requested.
            if args.state_output:
                trainer.save_state(args.state_output)


__all__ = [
    "IntentSample",
    "IntentTrainingResult",
    "AffectLexicon",
    "load_intent_dataset",
    "IntentTrainer",
    "AffectTrainer",
    "LanguageCortexTrainer",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
