"""
Internal intent recognition utilities.

This module provides a small rule-augmented intent classifier that relies on
local linguistic cues, the project's semantic network activations, and the
internal language cortex vectors instead of external LLM providers. It follows
an augmented two-stage approach:

1. Fast heuristics identify high-confidence cases such as explicit questions or
   imperative commands.
2. A lightweight scorer combines semantic activation patterns, lexical evidence,
   and cortex embeddings to choose among the supported intent labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .language_processing import SemanticNetwork, SyntaxProcessor

# Core vocabularies for intent cues.
QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "where",
    "when",
    "which",
    "who",
    "whom",
    "whose",
    "\u662f\u5426",
    "\u5417",
}
COMMAND_TRIGGERS = {
    "please",
    "start",
    "stop",
    "do",
    "run",
    "open",
    "close",
    "execute",
    "perform",
}
GREETING_WORDS = {"hi", "hello", "hey", "greetings"}
AFFIRMATION_WORDS = {"yes", "sure", "ok", "okay"}


@dataclass
class IntentResult:
    """Return container for intent recognition."""

    label: str
    confidence: float
    source: str
    scores: Dict[str, float]
    details: Dict[str, Any]


class IntentRecognizer:
    """
    Combine heuristics, semantic activations, and cortex embeddings to infer intent.

    Parameters
    ----------
    params:
        Optional configuration dictionary. Recognised keys:

        - ``fallback_intent``: label returned when no evidence is available.
        - ``rule_confidence``: minimal confidence for rule-based shortcuts.
        - ``semantic_intent_nodes``: mapping of label -> list of semantic node
          names whose activations should boost that label.
        - ``cortex_weight``: additive weight applied to cortex similarity scores.
        - ``cortex_momentum``: momentum factor for updating intent prototypes.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.fallback_intent = self.params.get("fallback_intent", "statement")
        self.rule_confidence = float(self.params.get("rule_confidence", 0.82))
        self.semantic_intent_nodes: Dict[str, Sequence[str]] = self.params.get(
            "semantic_intent_nodes",
            {
                "question": ["question", "query", "ask"],
                "command": ["command", "action", "task"],
                "greeting": ["greeting", "social"],
                "statement": ["inform", "report"],
            },
        )
        self.cortex_weight = float(self.params.get("cortex_weight", 0.15))
        self.prototype_momentum = float(self.params.get("cortex_momentum", 0.75))
        self.intent_embeddings: Dict[str, np.ndarray] = {}
        self.auto_expand_terms = bool(self.params.get("auto_expand_terms", True))
        self.auto_expand_threshold = float(self.params.get("auto_expand_threshold", 0.78))
        self.max_learned_terms = int(self.params.get("max_learned_terms", 32))
        base_labels = set(self.semantic_intent_nodes.keys()) | {self.fallback_intent}
        self.learned_terms: Dict[str, List[str]] = {label: [] for label in base_labels}
        custom_stop_words = set(self.params.get("stop_words", []))
        self.learned_stop_words = (
            custom_stop_words
            | QUESTION_WORDS
            | COMMAND_TRIGGERS
            | GREETING_WORDS
            | AFFIRMATION_WORDS
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def classify(
        self,
        text: str,
        tokens: Sequence[str],
        semantic_network: SemanticNetwork,
        syntax_processor: SyntaxProcessor,
        cortex_vector: Optional[Sequence[float]] = None,
    ) -> IntentResult:
        """
        Infer the intent label for ``text`` using internal signals.

        Parameters
        ----------
        text:
            Raw input string.
        tokens:
            Tokenised representation of the input (lower case, punctuation stripped).
        semantic_network:
            Active semantic network instance with the latest concept activations.
        syntax_processor:
            Syntax processor used for higher-level patterns.
        cortex_vector:
            Optional embedding produced by the LanguageCortex for the current input.
        """

        token_list = [token for token in tokens if token]
        rule_hit = self._rule_based(text, token_list)
        if rule_hit is not None:
            label, confidence, evidence = rule_hit
            self._maybe_learn_terms(label, token_list, confidence)
            return IntentResult(
                label=label,
                confidence=confidence,
                source="rule",
                scores={label: confidence},
                details={"evidence": evidence},
            )

        scores, feature_details = self._score_candidates(
            text,
            token_list,
            semantic_network,
            syntax_processor,
        )

        cortex_bonus: Dict[str, float] = {}
        used_cortex = False
        vector: Optional[np.ndarray] = None
        if cortex_vector is not None:
            vector = np.asarray(list(cortex_vector), dtype=float)
            if vector.size:
                used_cortex = True
                labels = set(scores) | set(self.semantic_intent_nodes.keys())
                labels.add(self.fallback_intent)
                for label in labels:
                    scores.setdefault(label, 0.0)
                    prototype = self._get_intent_prototype(label, vector.shape[0])
                    similarity = self._cosine_similarity(vector, prototype)
                    cortex_bonus[label] = similarity
                    scores[label] += self.cortex_weight * (1.0 + similarity)
            else:
                vector = None
        feature_details["cortex_vector_used"] = used_cortex

        selected_label = max(scores, key=scores.get) if scores else self.fallback_intent
        total = sum(scores.values()) or 1.0
        confidence = scores.get(selected_label, 0.0) / total

        if confidence < 0.2:
            selected_label = self.fallback_intent
            confidence = 0.2

        feature_details["token_count"] = len(token_list)
        if cortex_bonus:
            feature_details["cortex_bonus"] = {k: float(round(v, 4)) for k, v in cortex_bonus.items()}
        if used_cortex and vector is not None and vector.size:
            similarity = cortex_bonus.get(selected_label, 0.0)
            feature_details["cortex_similarity"] = float(round(similarity, 4))
            prototype = self._update_intent_prototype(selected_label, vector)
            feature_details["cortex_prototype_norm"] = float(round(float(np.linalg.norm(prototype)), 4))

        intent_result = IntentResult(
            label=selected_label,
            confidence=float(round(confidence, 3)),
            source="propagation",
            scores={k: float(round(v, 4)) for k, v in scores.items()},
            details=feature_details,
        )
        self._maybe_learn_terms(intent_result.label, token_list, intent_result.confidence)
        return intent_result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _rule_based(
        self,
        text: str,
        tokens: Sequence[str],
    ) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """Fast rules for obvious patterns."""
        lowered_text = text.strip().lower()
        if not lowered_text:
            return None

        if lowered_text.endswith("?") or "?" in lowered_text:
            return (
                "question",
                self.rule_confidence,
                {"pattern": "question_mark"},
            )

        if any(token in QUESTION_WORDS for token in tokens):
            return (
                "question",
                self.rule_confidence,
                {"pattern": "question_word"},
            )

        if tokens:
            first_token = tokens[0]
            if first_token in COMMAND_TRIGGERS:
                return ("command", self.rule_confidence, {"pattern": "command_trigger"})

            if first_token not in GREETING_WORDS and not lowered_text.endswith("."):
                # Imperative heuristic: leading verb-like token with optional please.
                if self._looks_verb_like(first_token):
                    return ("command", 0.78, {"pattern": "leading_verb"})

        if tokens and tokens[0] in GREETING_WORDS:
            return ("greeting", 0.75, {"pattern": "greeting"})

        learned_label = self._match_learned_keywords(tokens)
        if learned_label:
            return (
                learned_label,
                0.7,
                {"pattern": "learned_keyword"},
            )

        return None

    def _score_candidates(
        self,
        text: str,
        tokens: Sequence[str],
        semantic_network: SemanticNetwork,
        syntax_processor: SyntaxProcessor,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Combine lexical cues and semantic activations to produce scores."""
        lowered_text = text.lower()
        scores: Dict[str, float] = {
            "question": 0.05,
            "command": 0.05,
            "statement": 0.05,
            "greeting": 0.01,
        }
        details: Dict[str, Any] = {}

        question_word_hits = sum(1 for token in tokens if token in QUESTION_WORDS)
        command_trigger_hits = sum(1 for token in tokens if token in COMMAND_TRIGGERS)
        greeting_hits = sum(1 for token in tokens if token in GREETING_WORDS)
        affirmation_hits = sum(1 for token in tokens if token in AFFIRMATION_WORDS)

        if tokens:
            scores["question"] += question_word_hits / len(tokens)
            scores["command"] += command_trigger_hits / len(tokens)
            scores["greeting"] += greeting_hits / len(tokens)
            scores["statement"] += affirmation_hits / len(tokens) * 0.4

        if "?" in lowered_text:
            scores["question"] += 0.35
        if lowered_text.endswith("!"):
            scores["command"] += 0.15

        imperative_hint = self._looks_imperative(tokens, syntax_processor)
        if imperative_hint:
            scores["command"] += 0.25

        semantic_stats = self._semantic_activation_scores(semantic_network)
        for label, boost in semantic_stats.items():
            scores[label] = scores.get(label, 0.0) + boost

        details.update(
            {
                "question_word_hits": question_word_hits,
                "command_trigger_hits": command_trigger_hits,
                "greeting_hits": greeting_hits,
                "affirmation_hits": affirmation_hits,
                "imperative_hint": imperative_hint,
                "semantic_boost": semantic_stats,
            }
        )
        return scores, details

    def _semantic_activation_scores(
        self,
        semantic_network: SemanticNetwork,
    ) -> Dict[str, float]:
        """Aggregate semantic activation energy for intent-related nodes."""
        boosts: Dict[str, float] = {}
        if not hasattr(semantic_network, "nodes"):
            return boosts

        for label, node_names in self.semantic_intent_nodes.items():
            activation = 0.0
            for name in node_names:
                node = semantic_network.nodes.get(name)
                if not node:
                    continue
                activation += float(node.get("activation", 0.0))
            if node_names:
                activation /= max(len(node_names), 1)
            if activation > 0.0:
                boosts[label] = activation
        return boosts

    def _match_learned_keywords(self, tokens: Sequence[str]) -> Optional[str]:
        if not tokens or not self.learned_terms:
            return None
        for token in tokens:
            candidate = token.lower()
            for label, terms in self.learned_terms.items():
                if candidate in terms:
                    return label
        return None

    def _maybe_learn_terms(
        self,
        label: str,
        tokens: Sequence[str],
        confidence: float,
    ) -> None:
        if (
            not self.auto_expand_terms
            or confidence < self.auto_expand_threshold
            or not tokens
        ):
            return

        bucket = self.learned_terms.setdefault(label, [])
        for token in tokens:
            candidate = token.lower()
            if len(candidate) <= 3 or candidate in self.learned_stop_words:
                continue
            if candidate in bucket:
                continue
            bucket.append(candidate)
            if len(bucket) > self.max_learned_terms:
                bucket.pop(0)

    @staticmethod
    def _looks_verb_like(token: str) -> bool:
        """Simple heuristic to guess verb-like tokens."""
        if not token:
            return False
        if token.endswith("e") or token.endswith("en"):
            return True
        if token.endswith("ing"):
            return True
        if token.startswith("do") or token.startswith("run"):
            return True
        return False

    def _looks_imperative(
        self,
        tokens: Sequence[str],
        syntax_processor: SyntaxProcessor,
    ) -> bool:
        """
        Detect basic imperative mood using syntax cues.

        This relies on limited POS knowledge, so it remains heuristic.
        """
        if not tokens:
            return False

        first = tokens[0]
        if first in {"please", "let"}:
            return True

        pos_tags = getattr(syntax_processor, "pos_tags", {})
        if first in pos_tags.get("V", []):
            return True

        if len(tokens) > 1 and tokens[1] in {"please"}:
            return True

        return self._looks_verb_like(first)

    def _get_intent_prototype(self, label: str, dimension: int) -> np.ndarray:
        """Retrieve or initialise the prototype vector for an intent label."""
        prototype = self.intent_embeddings.get(label)
        if prototype is None or prototype.shape[0] != dimension:
            prototype = np.zeros(dimension, dtype=float)
            self.intent_embeddings[label] = prototype
        return prototype

    def _update_intent_prototype(self, label: str, vector: np.ndarray) -> np.ndarray:
        """Update the prototype for ``label`` using the supplied vector."""
        prototype = self._get_intent_prototype(label, vector.shape[0])
        updated = self.prototype_momentum * prototype + (1.0 - self.prototype_momentum) * vector
        self.intent_embeddings[label] = updated
        return updated

    @staticmethod
    def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Compute cosine similarity with numerical safeguards."""
        norm_product = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        if norm_product <= 1e-8:
            return 0.0
        return float(np.dot(vector_a, vector_b) / norm_product)

    # ------------------------------------------------------------------ #
    # Persistence utilities                                              #
    # ------------------------------------------------------------------ #
    def load_prototypes(self, mapping: Optional[Dict[str, Sequence[float]]]) -> None:
        """
        Load intent prototype vectors from an external mapping.

        Parameters
        ----------
        mapping:
            Dictionary mapping label -> sequence of floats representing the
            prototype vector for that intent.
        """
        if not mapping:
            self.intent_embeddings = {}
            return
        loaded: Dict[str, np.ndarray] = {}
        for label, values in mapping.items():
            array = np.asarray(list(values), dtype=float)
            if array.size == 0:
                continue
            loaded[label] = array
        self.intent_embeddings = loaded

    def export_prototypes(self) -> Dict[str, List[float]]:
        """Return a JSON-serialisable copy of the current intent prototypes."""
        exported: Dict[str, List[float]] = {}
        for label, vector in self.intent_embeddings.items():
            exported[label] = vector.astype(float).tolist()
        return exported

    def reset_prototypes(self) -> None:
        """Clear any stored cortex prototype vectors."""
        self.intent_embeddings = {}


__all__ = ["IntentRecognizer", "IntentResult"]
