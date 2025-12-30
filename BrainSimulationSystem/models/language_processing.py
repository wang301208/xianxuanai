from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from BrainSimulationSystem.models.sequence_model import LightweightSequenceModel


class PhonemeProcessor:
    """音素处理，支持简单的音素缓冲与特征映射。"""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.buffer_size = int(self.params.get("buffer_size", 8))
        self.phoneme_buffer: List[str] = []
        self.phoneme_features: Dict[str, Dict[str, float]] = self.params.get(
            "phoneme_features",
            {
                "a": {"type": "vowel", "front": 0.5, "height": 0.1},
                "e": {"type": "vowel", "front": 0.9, "height": 0.4},
                "i": {"type": "vowel", "front": 1.0, "height": 0.9},
                "o": {"type": "vowel", "front": 0.2, "height": 0.4},
                "u": {"type": "vowel", "front": 0.1, "height": 0.8},
            },
        )

    def process_phoneme(self, symbol: str) -> Dict[str, float]:
        phoneme = symbol.lower()
        if len(phoneme) != 1:
            phoneme = phoneme[0:1]
        self.phoneme_buffer.append(phoneme)
        if len(self.phoneme_buffer) > self.buffer_size:
            self.phoneme_buffer.pop(0)
        return self.phoneme_features.get(phoneme, {"type": "consonant"})


# --------------------------------------------------------------------------- #
# Auditory / lexicon discovery                                                #
# --------------------------------------------------------------------------- #


class AuditoryLexiconLearner:
    """简易的音素-词汇发现器，基于重复的音素序列生成原型词。"""

    def __init__(
        self,
        word_recognizer: "WordRecognizer",
        semantic_network: "SemanticNetwork",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.word_recognizer = word_recognizer
        self.semantic_network = semantic_network
        self.params = params or {}
        self.max_length = int(self.params.get("max_length", 5))
        self.min_occurrences = int(self.params.get("min_occurrences", 3))
        self.sequence_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.learned_words: Dict[Tuple[str, ...], str] = {}
        self.memory_size = int(self.params.get("memory_size", 64))
        self.recent_sequences: List[Tuple[str, ...]] = []

    def ingest_phonemes(self, phonemes: Sequence[str], context_tokens: Sequence[str]) -> None:
        normalized: List[str] = []
        for p in phonemes:
            if p is None:
                continue
            if isinstance(p, (int, float)):
                symbol = str(int(p)) if float(p).is_integer() else f"{p:.2f}"
                normalized.append(symbol)
                continue
            phoneme = str(p).strip()
            if not phoneme:
                continue
            lowered = phoneme.lower()
            normalized.append(lowered if len(lowered) == 1 else lowered)
        for start in range(len(normalized)):
            for end in range(start + 1, min(len(normalized), start + 1 + self.max_length)):
                seq = tuple(normalized[start:end])
                self.sequence_counts[seq] += 1
                if self.sequence_counts[seq] >= self.min_occurrences:
                    self._register_proto_word(seq, context_tokens)

        self.recent_sequences.append(tuple(normalized))
        if len(self.recent_sequences) > self.memory_size:
            self.recent_sequences.pop(0)

    def _register_proto_word(self, seq: Tuple[str, ...], context_tokens: Sequence[str]) -> None:
        if seq in self.learned_words:
            return

        candidate = "".join(seq)
        label = candidate
        suffix = 1
        while label in self.word_recognizer.mental_lexicon:
            label = f"{candidate}_{suffix}"
            suffix += 1

        frequency = float(self.sequence_counts.get(seq, 1))
        self.word_recognizer.add_word(label, seq, frequency=frequency, concept=label)
        self.learned_words[seq] = label

        if not self.semantic_network:
            return

        self.semantic_network.add_node(label, {"source": "auditory_proto"})
        for token in context_tokens:
            ctx = token.lower()
            if not ctx:
                continue
            if ctx not in self.semantic_network.nodes:
                self.semantic_network.add_node(ctx, {"source": "context"})
            self.semantic_network.add_relation(label, ctx, "context", strength=0.3)


class SemanticGrounder:
    """记录词汇与上下文的共现，用于逐步绑定语义。

    The grounder now logs episodic co-occurrences with timestamps so that
    hippocampal/short-term memory components can trigger ``replay`` to
    reinforce associations. External modules (e.g., affect or tutor feedback)
    can call ``add_feedback`` or ``add_tutor_label`` before invoking
    ``replay`` to adjust relation strengths during consolidation.
    """

    def __init__(self, semantic_network: "SemanticNetwork", params: Optional[Dict[str, Any]] = None) -> None:
        self.semantic_network = semantic_network
        self.params = params or {}
        self.relation_weight = float(self.params.get("relation_weight", 0.1))
        self.min_count = int(self.params.get("min_count", 2))
        self.feedback_weight = float(self.params.get("feedback_weight", 0.25))
        self.replay_decay = float(self.params.get("replay_decay", 0.98))
        self.max_episode_memory = int(self.params.get("max_episode_memory", 500))
        self.associations: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.episodes: List[Dict[str, Any]] = []

    def observe(self, word: str, context: Sequence[str], reward: float = 0.0) -> None:
        head = word.lower()
        if not head:
            return

        filtered_context: List[str] = []
        for token in context:
            dep = token.lower()
            if not dep or dep == head:
                continue
            filtered_context.append(dep)
            self.associations[head][dep] += 1
            count = self.associations[head][dep]
            if count < self.min_count:
                continue

            self._reinforce_relation(head, dep, count, reward)

        if filtered_context:
            self._record_episode(head, filtered_context, reward)

    def _record_episode(self, head: str, context: List[str], feedback: float) -> None:
        """Persist a time-stamped episode for later replay."""

        self.episodes.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "word": head,
                "context": list(context),
                "feedback": float(feedback),
            }
        )
        if len(self.episodes) > self.max_episode_memory:
            self.episodes.pop(0)

    def _reinforce_relation(self, head: str, dep: str, count: int, feedback: float = 0.0) -> None:
        if not self.semantic_network:
            return

        self.semantic_network.add_node(head, {"source": "grounder"})
        self.semantic_network.add_node(dep, {"source": "context"})
        feedback_bonus = feedback * self.feedback_weight
        strength = min(1.0, max(0.0, count * self.relation_weight + feedback_bonus))
        self.semantic_network.add_relation(head, dep, "co_occurs", strength=strength)
        self.semantic_network.add_relation(dep, head, "co_occurs", strength=strength)

    def replay(self, limit: Optional[int] = None, decay: Optional[float] = None) -> None:
        """
        Re-apply stored co-occurrence episodes to reinforce semantic edges.

        Hippocampal/short-term memory systems can call this after pushing new
        episodes or providing feedback, optionally limiting the number of
        most recent episodes replayed.
        """

        if not self.semantic_network:
            return

        selected = self.episodes if limit is None else self.episodes[-limit:]
        decay_factor = decay if decay is not None else self.replay_decay

        for idx, episode in enumerate(reversed(selected)):
            weight = decay_factor ** idx
            head = episode["word"]
            feedback = episode.get("feedback", 0.0) * weight
            for dep in episode.get("context", []):
                count = self.associations[head].get(dep, 1)
                self._reinforce_relation(head, dep, count, feedback)

    def add_feedback(self, word: str, context: Sequence[str], reward: float) -> None:
        """
        Attach an external feedback signal (reward/penalty) to matching episodes.

        Modules such as affect analyzers or memory evaluators can call this
        before ``replay`` to bias strengthening or weakening of associations.
        """

        head = word.lower()
        targets = {c.lower() for c in context if c}
        for episode in self.episodes:
            if episode.get("word") != head:
                continue
            if targets and not targets.intersection(set(episode.get("context", []))):
                continue
            episode["feedback"] = float(episode.get("feedback", 0.0) + reward)

    def add_tutor_label(
        self,
        word: str,
        label: str,
        relation_type: str = "tutor_label",
        strength: float = 0.4,
    ) -> None:
        """Apply a corrective label from a tutor and adjust the relation strength."""

        head = word.lower()
        corrected = label.lower()
        if not self.semantic_network:
            return

        self.semantic_network.add_node(head, {"source": "grounder"})
        self.semantic_network.add_node(corrected, {"source": "tutor"})
        self.semantic_network.add_relation(head, corrected, relation_type, strength=strength)
        self.semantic_network.add_relation(corrected, head, relation_type, strength=strength)

    def export_associations(self) -> Dict[str, Dict[str, int]]:
        return {head: dict(deps) for head, deps in self.associations.items()}


# --------------------------------------------------------------------------- #
# Lexicon + semantics                                                         #
# --------------------------------------------------------------------------- #


class WordRecognizer:
    """词汇识别器，衔接心理词典与语义网络。"""

    def __init__(self, params=None, semantic_network: Optional["SemanticNetwork"] = None) -> None:
        self.params = params or {}
        self.semantic_network = semantic_network
        self.syntax_processor: Optional["SyntaxProcessor"] = None
        self.mental_lexicon = defaultdict(dict)
        self.activation_decay = float(self.params.get("activation_decay", 0.1))
        self.activation_threshold = float(self.params.get("activation_threshold", 0.5))
        self.active_words: Dict[str, float] = {}
        hebbian_cfg = self.params.get("hebbian", {})
        if not isinstance(hebbian_cfg, dict) and self.semantic_network is not None:
            hebbian_cfg = getattr(self.semantic_network, "params", {}).get("hebbian", {})
        self.hebbian_enabled = bool(hebbian_cfg.get("enabled", False))
        self.hebbian_params = {
            "potentiation": float(hebbian_cfg.get("potentiation", 0.02)),
            "decay": float(hebbian_cfg.get("decay", 0.01)),
            "max_weight": float(hebbian_cfg.get("max_weight", 5.0)),
            "min_weight": float(hebbian_cfg.get("min_weight", 0.0)),
            "coactivation_threshold": float(hebbian_cfg.get("coactivation_threshold", 0.3)),
        }

    def set_syntax_processor(self, syntax_processor: "SyntaxProcessor") -> None:
        self.syntax_processor = syntax_processor

    def add_word(
        self,
        word: str,
        phonemes: Sequence[str],
        frequency: float = 1.0,
        concept: Optional[str] = None,
        synonyms: Optional[Sequence[str]] = None,
    ) -> None:
        lemma = self._infer_lemma(word)
        concept_label = concept or lemma

        entry = self.mental_lexicon[word]
        entry["phonemes"] = list(phonemes)
        entry["frequency"] = frequency
        entry.setdefault("activation", 0.0)
        entry["lemma"] = lemma
        entry["concept"] = concept_label
        entry.setdefault("synonyms", set()).update(synonyms or [])

        if self.semantic_network is not None and concept_label:
            if concept_label not in self.semantic_network.nodes:
                self.semantic_network.add_node(concept_label, {"source": "lexicon"})
            symbol_node = self._symbol_node(word)
            self.semantic_network.add_node(symbol_node, {"source": "lexicon_symbol", "type": "symbol"})
            initial_strength = float(np.clip(frequency, 0.05, 1.0))
            self.semantic_network.add_relation(symbol_node, concept_label, "symbol_concept", strength=initial_strength)
            self.semantic_network.add_relation(concept_label, symbol_node, "symbol_concept", strength=initial_strength)
            self.semantic_network.activate_concept(concept_label, amount=0.05)

        self._update_pos_tags(word)

    def register_synonym(self, word: str, concept: str) -> None:
        entry = self.mental_lexicon[word]
        entry["concept"] = concept
        if self.semantic_network is not None and concept not in self.semantic_network.nodes:
            self.semantic_network.add_node(concept, {"source": "lexicon_syn"})

    def activate_word(self, word: str, amount: float = 0.1) -> None:
        if word not in self.mental_lexicon:
            return
        entry = self.mental_lexicon[word]
        entry["activation"] = min(1.0, entry.get("activation", 0.0) + amount)
        self.active_words[word] = entry["activation"]

        concept = entry.get("concept")
        if concept and self.semantic_network is not None:
            symbol_node = self._symbol_node(word)
            self.semantic_network.add_node(symbol_node, {"source": "lexicon_symbol", "type": "symbol"})
            self.semantic_network.activate_concept(concept, amount=amount * 1.2)
            concept_activation = self.semantic_network.nodes.get(concept, {}).get("activation", 0.0)
            self._apply_symbol_concept_plasticity(
                symbol_node,
                concept,
                symbol_activation=entry["activation"],
                concept_activation=concept_activation,
            )

        for synonym in entry.get("synonyms", []):
            syn_entry = self.mental_lexicon.get(synonym)
            if syn_entry is None:
                continue
            syn_entry["activation"] = min(1.0, syn_entry.get("activation", 0.0) + amount * 0.6)
            self.active_words[synonym] = syn_entry["activation"]
            syn_concept = syn_entry.get("concept")
            if syn_concept and self.semantic_network is not None:
                self.semantic_network.activate_concept(syn_concept, amount=amount * 0.6)

    def decay_activations(self) -> None:
        for word in list(self.active_words.keys()):
            entry = self.mental_lexicon[word]
            entry["activation"] = max(0.0, entry.get("activation", 0.0) - self.activation_decay)
            if entry["activation"] < self.activation_threshold:
                self.active_words.pop(word, None)
            concept = entry.get("concept")
            if concept and self.semantic_network is not None:
                symbol_node = self._symbol_node(word)
                concept_activation = self.semantic_network.nodes.get(concept, {}).get("activation", 0.0)
                self._apply_symbol_concept_plasticity(
                    symbol_node,
                    concept,
                    symbol_activation=entry.get("activation", 0.0),
                    concept_activation=concept_activation,
                )

    def recognize_word(self, phonemes: Sequence[str]) -> Optional[str]:
        best_word = None
        best_score = 0.0
        for word, info in self.mental_lexicon.items():
            match = self._match_phonemes(phonemes, info.get("phonemes", []))
            score = match * info.get("frequency", 1.0) * (1 + info.get("activation", 0.0))
            if score > best_score:
                best_score = score
                best_word = word
        if best_word and best_score > self.activation_threshold:
            self.activate_word(best_word, amount=0.3)
            return best_word
        return None

    def _match_phonemes(self, input_phonemes: Sequence[str], word_phonemes: Sequence[str]) -> float:
        if not input_phonemes or not word_phonemes:
            return 0.0
        match = sum(1 for a, b in zip(input_phonemes, word_phonemes) if a == b)
        return match / max(len(input_phonemes), len(word_phonemes))

    def _infer_lemma(self, word: str) -> str:
        lower = word.lower()
        for suffix in ("ingly", "edly", "ing", "ed", "ly", "es", "s"):
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                return lower[: -len(suffix)]
        return lower

    def _update_pos_tags(self, word: str) -> None:
        if not self.syntax_processor:
            return
        lower = word.lower()
        candidates: List[str] = []
        if lower.endswith("ly"):
            candidates.append("Adv")
        elif lower.endswith("ing"):
            candidates.append("Gerund")
        elif lower.endswith("ed"):
            candidates.append("V")
        elif lower.endswith("ous") or lower.endswith("ive") or lower.endswith("al"):
            candidates.append("Adj")
        elif lower.endswith("s") and len(lower) > 3:
            candidates.append("N")

        for tag in candidates:
            vocab = self.syntax_processor.lexicon.get(tag)
            if vocab is None:
                continue
            if lower not in vocab:
                self.syntax_processor.lexicon[tag] = tuple(list(vocab) + [lower])

    @staticmethod
    def _symbol_node(word: str) -> str:
        return f"symbol:{word.lower()}"

    def _apply_symbol_concept_plasticity(
        self,
        symbol_node: str,
        concept: str,
        *,
        symbol_activation: float,
        concept_activation: float,
    ) -> None:
        if not self.hebbian_enabled or self.semantic_network is None:
            return

        coactive = (
            symbol_activation >= self.hebbian_params["coactivation_threshold"]
            and concept_activation >= self.hebbian_params["coactivation_threshold"]
        )
        potentiation = self.hebbian_params["potentiation"] if coactive else 0.0
        decay = self.hebbian_params["decay"] if not coactive else 0.0

        for head, dep in ((symbol_node, concept), (concept, symbol_node)):
            self.semantic_network.adjust_relation_strength(
                head,
                dep,
                potentiation=potentiation,
                decay=decay,
                min_strength=self.hebbian_params["min_weight"],
                max_strength=self.hebbian_params["max_weight"],
                relation_type="symbol_concept",
            )


class SemanticNetwork:
    """简化语义网络，维护概念节点和关系并支持激活扩散。"""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.activation_spread = float(self.params.get("activation_spread", 0.3))
        self.activation_decay = float(self.params.get("activation_decay", 0.05))

    def add_node(self, concept: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        if concept not in self.nodes:
            self.nodes[concept] = {
                "activation": 0.0,
                "attributes": dict(attributes or {}),
            }
        else:
            if attributes:
                self.nodes[concept]["attributes"].update(attributes)

    def add_relation(self, head: str, dependent: str, relation_type: str, strength: float = 1.0) -> None:
        if head not in self.nodes or dependent not in self.nodes:
            return
        rel = self.relations[head].get(dependent)
        if rel:
            rel["strength"] = max(rel.get("strength", 0.0), strength)
            rel.setdefault("types", set()).add(relation_type)
        else:
            self.relations[head][dependent] = {"types": {relation_type}, "strength": strength}

    def relation_strength(self, head: str, dependent: str) -> float:
        relation = self.relations.get(head, {}).get(dependent)
        if relation is None:
            return 0.0
        try:
            return float(relation.get("strength", 0.0))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0

    def adjust_relation_strength(
        self,
        head: str,
        dependent: str,
        *,
        potentiation: float = 0.0,
        decay: float = 0.0,
        min_strength: float = 0.0,
        max_strength: float = 5.0,
        relation_type: Optional[str] = None,
    ) -> float:
        if head not in self.nodes or dependent not in self.nodes:
            return 0.0

        relation = self.relations[head].get(dependent)
        if relation is None:
            relation = {"types": set()}
            self.relations[head][dependent] = relation

        strength = self.relation_strength(head, dependent)
        if relation_type:
            relation.setdefault("types", set()).add(relation_type)

        if potentiation:
            strength += float(potentiation)
        elif decay:
            strength *= max(0.0, 1.0 - float(decay))

        strength = max(min_strength, min(max_strength, strength))
        relation["strength"] = strength
        return strength

    def activate_concept(self, concept: str, amount: float = 0.2) -> None:
        if concept not in self.nodes:
            return
        node = self.nodes[concept]
        node["activation"] = min(1.0, node.get("activation", 0.0) + amount)
        for dep, rel in self.relations.get(concept, {}).items():
            propagated = amount * rel.get("strength", 1.0) * self.activation_spread
            if propagated > 0.01:
                self.activate_concept(dep, propagated)

    def decay_activations(self) -> None:
        for node in self.nodes.values():
            node["activation"] = max(0.0, node.get("activation", 0.0) - self.activation_decay)

    def get_related_concepts(self, concept: str, relation_type: Optional[str] = None) -> List[str]:
        relations = self.relations.get(concept, {})
        results: List[str] = []
        for dep, info in relations.items():
            if relation_type is None or relation_type in info.get("types", {}):
                results.append(dep)
        return results

    def get_most_activated(self, threshold: float = 0.3) -> List[str]:
        return [c for c, info in self.nodes.items() if info.get("activation", 0.0) >= threshold]

    def prune_relations(
        self,
        *,
        min_strength: float = 0.1,
        drop_isolated: bool = False,
    ) -> int:
        """Remove weak relations and optionally drop isolated nodes."""

        removed = 0
        for head in list(self.relations.keys()):
            deps = self.relations.get(head, {})
            for dep in list(deps.keys()):
                strength = deps.get(dep, {}).get("strength", 0.0)
                try:
                    strength_value = float(strength)
                except (TypeError, ValueError):
                    strength_value = 0.0
                if strength_value < float(min_strength):
                    deps.pop(dep, None)
                    removed += 1
            if not deps:
                self.relations.pop(head, None)

        if drop_isolated:
            connected = set(self.relations.keys())
            for deps in self.relations.values():
                connected.update(deps.keys())
            for node in list(self.nodes.keys()):
                if node not in connected:
                    self.nodes.pop(node, None)
        return removed


class SyntaxProcessor:
    """规则+启发式句法处理器，带有轻量级序列学习加权。"""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.lexicon: Dict[str, Tuple[str, ...]] = {
            "Det": ("the", "a", "an", "this", "that", "these", "those"),
            "Pronoun": ("i", "you", "he", "she", "it", "we", "they"),
            "N": ("dog", "cat", "system", "agent", "network", "response"),
            "V": ("is", "are", "run", "execute", "answer", "confirm", "acknowledge"),
            "Adj": ("new", "old", "urgent", "polite"),
            "Adv": ("quickly", "slowly", "politely"),
            "Gerund": ("running", "processing"),
            "Modal": ("can", "could", "will", "would", "should"),
            "Aux": ("do", "does", "did", "have", "has"),
            "P": ("in", "on", "with", "for"),
        }
        self.sequence_model = None
        self.grammar_inducer = None
        self.sequence_weight = float(self.params.get("sequence_weight", 0.55))
        self.grammar_weight = float(self.params.get("grammar_weight", 0.35))

    def set_sequence_model(self, sequence_model: Any) -> None:
        self.sequence_model = sequence_model

    def set_grammar_inducer(self, grammar_inducer: Any) -> None:
        self.grammar_inducer = grammar_inducer

    def parse_sentence(self, words: Sequence[str]) -> Dict[str, Any]:
        tokens = [w.lower() for w in words if w]
        pos_tags: List[str] = []
        for tok in tokens:
            pos_tags.append(self._tag_token(tok, pos_tags))
        tree: Dict[str, Any] = {"type": "S", "children": []}
        if not tokens:
            tree["pos"] = []
            tree["meta"] = {
                "tokens": [],
                "pos_tags": [],
                "dependency": {"root": None, "arcs": []},
            }
            return tree

        subject, predicate = self._split_subject_predicate(tokens, pos_tags)
        if subject:
            tree["children"].append({"type": "NP", "tokens": subject})
        if predicate:
            tree["children"].append({"type": "VP", "tokens": predicate})

        root_idx, arcs = self._infer_dependency_arcs(tokens, pos_tags)

        tree["pos"] = pos_tags  # backwards compatible
        tree["meta"] = {
            "tokens": tokens,
            "pos_tags": pos_tags,
            "dependency": {"root": root_idx, "arcs": arcs},
        }
        return tree

    def _tag_token(self, token: str, previous_tags: Optional[Sequence[str]] = None) -> str:
        previous_tags = list(previous_tags or [])
        heuristic_tag = "UNK"
        for pos, vocab in self.lexicon.items():
            if token in vocab:
                heuristic_tag = pos
                break
        else:
            if token.endswith("ly"):
                heuristic_tag = "Adv"
            elif token.endswith("ing"):
                heuristic_tag = "Gerund"
            elif token.endswith("ed"):
                heuristic_tag = "V"
            elif token.endswith("s") and len(token) > 3:
                # Prefer a 3rd-person singular verb if the lemma exists in our verb lexicon.
                lemma = token[:-1]
                if lemma and lemma in self.lexicon.get("V", ()):
                    heuristic_tag = "V"
                else:
                    heuristic_tag = "N"

        candidates: Dict[str, float] = defaultdict(float)
        candidates[heuristic_tag] += 0.45

        if self.sequence_model:
            predicted = self.sequence_model.predict_tags(previous_tags)
            for tag, prob in predicted.items():
                candidates[tag] += prob * self.sequence_weight

        if self.grammar_inducer:
            transition_probs = self.grammar_inducer.transition_probabilities(tuple(previous_tags))
            for tag, prob in transition_probs.items():
                candidates[tag] += prob * self.grammar_weight

        return max(candidates.items(), key=lambda item: item[1])[0]

    def _split_subject_predicate(self, tokens: Sequence[str], pos_tags: Sequence[str]) -> Tuple[List[str], List[str]]:
        if not tokens:
            return [], []
        try:
            verb_index = next(i for i, pos in enumerate(pos_tags) if pos in {"V", "Gerund", "Aux", "Modal"})
        except StopIteration:
            return list(tokens), []
        subject = list(tokens[:verb_index]) or list(tokens[:1])
        predicate = list(tokens[verb_index:])
        return subject, predicate

    def _infer_dependency_arcs(
        self,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
    ) -> Tuple[Optional[int], List[Tuple[int, int, str]]]:
        """Infer a lightweight dependency structure for downstream semantic grounding.

        Returns (root_index, arcs) where arcs are (head_index, dependent_index, relation).
        The head_index is -1 for the sentence root.
        """

        if not tokens:
            return None, []

        root_idx: Optional[int] = None
        for i, pos in enumerate(pos_tags):
            if pos in {"V", "Gerund"}:
                root_idx = i
                break
        if root_idx is None:
            for i, pos in enumerate(pos_tags):
                if pos in {"N", "Pronoun"}:
                    root_idx = i
                    break
        if root_idx is None:
            root_idx = 0

        arcs: List[Tuple[int, int, str]] = [(-1, int(root_idx), "root")]

        def _next_content_word(start: int) -> Optional[int]:
            for j in range(start, len(tokens)):
                if pos_tags[j] in {"N", "Pronoun", "ProperNoun", "UNK"}:
                    return j
            return None

        # Subject: select the closest content word before the root.
        subject_idx: Optional[int] = None
        for j in range(int(root_idx) - 1, -1, -1):
            if pos_tags[j] in {"N", "Pronoun", "ProperNoun", "UNK"}:
                subject_idx = j
                break
        if subject_idx is not None and subject_idx != root_idx:
            arcs.append((int(root_idx), int(subject_idx), "subject"))

        # Object: select the first content word after the root.
        object_idx = _next_content_word(int(root_idx) + 1)
        if object_idx is not None and object_idx != root_idx:
            arcs.append((int(root_idx), int(object_idx), "object"))

        # Attach auxiliaries/modals near the root.
        for i, pos in enumerate(pos_tags):
            if i == root_idx:
                continue
            if pos in {"Aux", "Modal"} and abs(i - int(root_idx)) <= 2:
                arcs.append((int(root_idx), i, "aux"))

        # Adverbs typically modify the verb phrase.
        for i, pos in enumerate(pos_tags):
            if i == root_idx:
                continue
            if pos == "Adv":
                arcs.append((int(root_idx), i, "modifier"))

        # Determiners/adjectives attach to the next noun-like token.
        for i, pos in enumerate(pos_tags):
            if pos not in {"Det", "Adj"}:
                continue
            noun_idx = _next_content_word(i + 1)
            if noun_idx is None:
                continue
            relation = "attribute" if pos == "Det" else "modifier"
            arcs.append((noun_idx, i, relation))

        # Simple prepositional phrase handling: attach preposition to root and its object to the preposition.
        for i, pos in enumerate(pos_tags):
            if pos != "P":
                continue
            if i != root_idx:
                arcs.append((int(root_idx), i, "attribute"))
            pobj = _next_content_word(i + 1)
            if pobj is not None and pobj != i:
                arcs.append((i, pobj, "object"))

        # De-duplicate while preserving order.
        deduped: List[Tuple[int, int, str]] = []
        seen = set()
        for arc in arcs:
            if arc in seen:
                continue
            seen.add(arc)
            deduped.append(arc)
        return int(root_idx), deduped


class GrammarInducer:
    """简易语法诱导器，跟踪POS转移以发现稳定句法模式。"""

    def __init__(
        self,
        syntax_processor: SyntaxProcessor,
        params: Optional[Dict[str, Any]] = None,
        *,
        sequence_model: Optional[LightweightSequenceModel] = None,
    ) -> None:
        self.syntax_processor = syntax_processor
        self.params = params or {}
        self.max_history = int(self.params.get("max_history", 2))
        self.minimum_count = int(self.params.get("minimum_count", 2))
        self.min_probability = float(self.params.get("min_probability", 0.2))
        self.transitions: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pattern_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.sequence_model = sequence_model
        self.storage_path = Path(self.params.get("storage_path", "BrainSimulationSystem/data/learned_grammar.json"))
        self.auto_persist = bool(self.params.get("auto_persist", True))
        self.syntax_processor.set_grammar_inducer(self)
        self._load_persisted_rules()

    def observe_sentence(self, tokens: Sequence[str]) -> None:
        if not tokens:
            return
        tags: List[str] = []
        for tok in tokens:
            tag = self.syntax_processor._tag_token(tok.lower(), tags)
            tags.append(tag)
        if not tags:
            return

        padded = ["<s>"] + tags + ["</s>"]
        for i in range(1, len(padded)):
            for history_len in range(1, min(self.max_history, i) + 1):
                history = tuple(padded[i - history_len : i])
                next_tag = padded[i]
                self.transitions[history][next_tag] += 1

        self.pattern_counts[tuple(tags)] += 1
        if self.sequence_model:
            normalized_tokens = [str(tok).lower() for tok in tokens if tok]
            self.sequence_model.observe_sequence(normalized_tokens, tags)

        if self.auto_persist:
            self.save_rules()

    def induce_rules(self, top_k: int = 8) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        for history, outcomes in self.transitions.items():
            total = sum(outcomes.values())
            if total < self.minimum_count:
                continue
            model_probs = self.sequence_model.predict_tags(list(history)) if self.sequence_model else {}
            for tag, count in outcomes.items():
                empirical = count / max(total, 1)
                blended = 0.6 * empirical + 0.4 * model_probs.get(tag, 0.0)
                if blended < self.min_probability:
                    continue
                rules.append({"history": history, "next": tag, "probability": blended})

        if self.sequence_model:
            rules.extend(self.sequence_model.get_rule_candidates(self.min_probability, top_k))

        rules.sort(key=lambda r: r["probability"], reverse=True)
        if top_k:
            rules = rules[:top_k]
        return rules

    def predict_next(self, recent_tags: Sequence[str]) -> Optional[str]:
        probabilities = self.transition_probabilities(tuple(recent_tags))
        if not probabilities:
            return None
        return max(probabilities.items(), key=lambda item: item[1])[0]

    def transition_probabilities(self, history: Tuple[str, ...]) -> Dict[str, float]:
        if not history:
            return {}
        outcomes = self.transitions.get(tuple(history)) or {}
        total = sum(outcomes.values()) or 1
        empirical = {tag: count / total for tag, count in outcomes.items()}
        model_probs = self.sequence_model.predict_tags(list(history)) if self.sequence_model else {}
        combined: Dict[str, float] = defaultdict(float)
        for tag, prob in empirical.items():
            combined[tag] += prob * 0.6
        for tag, prob in model_probs.items():
            combined[tag] += prob * 0.4
        return dict(combined)

    def save_rules(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "transitions": {
                    "::".join(history): dict(counts) for history, counts in self.transitions.items()
                },
                "patterns": {"::".join(pattern): count for pattern, count in self.pattern_counts.items()},
            }
            with self.storage_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:
            # Persistence is best-effort to avoid impacting runtime behaviour.
            return

    def _load_persisted_rules(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            return

        for key, counts in data.get("transitions", {}).items():
            history = tuple(key.split("::"))
            for tag, count in counts.items():
                self.transitions[history][tag] += int(count)
        for pattern, count in data.get("patterns", {}).items():
            self.pattern_counts[tuple(pattern.split("::"))] += int(count)


class LanguageGenerator:
    """模板+语义驱动的语言生成器。"""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.templates: Dict[str, List[str]] = self._load_templates(self.params.get("templates"))
        self.max_key_terms = int(self.params.get("max_key_terms", 4))
        self.append_key_terms = bool(self.params.get("append_key_terms", True))
        self.append_action_plan = bool(self.params.get("append_action_plan", True))

        self.sequence_model: Optional[LightweightSequenceModel] = None
        seq_cfg = self.params.get("sequence_generation", {})
        if not isinstance(seq_cfg, dict):
            seq_cfg = {}
        self.sequence_generation_enabled = bool(seq_cfg.get("enabled", False))
        self.sequence_generation_mode = str(seq_cfg.get("mode", "fallback")).lower()
        try:
            self.sequence_generation_min_vocab = int(seq_cfg.get("min_vocab", 32))
        except (TypeError, ValueError):
            self.sequence_generation_min_vocab = 32
        try:
            self.sequence_generation_max_tokens = int(seq_cfg.get("max_tokens", 18))
        except (TypeError, ValueError):
            self.sequence_generation_max_tokens = 18
        try:
            self.sequence_generation_top_k = int(seq_cfg.get("top_k", 8))
        except (TypeError, ValueError):
            self.sequence_generation_top_k = 8
        self.sequence_generation_deterministic = bool(seq_cfg.get("deterministic", True))

        if self.sequence_generation_min_vocab <= 0:
            self.sequence_generation_min_vocab = 32
        if self.sequence_generation_max_tokens <= 0:
            self.sequence_generation_max_tokens = 18
        if self.sequence_generation_top_k <= 0:
            self.sequence_generation_top_k = 8

    def set_sequence_model(self, sequence_model: Optional[LightweightSequenceModel]) -> None:
        self.sequence_model = sequence_model

    def generate_response(
        self,
        goal: Dict[str, Any],
        comprehension: Dict[str, Any],
        action_plan: Any,
        affect_info: Any,
        semantic_network: SemanticNetwork,
        template_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        intent = goal.get("intent") or comprehension.get("intent", "inform")
        summary = goal.get("reference") or comprehension.get("summary") or ""
        key_terms = goal.get("key_terms") or comprehension.get("key_terms", [])
        semantic_data = comprehension.get("semantic", {})
        tone = self._extract_field(affect_info, "tone", [])
        polarity = self._extract_field(affect_info, "polarity", "neutral")
        actions = self._extract_field(action_plan, "actions", [])
        primary_action = actions[0] if actions else None

        values = self._build_template_values(intent, summary, key_terms, primary_action, tone, polarity, semantic_data, comprehension)
        template_name, template = self._select_template(intent, tone, primary_action, values, template_hint)
        text = self._render_template(template, values)

        if not text:
            text = self._compose_from_semantics(semantic_network, semantic_data)

        if self.append_key_terms and values["key_terms_phrase"] and values["key_terms_phrase"].lower() not in text.lower():
            text = f"{text} Key points: {values['key_terms_phrase']}."

        if (
            self.append_action_plan
            and primary_action
            and self._verbalise_action(primary_action)
            and self._verbalise_action(primary_action).lower() not in text.lower()
        ):
            text = f"{text} Next step: {self._verbalise_action(primary_action)}."

        if self.sequence_generation_enabled and self.sequence_model is not None:
            sequence_sentence = self._generate_sequence_sentence(values, semantic_data)
            if sequence_sentence:
                mode = self.sequence_generation_mode
                if mode == "replace":
                    text = sequence_sentence
                elif mode == "append":
                    text = f"{text} {sequence_sentence}"
                elif mode == "fallback":
                    if template_name.startswith("fallback") or not text.strip():
                        text = sequence_sentence

        return {
            "text": self._finalise_sentence(text),
            "template": template_name,
            "values": values,
            "tone": tone,
            "polarity": polarity,
            "actions": actions,
        }

    def generate_utterance(self, semantic_network, activated_concepts):
        return self._compose_from_semantics(semantic_network, {"activation_map": {concept: 1.0 for concept in activated_concepts}})

    def _sequence_model_ready(self) -> bool:
        model = self.sequence_model
        if model is None:
            return False
        embeddings = getattr(model, "token_embeddings", None)
        try:
            vocab = len(embeddings) if isinstance(embeddings, dict) else 0
        except Exception:
            vocab = 0
        return vocab >= max(2, int(self.sequence_generation_min_vocab))

    def _generate_sequence_sentence(self, values: Dict[str, Any], semantic_data: Dict[str, Any]) -> str:
        if not self._sequence_model_ready():
            return ""

        seed: List[str] = []
        for key in ("subject", "action", "object"):
            value = values.get(key)
            token = str(value).strip().lower() if value else ""
            if token and token not in seed:
                seed.append(token)

        key_terms = values.get("key_terms_phrase")
        if isinstance(key_terms, str):
            for term in key_terms.split(","):
                token = term.strip().lower()
                if token and token not in seed:
                    seed.append(token)
                if len(seed) >= 4:
                    break

        if not seed:
            return ""

        tokens: List[str] = list(seed[:3])
        max_tokens = int(self.sequence_generation_max_tokens)
        top_k = int(self.sequence_generation_top_k)

        for _ in range(max(0, max_tokens - len(tokens))):
            context = tokens[-4:]
            token_probs, _ = self.sequence_model.predict_next(context)  # type: ignore[union-attr]
            if not token_probs:
                break

            candidates = [
                (tok, prob)
                for tok, prob in token_probs.items()
                if tok and not tok.startswith("<") and prob is not None
            ]
            candidates.sort(key=lambda item: item[1], reverse=True)
            if top_k and len(candidates) > top_k:
                candidates = candidates[:top_k]

            next_token = ""
            if candidates:
                if self.sequence_generation_deterministic:
                    next_token = str(candidates[0][0]).strip().lower()
                else:
                    weights = np.asarray([float(p) for _, p in candidates], dtype=float)
                    weights = weights / max(float(np.sum(weights)), 1e-12)
                    next_token = str(np.random.choice([tok for tok, _ in candidates], p=weights)).strip().lower()

            if not next_token:
                break
            if next_token in tokens[-2:]:
                break
            tokens.append(next_token)

        return " ".join(tokens).strip()

    def _load_templates(self, overrides: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        default = {
            "answer": [
                "Here is what I found: {reference}.",
                "{reference}.",
                "The key point is {key_terms_phrase}.",
            ],
            "confirm": [
                "Understood. I will {primary_action_phrase}.",
                "Acknowledged, {primary_action_phrase} now.",
            ],
            "inform": [
                "{summary_sentence}",
                "Noted. {summary_sentence}",
            ],
            "acknowledge": [
                "I hear you.",
                "Noted.",
            ],
            "greet_back": [
                "Hello! How can I assist you further?",
            ],
            "fallback": [
                "I am processing that information.",
            ],
        }
        if overrides:
            for key, value in overrides.items():
                default[key] = list(value)
        return default

    def _extract_field(self, obj: Any, field: str, default: Any) -> Any:
        if obj is None:
            return default
        if hasattr(obj, field):
            return getattr(obj, field)
        if isinstance(obj, dict):
            return obj.get(field, default)
        return default

    def _build_template_values(
        self,
        intent: str,
        summary: str,
        key_terms: Sequence[str],
        primary_action: Optional[str],
        tone: Sequence[str],
        polarity: str,
        semantic_data: Dict[str, Any],
        comprehension: Dict[str, Any],
    ) -> Dict[str, Any]:
        key_terms_phrase = ", ".join(key_terms[: self.max_key_terms])
        summary_sentence = summary if summary else comprehension.get("input", "")
        if summary_sentence and not summary_sentence.endswith((".", "!", "?")):
            summary_sentence = f"{summary_sentence}."

        primary_action_phrase = self._verbalise_action(primary_action) if primary_action else ""
        tone_phrase = ", ".join(tone)

        return {
            "intent": intent,
            "summary": summary,
            "summary_sentence": summary_sentence.strip(),
            "reference": summary or summary_sentence.strip(),
            "key_terms_phrase": key_terms_phrase,
            "primary_action": primary_action,
            "primary_action_phrase": primary_action_phrase,
            "tone_phrase": tone_phrase,
            "polarity": polarity,
            "subject": self._select_semantic_role(semantic_data, "subject"),
            "object": self._select_semantic_role(semantic_data, "object"),
            "action": self._select_semantic_role(semantic_data, "action"),
        }

    def _select_template(
        self,
        intent: str,
        tone: Sequence[str],
        primary_action: Optional[str],
        values: Dict[str, Any],
        preferred_template: Optional[str],
    ) -> Tuple[str, str]:
        intent_key = preferred_template or intent
        if intent == "command" and not preferred_template:
            intent_key = "confirm"
        elif intent == "question" and not preferred_template:
            intent_key = "answer"

        candidates = list(self.templates.get(intent_key, []))
        if not candidates:
            candidates = self.templates.get("fallback", ["I am processing that information."])

        selected = None
        for template in candidates:
            if "{primary_action_phrase}" in template and not values["primary_action_phrase"]:
                continue
            if "{reference}" in template and not values["reference"]:
                continue
            if "{key_terms_phrase}" in template and not values["key_terms_phrase"]:
                continue
            selected = template
            break
        if selected is None:
            selected = candidates[0]

        template_name = intent_key
        if "urgent" in tone:
            selected = f"{selected} I will prioritise this."
            template_name += "_urgent"
        return template_name, selected

    def _render_template(self, template: str, values: Dict[str, Any]) -> str:
        safe_values = {key: (val or "") for key, val in values.items()}
        try:
            return template.format(**safe_values).strip()
        except KeyError:

            return template

    def _compose_from_semantics(self, semantic_network: SemanticNetwork, semantic_data: Dict[str, Any]) -> str:
        activation_map = semantic_data.get("activation_map") or {}
        if not activation_map and hasattr(semantic_network, "nodes"):
            activation_map = {
                concept: info.get("activation", 0.0)
                for concept, info in semantic_network.nodes.items()
            }

        if not activation_map:
            return "I acknowledge the information."

        sorted_concepts = sorted(activation_map.items(), key=lambda item: item[1], reverse=True)
        core_concept = sorted_concepts[0][0]

        agent_candidates = semantic_network.get_related_concepts(core_concept, "subject")
        object_candidates = semantic_network.get_related_concepts(core_concept, "object")
        action_candidates = semantic_network.get_related_concepts(core_concept, "action")

        pieces = []
        if agent_candidates:
            pieces.append(agent_candidates[0])
        if action_candidates:
            pieces.append(action_candidates[0])
        if object_candidates:
            pieces.append(object_candidates[0])
        if not pieces:
            pieces = [core_concept]
        return " ".join(pieces)

    def _verbalise_action(self, action: Optional[str]) -> str:
        if not action:
            return ""
        mapping = {
            "answer_question": "answer the question",
            "execute_request": "carry out the request",
            "acknowledge": "acknowledge the update",
            "greet_back": "offer a greeting",
            "prioritise_request": "prioritise the request",
        }
        if action in mapping:
            return mapping[action]
        if action.startswith("execute_"):
            return f"execute {action.replace('execute_', '').replace('_', ' ')}"
        if action.startswith("confirm_"):
            return f"confirm {action.replace('confirm_', '').replace('_', ' ')}"
        if action.startswith("utilise_"):
            return f"use {action.replace('utilise_', '').replace('_', ' ')}"
        return action.replace("_", " ")

    def _select_semantic_role(self, semantic_data: Dict[str, Any], role: str) -> str:
        relations = semantic_data.get("relations") or []
        for relation in relations:
            if relation.get("relation") == role:
                return relation.get("dependent") or relation.get("head") or ""
        return ""

    def _finalise_sentence(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text = f"{text}."
        return text


class LanguageProcessor:
    """示例封装类，连接各处理模块并提供可视化数据。"""

    def __init__(self, params=None):
        self.params = params or {}

        # 语音处理模块
        self.phoneme_processor = PhonemeProcessor(self.params.get("phoneme", {}))

        # 语义处理模块与词汇识别
        self.semantic_network = SemanticNetwork(self.params.get("semantic", {}))
        self.word_recognizer = WordRecognizer(self.params.get("word", {}), self.semantic_network)
        self.auditory_learner = AuditoryLexiconLearner(
            self.word_recognizer,
            self.semantic_network,
            self.params.get("phoneme_discovery", {}),
        )
        self.semantic_grounder = SemanticGrounder(self.semantic_network, self.params.get("grounding", {}))

        # 句法处理模块
        self.sequence_model = LightweightSequenceModel(self.params.get("sequence_model", {}))
        self.syntax_processor = SyntaxProcessor(self.params.get("syntax", {}))
        self.syntax_processor.set_sequence_model(self.sequence_model)
        self.grammar_inducer = GrammarInducer(
            self.syntax_processor,
            self.params.get("grammar_induction", {}),
            sequence_model=self.sequence_model,
        )
        self.word_recognizer.set_syntax_processor(self.syntax_processor)

        # 语言生成模块
        self.language_generator = LanguageGenerator(self.params.get("generation", {}))
        if hasattr(self.language_generator, "set_sequence_model"):
            try:
                self.language_generator.set_sequence_model(self.sequence_model)
            except Exception:
                pass

        # 可视化状态
        self.visualization_data = {
            "phoneme_processing": [],
            "word_activation": {},
            "semantic_activation": {},
            "induced_grammar": [],
            "grounded_pairs": {},
            "syntax_tree": None,
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        raw_phonemes = [ch for ch in text if ch.strip()]
        phonemes = [self.phoneme_processor.process_phoneme(ch) for ch in raw_phonemes]
        tokens = text.split()
        self.auditory_learner.ingest_phonemes(raw_phonemes, tokens)
        for token in tokens:
            if token not in self.word_recognizer.mental_lexicon:
                self.word_recognizer.add_word(token, list(token), concept=token.lower())
            self.word_recognizer.activate_word(token, amount=0.2)
            self.semantic_grounder.observe(token, tokens)
        if tokens:
            self.grammar_inducer.observe_sentence(tokens)
        syntax_tree = self.syntax_processor.parse_sentence(tokens)
        self.visualization_data.update(
            {
                "phoneme_processing": phonemes,
                "word_activation": dict(self.word_recognizer.active_words),
                "semantic_activation": {
                    concept: info.get("activation", 0.0) for concept, info in self.semantic_network.nodes.items()
                },
                "induced_grammar": self.grammar_inducer.induce_rules(),
                "grounded_pairs": self.semantic_grounder.export_associations(),
                "syntax_tree": syntax_tree,
            }
        )
        return self.visualization_data

    def get_visualization_data(self) -> Dict[str, Any]:
        return self.visualization_data

