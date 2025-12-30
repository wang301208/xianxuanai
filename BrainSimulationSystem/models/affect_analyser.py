"""
Internal affect (emotion & tone) analyser.

This module replaces GPT-based sentiment cues with lightweight heuristics that
combine lexicon lookups, punctuation patterns and intent hints. It outputs a
compact ``AffectAnalysisResult`` containing sentiment polarity, detected
emotions, tone descriptors and supporting evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class AffectAnalysisResult:
    polarity: str
    score: float
    emotions: List[str]
    tone: List[str]
    confidence: float
    evidence: Dict[str, int]


class AffectAnalyser:
    """
    Heuristic affect analyser built on lexicon rules and punctuation cues.
    """

    def __init__(self, params: Optional[Dict[str, Iterable[str]]] = None) -> None:
        self.params = params or {}

        self.positive_words = set(
            params.get(
                "positive_words",
                [
                    "good",
                    "great",
                    "excellent",
                    "nice",
                    "love",
                    "hope",
                    "thanks",
                    "thank",
                    "appreciate",
                    "happy",
                    "pleased",
                    "awesome",
                    "wonderful",
                ],
            )
        )
        self.negative_words = set(
            params.get(
                "negative_words",
                [
                    "bad",
                    "terrible",
                    "awful",
                    "hate",
                    "angry",
                    "annoyed",
                    "upset",
                    "issue",
                    "problem",
                    "fail",
                    "broken",
                    "error",
                ],
            )
        )
        self.anger_words = set(
            params.get(
                "anger_words",
                ["angry", "furious", "outraged", "mad", "furious", "rage"],
            )
        )
        self.sorrow_words = set(
            params.get(
                "sorrow_words",
                ["sad", "unhappy", "depressed", "sorry", "regret"],
            )
        )
        self.polite_words = set(
            params.get(
                "polite_words",
                ["please", "kindly", "would", "could", "appreciate"],
            )
        )
        self.urgent_words = set(
            params.get(
                "urgent_words",
                ["asap", "urgent", "immediately", "now", "quick", "hurry"],
            )
        )
        self.uncertainty_words = set(
            params.get(
                "uncertainty_words",
                ["maybe", "perhaps", "unsure", "uncertain", "guess"],
            )
        )
        self.intensifiers = set(
            params.get(
                "intensifiers",
                ["very", "really", "extremely", "super", "totally", "absolutely"],
            )
        )
        self.diminishers = set(
            params.get(
                "diminishers",
                ["slightly", "somewhat", "maybe", "kind of", "kinda"],
            )
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def analyse(
        self,
        text: str,
        tokens: Sequence[str],
        normalized_tokens: Sequence[str],
        intent_label: str,
    ) -> AffectAnalysisResult:
        """
        Produce a sentiment/tone estimate for the given text.

        Parameters
        ----------
        text:
            Raw input string.
        tokens:
            Original tokens (before lowercasing).
        normalized_tokens:
            Lowercase tokens aligned to ``tokens``.
        intent_label:
            Intent inferred by the intent recogniser (used as a hint for tone).
        """

        score = 0.0
        emotion_hits: Dict[str, int] = {"positive": 0, "negative": 0, "anger": 0, "sorrow": 0}
        token_count = 0
        last_was_intensifier = False

        for idx, token in enumerate(normalized_tokens):
            if not token:
                continue
            token_count += 1
            if token in self.intensifiers:
                last_was_intensifier = True
                continue
            multiplier = 1.5 if last_was_intensifier else 1.0
            last_was_intensifier = False

            if token in self.positive_words:
                score += 1.0 * multiplier
                emotion_hits["positive"] += 1
            elif token in self.negative_words:
                score -= 1.0 * multiplier
                emotion_hits["negative"] += 1
            if token in self.anger_words:
                emotion_hits["anger"] += 1
            if token in self.sorrow_words:
                emotion_hits["sorrow"] += 1
            if token in self.diminishers:
                score *= 0.8

        exclamations = text.count("!")
        question_marks = text.count("?")
        uppercase_emphasis = sum(1 for tok in tokens if tok.isalpha() and tok.isupper() and len(tok) > 1)

        score += exclamations * 0.2
        score -= question_marks * 0.05  # interrogatives often seek info rather than express positivity
        score += uppercase_emphasis * 0.15

        polarity = "neutral"
        if score > 0.4:
            polarity = "positive"
        elif score < -0.4:
            polarity = "negative"

        emotions: List[str] = []
        if emotion_hits["positive"] > emotion_hits["negative"] and polarity == "positive":
            emotions.append("joy")
        if emotion_hits["negative"] > 0 and polarity == "negative":
            emotions.append("discontent")
        if emotion_hits["anger"] > 0:
            emotions.append("anger")
        if emotion_hits["sorrow"] > 0:
            emotions.append("sadness")

        tone_tags: List[str] = []
        normalized_full = " ".join(normalized_tokens)
        if any(tok in self.polite_words for tok in normalized_tokens):
            tone_tags.append("polite")
        if any(tok in self.urgent_words for tok in normalized_tokens) or exclamations >= 2:
            tone_tags.append("urgent")
        if uppercase_emphasis > 0 or exclamations > 0:
            tone_tags.append("emphatic")
        if any(tok in self.uncertainty_words for tok in normalized_tokens) or "?" in text:
            tone_tags.append("uncertain")
        if intent_label == "command" and "polite" not in tone_tags:
            tone_tags.append("directive")
        if intent_label == "question":
            tone_tags.append("inquisitive")

        nonzero_hits = emotion_hits["positive"] + emotion_hits["negative"]
        confidence = 0.35 + min(0.65, abs(score) * 0.25 + nonzero_hits * 0.1 + exclamations * 0.05)

        evidence = {
            "positive": emotion_hits["positive"],
            "negative": emotion_hits["negative"],
            "anger": emotion_hits["anger"],
            "sorrow": emotion_hits["sorrow"],
            "exclamations": exclamations,
            "upper_tokens": uppercase_emphasis,
            "token_count": token_count,
        }

        return AffectAnalysisResult(
            polarity=polarity,
            score=float(round(score, 3)),
            emotions=list(dict.fromkeys(emotions)),
            tone=list(dict.fromkeys(tone_tags)),
            confidence=float(round(confidence, 3)),
            evidence=evidence,
        )
