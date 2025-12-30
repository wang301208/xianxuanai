from __future__ import annotations

"""Sentiment analysis domain adapter using a simple lexicon."""

from .core import DomainAdapter, register_adapter


class SentimentDomainAdapter(DomainAdapter):
    """Classify sentiment as positive, negative or neutral."""

    POSITIVE = {"love", "happy", "great", "good", "awesome"}
    NEGATIVE = {"hate", "sad", "bad", "terrible", "awful"}

    def process(self, query: str) -> str:
        tokens = query.lower().split()
        pos = sum(1 for t in tokens if t in self.POSITIVE)
        neg = sum(1 for t in tokens if t in self.NEGATIVE)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"


register_adapter("sentiment", SentimentDomainAdapter)
