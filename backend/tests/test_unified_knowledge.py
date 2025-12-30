import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.knowledge.unified import KnowledgeSource, UnifiedKnowledgeBase
from backend.memory.long_term import LongTermMemory


class DummyEmbedder:
    """Minimal embedder mapping tokens to 2D vectors for testing."""

    vocab = {
        "cat": np.array([1.0, 0.0]),
        "feline": np.array([1.0, 0.0]),
        "dog": np.array([0.0, 1.0]),
        "canine": np.array([0.0, 1.0]),
    }

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        vectors = []
        for sent in sentences:
            vec = np.zeros(2)
            for word, rep in self.vocab.items():
                if word in sent:
                    vec += rep
            vectors.append(vec.tolist())
        return vectors[0] if len(vectors) == 1 else vectors


def test_semantic_query(tmp_path):
    memory = LongTermMemory(tmp_path / "mem.db")
    kb = UnifiedKnowledgeBase(embedder=DummyEmbedder(), memory=memory)
    kb.add_source(KnowledgeSource(name="animals", data={"cat": "A small domesticated feline"}))

    results = kb.query("feline pet", semantic=True)
    assert "animals:cat" in results
    memory.close()


def test_multi_source_aggregation():
    kb = UnifiedKnowledgeBase()
    kb.add_source(KnowledgeSource(name="bio", data={"evolution": "Biological evolution in species"}))
    kb.add_source(KnowledgeSource(name="tech", data={"evolution": "Process in technology improvement"}))

    results = kb.query("evolution")
    assert results == {
        "bio": "Biological evolution in species",
        "tech": "Process in technology improvement",
    }

