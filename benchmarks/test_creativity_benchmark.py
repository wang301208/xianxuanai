from backend.creative_engine import CrossModalCreativeEngine
from modules.metrics.creative_evaluator import CreativeEvaluator
from modules.optimization.meta_learner import MetaLearner


class DummyAligner:
    def align(self, embedding, vector_type):
        return []


def dummy_encoder(prompt: str):
    return [0.0]


def text_generator(prompt: str, concepts):
    return prompt


def image_generator(prompt: str, concepts):
    return f"image:{prompt}"


def make_engine():
    evaluator = CreativeEvaluator()
    meta = MetaLearner(["text", "image"])
    engine = CrossModalCreativeEngine(
        aligner=DummyAligner(),
        encoders={"text": dummy_encoder, "image": dummy_encoder},
        generators={"text": text_generator, "image": image_generator},
        evaluator=evaluator,
        meta_learner=meta,
    )
    return engine, meta


def test_creativity_feedback():
    engine, meta = make_engine()
    result1 = engine.generate("hello hello", ["text", "image"])
    score1 = result1["text"]["creative_score"]
    weight_before = meta.weights["text"]
    result2 = engine.generate("hello unique world", ["text", "image"])
    score2 = result2["text"]["creative_score"]
    weight_after = meta.weights["text"]
    s1 = (score1.novelty + score1.usefulness) / 2
    s2 = (score2.novelty + score2.usefulness) / 2
    assert s2 >= s1
    assert weight_after > weight_before
