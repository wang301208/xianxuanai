from __future__ import annotations

from modules.skills.generator import SkillCodeGenerator
from modules.skills.registry import SkillSpec


def test_skill_prompt_includes_reference_material() -> None:
    spec = SkillSpec(
        name="Vector Similarity Search",
        description="Return the top-k most similar vectors.",
        execution_mode="local",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    generator = SkillCodeGenerator(llm_client=None)

    prompt = generator._build_prompt(
        spec,
        include_tests=False,
        references=[
            {
                "title": "ChromaDB docs",
                "url": "https://docs.trychroma.com/",
                "snippet": "Collection.query returns ids/documents/distances.",
            }
        ],
    )

    assert "Reference material" in prompt
    assert "https://docs.trychroma.com/" in prompt


def test_skill_prompt_includes_few_shot_examples() -> None:
    spec = SkillSpec(
        name="Dijkstra",
        description="Compute shortest paths in a weighted graph.",
        execution_mode="local",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    generator = SkillCodeGenerator(llm_client=None)

    prompt = generator._build_prompt(
        spec,
        include_tests=False,
        references=None,
        few_shot_examples={
            "signatures": ["def dijkstra(graph, source):"],
            "snippets": ["Initialize distances[source]=0; use priority queue; relax edges."],
            "examples": [{"payload": {"graph": {"A": {}}, "source": "A"}, "expected": {"distances": {"A": 0}}}],
        },
    )

    assert "Few-shot examples" in prompt
    assert "def dijkstra" in prompt
