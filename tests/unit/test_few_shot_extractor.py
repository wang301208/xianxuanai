from __future__ import annotations

from modules.knowledge.few_shot_extractor import extract_few_shot_material


def test_extracts_signatures_and_snippets_from_doc_sources() -> None:
    sources = [
        {
            "search": {"title": "Demo", "url": "https://example.com/doc"},
            "page": {
                "title": "Demo",
                "final_url": "https://example.com/doc",
                "code_blocks": [
                    "def foo(x: int) -> int:\n    return x + 1\n",
                    "Algorithm:\n  1) init\n  2) loop\n",
                ],
            },
        }
    ]

    extracted = extract_few_shot_material(sources, max_signatures=5, max_snippets=5)

    assert extracted["signatures"] == ["def foo(x: int) -> int:"]
    assert extracted["snippets"]
    assert "Algorithm" in extracted["snippets"][0]

