from __future__ import annotations

from types import SimpleNamespace

from modules.knowledge.acquisition import KnowledgeAcquisitionManager


def _spec(name: str, parameters: dict) -> SimpleNamespace:
    return SimpleNamespace(name=name, parameters=parameters)


def _task(text: str) -> SimpleNamespace:
    return SimpleNamespace(description=text, objective=text)


def test_knowledge_acquisition_prefers_query_abilities_over_url_only() -> None:
    manager = KnowledgeAcquisitionManager(confidence_threshold=0.9)
    scrape = _spec("web_scrape", {"url": {}, "max_chars": {}})
    search = _spec("web_search", {"query": {}, "max_results": {}})

    override = manager.maybe_acquire(
        metadata={"confidence": 0.1},
        ability_specs=[scrape, search],
        task=_task("Explain Monte Carlo Tree Search."),
        current_selection=None,
    )

    assert override is not None
    assert override["next_ability"] == "web_search"
    assert override["ability_arguments"] == {"query": "Explain Monte Carlo Tree Search."}


def test_knowledge_acquisition_uses_web_scrape_when_url_present() -> None:
    manager = KnowledgeAcquisitionManager(confidence_threshold=0.9)
    scrape = _spec("web_scrape", {"url": {}, "max_chars": {}})
    search = _spec("web_search", {"query": {}, "max_results": {}})

    override = manager.maybe_acquire(
        metadata={"confidence": 0.1, "knowledge_url": "https://93.184.216.34/"},
        ability_specs=[search, scrape],
        task=_task("Fetch content."),
        current_selection=None,
    )

    assert override is not None
    assert override["next_ability"] == "web_scrape"
    assert override["ability_arguments"] == {"url": "https://93.184.216.34/"}


def test_knowledge_acquisition_prefers_search_and_scrape_composite() -> None:
    manager = KnowledgeAcquisitionManager(confidence_threshold=0.9)
    composite = _spec("web_search_and_scrape", {"query": {}, "max_sources": {}})
    search = _spec("web_search", {"query": {}, "max_results": {}})

    override = manager.maybe_acquire(
        metadata={"confidence": 0.1},
        ability_specs=[search, composite],
        task=_task("Summarize policy gradient methods."),
        current_selection=None,
    )

    assert override is not None
    assert override["next_ability"] == "web_search_and_scrape"

