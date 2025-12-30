from __future__ import annotations

from BrainSimulationSystem.models.language_processing import SemanticNetwork, WordRecognizer


def test_symbol_concept_links_strengthen_with_coactivation():
    network = SemanticNetwork({"hebbian": {"enabled": True}})
    recognizer = WordRecognizer(
        {
            "activation_threshold": 0.2,
            "hebbian": {
                "enabled": True,
                "potentiation": 0.2,
                "decay": 0.05,
                "max_weight": 2.0,
                "min_weight": 0.0,
                "coactivation_threshold": 0.2,
            },
        },
        network,
    )

    recognizer.add_word("Sky", ["s", "k", "y"], frequency=0.2, concept="heaven")
    symbol_node = recognizer._symbol_node("Sky")  # noqa: SLF001 - intentional test hook

    # Co-activate symbol and concept to strengthen their bridge
    network.activate_concept("heaven", amount=0.25)
    recognizer.activate_word("Sky", amount=0.3)

    strengthened = network.relation_strength(symbol_node, "heaven")
    assert strengthened > 0.2

    # When neither side is active, the link should decay
    recognizer._apply_symbol_concept_plasticity(  # noqa: SLF001 - intentional test hook
        symbol_node,
        "heaven",
        symbol_activation=0.0,
        concept_activation=0.0,
    )
    weakened = network.relation_strength(symbol_node, "heaven")
    assert weakened < strengthened
