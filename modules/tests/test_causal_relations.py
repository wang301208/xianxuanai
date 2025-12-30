import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.common.causal_utils import add_causal_relation, get_effects
from modules.common import ConceptNode


def test_causal_relation_creation_and_retrieval():
    cause = ConceptNode(id="A", label="Cause")
    effect = ConceptNode(id="B", label="Effect")
    relation = add_causal_relation(cause, effect, weight=0.5)

    assert relation.cause == "A"
    assert relation.effect == "B"
    assert relation.weight == 0.5

    # ensure relation stored on both nodes
    assert relation in cause.causal_links
    assert relation in effect.causal_links

    effects = get_effects("A")
    assert effects == ["B"]
