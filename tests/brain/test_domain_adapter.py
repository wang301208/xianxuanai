"""Tests for the DomainAdapter system."""

import os
import sys

# Ensure repository root on path for 'modules' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.brain.domain_adaptation import DomainAdapterManager


def test_math_domain_adapter() -> None:
    mgr = DomainAdapterManager("math")
    assert mgr.process("2 + 2") == "4"
    assert mgr.process("3 * (4 + 1)") == "15"


def test_sentiment_domain_adapter() -> None:
    mgr = DomainAdapterManager("sentiment")
    assert mgr.process("I love sunny days") == "positive"
    assert mgr.process("This is bad and terrible") == "negative"


def test_switching_domains() -> None:
    mgr = DomainAdapterManager("math")
    assert mgr.process("5 - 2") == "3"
    mgr.set_domain("sentiment")
    assert mgr.process("I am happy") == "positive"
