import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.neuroplasticity import Neuroplasticity


def test_adapt_connections_uses_stdp_rule():
    plasticity = Neuroplasticity()
    pre = np.array([0.2, 0.4])
    post = np.array([0.5, 0.1])

    update = plasticity.adapt_connections(pre, post)

    expected = post - pre
    assert np.allclose(update, expected)
