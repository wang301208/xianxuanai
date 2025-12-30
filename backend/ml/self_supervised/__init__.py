from __future__ import annotations

"""Self-supervised learning building blocks."""

from .augmentations import AugmentationPipeline
from .simclr import SimCLR
from .moco import MoCo
from .autoencoder import Autoencoder, reconstruction_loss
from .vae import VariationalAutoencoder, kl_divergence
from .mlm import prepare_mlm_inputs

__all__ = [
    "AugmentationPipeline",
    "SimCLR",
    "MoCo",
    "Autoencoder",
    "VariationalAutoencoder",
    "reconstruction_loss",
    "kl_divergence",
    "prepare_mlm_inputs",
]
