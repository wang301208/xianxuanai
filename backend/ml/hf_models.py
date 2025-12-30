from __future__ import annotations

"""Wrappers for loading common Transformer models.

This module provides convenience functions for downloading and
initialising GPT, BERT and ViT models from the HuggingFace ``transformers``
library.  Each loader returns the model together with its corresponding
pre-processing object such as a tokenizer or feature extractor.
"""

from typing import Tuple


def load_gpt(model_name: str = "gpt2") -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """Load a GPT-like causal language model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_bert(model_name: str = "bert-base-uncased") -> Tuple["BertModel", "BertTokenizer"]:
    """Load a BERT encoder and tokenizer."""
    from transformers import BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return model, tokenizer


def load_vit(
    model_name: str = "google/vit-base-patch16-224",
) -> Tuple["ViTModel", "ViTFeatureExtractor"]:
    """Load a Vision Transformer model and feature extractor."""
    from transformers import ViTFeatureExtractor, ViTModel

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    return model, feature_extractor


__all__ = ["load_gpt", "load_bert", "load_vit"]
