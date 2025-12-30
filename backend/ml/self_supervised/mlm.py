from __future__ import annotations

"""Utilities for masked language model (MLM) pretraining."""

from typing import Dict, Sequence

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import PreTrainedTokenizerBase
except Exception:  # pragma: no cover - transformers may be missing
    PreTrainedTokenizerBase = object  # type: ignore


def prepare_mlm_inputs(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    mask_probability: float = 0.15,
) -> Dict[str, "torch.Tensor"]:
    """Tokenize ``texts`` and create MLM training inputs."""

    if torch is None:  # pragma: no cover - runtime dependency check
        raise ImportError("torch is required for prepare_mlm_inputs")

    encoding = tokenizer(list(texts), return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"]
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_probability)
    special_tokens_mask = input_ids.eq(getattr(tokenizer, "pad_token_id", -100))
    mask_indices = torch.bernoulli(probability_matrix).bool() & ~special_tokens_mask
    input_ids[mask_indices] = getattr(tokenizer, "mask_token_id")
    labels[~mask_indices] = -100
    encoding["input_ids"] = input_ids
    encoding["labels"] = labels
    return encoding
