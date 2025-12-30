"""Utilities for preparing data prior to model retraining.

This module provides a small ``DataPipeline`` class that performs simple
augmentation and synthetic sample generation for both text and image data.
The aim is to offer hook points for richer pipelines while keeping the
implementation lightweight and dependency free.

Example
-------
>>> import pandas as pd
>>> from backend.ml.data_pipeline import DataPipeline
>>> df = pd.DataFrame({"input": ["hello world"]})
>>> DataPipeline().process(df)
          input
0   hello world
1   world hello
2  hello world [synthetic]
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class DataPipeline:
    """Apply basic augmentation and synthesis to datasets.

    Text augmentation shuffles token order while image augmentation performs a
    horizontal flip. Synthetic samples are created by appending a marker for
    text or generating random noise for images. These operations are purposely
    simple but demonstrate where more sophisticated logic can be inserted.
    """

    def augment_text(self, text: str) -> str:
        tokens = text.split()
        random.shuffle(tokens)
        return " ".join(tokens)

    def synthesize_text(self, text: str) -> str:
        return f"{text} [synthetic]"

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        return np.fliplr(image)

    def synthesize_image(self, shape: tuple[int, int, int]) -> np.ndarray:
        return np.random.rand(*shape)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new dataframe with augmented and synthetic samples.

        The pipeline looks for ``input`` (text) and ``image`` columns. For each
        record two additional rows are produced: one augmented and one
        synthetic. Other columns are copied verbatim.
        """

        rows: List[pd.Series] = []
        for _, row in df.iterrows():
            rows.append(row)
            if "input" in row and isinstance(row["input"], str):
                aug = row.copy()
                aug["input"] = self.augment_text(row["input"])
                rows.append(aug)
                syn = row.copy()
                syn["input"] = self.synthesize_text(row["input"])
                rows.append(syn)
            if "image" in row and isinstance(row["image"], np.ndarray):
                aug_img = row.copy()
                aug_img["image"] = self.augment_image(row["image"])
                rows.append(aug_img)
                syn_img = row.copy()
                syn_img["image"] = self.synthesize_image(row["image"].shape)
                rows.append(syn_img)
        return pd.DataFrame(rows)
