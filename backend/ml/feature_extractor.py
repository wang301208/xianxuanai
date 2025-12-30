"""Utilities for converting raw inputs into numerical feature vectors."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Any

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold

try:  # Optional dependencies
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - dependency may be missing at runtime
    SentenceTransformer = None  # type: ignore

try:  # Optional dependencies
    import networkx as nx
    from node2vec import Node2Vec
except Exception:  # pragma: no cover - dependency may be missing at runtime
    nx = None  # type: ignore
    Node2Vec = None  # type: ignore

try:
    import open_clip
except Exception:  # pragma: no cover - dependency may be missing at runtime
    open_clip = None


class FeatureExtractor:
    """Convert text into numerical feature vectors.

    ``FeatureExtractor`` supports classic TF-IDF features as well as sentence
    embeddings obtained via ``sentence-transformers``. Optional dimensionality
    reduction with PCA and variance-based feature selection can be enabled via
    the corresponding flags.
    """

    def __init__(
        self,
        method: str = "tfidf",
        *,
        use_pca: bool = False,
        n_components: int | None = None,
        use_feature_selection: bool = False,
        var_threshold: float = 0.0,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.method = method
        self.use_pca = use_pca
        self.use_feature_selection = use_feature_selection
        self.model_name = model_name

        if method == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif method == "sentence":
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers package is required for sentence embeddings"
                )
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

        self.pca = PCA(n_components=n_components) if use_pca else None
        self.selector = (
            VarianceThreshold(var_threshold) if use_feature_selection else None
        )

    # ------------------------------------------------------------------
    def _postprocess(self, X: np.ndarray) -> np.ndarray:
        if self.selector is not None:
            X = self.selector.fit_transform(X)
        if self.pca is not None:
            X = self.pca.fit_transform(X)
        return X

    def _postprocess_transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector is not None:
            X = self.selector.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    def fit_transform(self, logs: Iterable[str]):
        """Fit the extractor on ``logs`` and return feature matrix."""

        texts = list(logs)
        if self.method == "tfidf":
            X = self.vectorizer.fit_transform(texts)
            X = X.toarray() if (self.use_pca or self.use_feature_selection) else X
        else:
            X = self.model.encode(texts, convert_to_numpy=True)
        if isinstance(X, np.ndarray):
            X = self._postprocess(X)
        return X

    def transform(self, logs: Iterable[str]):
        """Transform ``logs`` using the fitted extractor."""

        texts = list(logs)
        if self.method == "tfidf":
            X = self.vectorizer.transform(texts)
            X = X.toarray() if (self.use_pca or self.use_feature_selection) else X
        else:
            X = self.model.encode(texts, convert_to_numpy=True)
        if isinstance(X, np.ndarray):
            X = self._postprocess_transform(X)
        return X

    def save(self, path: str) -> None:
        """Serialise extractor state to ``path`` using ``joblib``."""

        state = {
            "method": self.method,
            "vectorizer": getattr(self, "vectorizer", None),
            "model_name": self.model_name,
            "pca": self.pca,
            "selector": self.selector,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """Load a previously saved extractor from ``path``."""

        state = joblib.load(path)
        instance = cls(
            method=state.get("method", "tfidf"),
            use_pca=state.get("pca") is not None,
            n_components=None,
            use_feature_selection=state.get("selector") is not None,
            var_threshold=0.0,
            model_name=state.get("model_name", "all-MiniLM-L6-v2"),
        )
        if instance.method == "tfidf" and state.get("vectorizer") is not None:
            instance.vectorizer = state["vectorizer"]
        elif instance.method == "sentence":
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers package is required to load sentence extractor"
                )
            instance.model = SentenceTransformer(instance.model_name)
        instance.pca = state.get("pca")
        instance.selector = state.get("selector")
        return instance


class TimeSeriesFeatureExtractor:
    """Extract features from time-series data.

    Generates sliding window statistics and optionally appends FFT components.
    """

    def __init__(self, window_size: int = 5, apply_fft: bool = False) -> None:
        self.window_size = window_size
        self.apply_fft = apply_fft

    def _extract(self, series: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(series), dtype=float)
        if arr.size < self.window_size:
            windows = arr
        else:
            windows = np.lib.stride_tricks.sliding_window_view(arr, self.window_size)
            windows = windows.mean(axis=1)
        feats = windows if windows.ndim == 1 else windows.reshape(-1)
        if self.apply_fft:
            fft_vals = np.abs(np.fft.rfft(arr))
            feats = np.concatenate([feats, fft_vals])
        return feats

    def fit_transform(self, series_list: Iterable[Iterable[float]]) -> np.ndarray:
        return np.array([self._extract(s) for s in series_list])

    transform = fit_transform


class GraphFeatureExtractor:
    """Extract node embeddings from graphs using Node2Vec."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def _extract(self, graph: Any) -> np.ndarray:
        if nx is None or Node2Vec is None:
            raise ImportError("networkx and node2vec packages are required for graph features")
        node2vec = Node2Vec(graph, dimensions=self.dimensions, quiet=True)
        model = node2vec.fit()
        embeddings = np.array([model.wv[str(node)] for node in graph.nodes()])
        return embeddings.mean(axis=0)

    def fit_transform(self, graphs: Iterable[Any]) -> np.ndarray:
        return np.array([self._extract(g) for g in graphs])

    transform = fit_transform


class CLIPFeatureExtractor:
    """Wrapper around a pretrained CLIP model for multimodal features."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ) -> None:
        if open_clip is None:
            raise ImportError("open_clip package is required for CLIPFeatureExtractor")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()

    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(image)
        return feats.squeeze(0)

    def extract_text_features(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return feats.squeeze(0)

    def extract(self, image: Image.Image, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both image and text feature vectors for ``image`` and ``text``."""

        return self.extract_image_features(image), self.extract_text_features(text)
