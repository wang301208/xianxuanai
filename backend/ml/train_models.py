"""Train baseline models on log data.

This script expects a CSV file with two columns:
``text`` containing the raw log message and ``target`` containing the numeric
value to predict.  It splits the dataset into train, validation and test sets,
trains a simple model and stores the model together with the fitted
``FeatureExtractor``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)

try:  # Optional dependency for graph parsing
    import networkx as nx
except Exception:  # pragma: no cover - dependency may be missing at runtime
    nx = None  # type: ignore

from .feature_extractor import (
    FeatureExtractor,
    GraphFeatureExtractor,
    TimeSeriesFeatureExtractor,
)


def load_data(path: str):
    df = pd.read_csv(path)
    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'target' columns")
    return df["text"].astype(str).tolist(), df["target"].values


def train_model(
    model_name: str,
    X_train,
    y_train,
    search_space: Optional[Dict[str, Any]] = None,
    search_type: str = "grid",
    cv: int = 5,
):
    if model_name == "linear":
        base_model = LinearRegression()
    else:
        base_model = RandomForestRegressor(random_state=42)

    if search_space:
        scoring = "neg_mean_squared_error"
        if search_type == "random":
            search = RandomizedSearchCV(
                base_model,
                param_distributions=search_space,
                n_iter=10,
                cv=cv,
                random_state=42,
                scoring=scoring,
            )
        else:
            search = GridSearchCV(
                base_model,
                param_grid=search_space,
                cv=cv,
                scoring=scoring,
            )
        search.fit(X_train, y_train)
        return search.best_estimator_, -search.best_score_, search.best_params_
    scores = cross_val_score(
        base_model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
    )
    base_model.fit(X_train, y_train)
    return base_model, -scores.mean(), {}


def _build_extractor(args) -> Any:
    if args.feature_type == "sentence":
        return FeatureExtractor(
            method="sentence",
            use_pca=args.use_pca,
            n_components=args.pca_components,
            use_feature_selection=args.use_feature_selection,
            var_threshold=args.var_threshold,
        )
    if args.feature_type == "time_series":
        return TimeSeriesFeatureExtractor(
            window_size=args.window_size, apply_fft=args.apply_fft
        )
    if args.feature_type == "graph":
        return GraphFeatureExtractor(dimensions=args.embedding_dim)
    return FeatureExtractor(
        method="tfidf",
        use_pca=args.use_pca,
        n_components=args.pca_components,
        use_feature_selection=args.use_feature_selection,
        var_threshold=args.var_threshold,
    )


def _prepare_inputs(texts, args, extractor) -> Any:
    if args.feature_type == "time_series":
        series_list = [list(map(float, t.split())) for t in texts]
        return extractor.fit_transform(series_list)
    if args.feature_type == "graph":
        if nx is None:  # type: ignore[name-defined]
            raise ImportError("networkx is required for graph features")
        graphs = []
        for t in texts:
            edges = [tuple(e.split("-")) for e in t.split()]
            g = nx.Graph()
            g.add_edges_from(edges)
            graphs.append(g)
        return extractor.fit_transform(graphs)
    return extractor.fit_transform(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on log data")
    parser.add_argument("data", help="Path to CSV file with columns 'text' and 'target'")
    parser.add_argument("--model", choices=["linear", "random_forest"], default="linear")
    parser.add_argument("--version", default="v1", help="Version string for saved artifacts")
    parser.add_argument(
        "--feature-type",
        choices=["tfidf", "sentence", "time_series", "graph"],
        default="tfidf",
    )
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--use-feature-selection", action="store_true")
    parser.add_argument("--var-threshold", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--apply-fft", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--search-space", type=str, default=None, help="JSON dict of hyperparameter search space")
    parser.add_argument(
        "--search-type",
        choices=["grid", "random"],
        default="grid",
        help="Use grid or random search for hyperparameters",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    texts, targets = load_data(args.data)

    extractor = _build_extractor(args)
    X = _prepare_inputs(texts, args, extractor)

    X_train, X_test, y_train, y_test = train_test_split(
        X, targets, test_size=0.2, random_state=42
    )

    search_space = json.loads(args.search_space) if args.search_space else None

    model, cv_mse, best_params = train_model(
        args.model, X_train, y_train, search_space, args.search_type, args.cv
    )

    print(f"Cross-validation MSE: {cv_mse:.4f}")
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    print(f"Test MSE: {test_mse:.4f}")

    artifacts_dir = Path("artifacts") / args.version
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / f"{args.model}_model.joblib")
    extractor.save(str(artifacts_dir / "feature_extractor.joblib"))
    with open(artifacts_dir / "metrics.txt", "w") as f:
        f.write(
            f"Cross-validation MSE: {cv_mse}\nTest MSE: {test_mse}\nBest Params: {best_params}\n"
        )


if __name__ == "__main__":
    main()
