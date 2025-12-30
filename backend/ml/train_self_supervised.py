"""Train a self-supervised autoencoder on agent interaction traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TextAutoencoder(nn.Module):
    """Two-layer autoencoder with batch normalization."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, bottleneck: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError("Expected 'sequence' column in dataset for self-supervised training")
    texts = df["sequence"].fillna("").astype(str).tolist()
    meta_cols = [col for col in df.columns if col != "sequence"]
    meta = (
        df[meta_cols].to_dict(orient="records")
        if meta_cols
        else [{} for _ in range(len(texts))]
    )
    return np.array(texts, dtype=object), meta


def build_dataloader(
    texts: np.ndarray,
    batch_size: int,
    *,
    max_features: int,
) -> Tuple[DataLoader, "TfidfVectorizer", np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    features = vectorizer.fit_transform(texts).astype(np.float32)
    dense = features.toarray()
    tensor = torch.from_numpy(dense)
    dataset = TensorDataset(tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
    )
    return loader, vectorizer, dense


def reconstruction_errors(model: TextAutoencoder, data: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(data).to(device)
        recon = model(tensor).cpu().numpy()
    diff = data - recon
    errors = np.mean(diff ** 2, axis=1)
    return errors


def train(
    model: TextAutoencoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> List[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        history.append(epoch_loss)
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            print(f"[epoch={epoch}] loss={epoch_loss:.6f}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised autoencoder training")
    parser.add_argument("data", type=Path, help="Path to CSV with 'sequence' column")
    parser.add_argument("--version", default="selfsup_v1", help="Artifact version")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--bottleneck", type=int, default=64, help="Latent dimension")
    parser.add_argument("--max-features", type=int, default=4096, help="TF-IDF feature limit")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--anomaly-percentile", type=float, default=95.0, help="Percentile for anomaly threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    texts, meta = load_features(args.data)
    loader, vectorizer, dense_features = build_dataloader(
        texts,
        args.batch_size,
        max_features=args.max_features,
    )
    input_dim = dense_features.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextAutoencoder(input_dim, args.hidden_dim, args.bottleneck).to(device)
    history = train(model, loader, device, args.epochs, args.learning_rate)

    dense_features = dense_features.astype(np.float32)
    errors = reconstruction_errors(model, dense_features, device)
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    threshold = float(np.percentile(errors, args.anomaly_percentile))
    anomaly_indices = np.where(errors >= threshold)[0].tolist()

    artifacts_dir = Path("artifacts") / args.version
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "autoencoder.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "bottleneck": args.bottleneck,
        },
        model_path,
    )
    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")

    anomalies = []
    for idx in anomaly_indices:
        metadata = meta[idx]
        metadata = metadata or {}
        metadata["sequence"] = texts[idx]
        metadata["reconstruction_error"] = float(errors[idx])
        anomalies.append(metadata)
    pd.DataFrame(anomalies).to_csv(artifacts_dir / "anomalies.csv", index=False)

    metrics = {
        "Mean Reconstruction": mean_error,
        "Std Reconstruction": std_error,
        "Anomaly Threshold": threshold,
        "Anomaly Count": len(anomaly_indices),
    }

    with open(artifacts_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    with open(artifacts_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"loss": history}, f, indent=2)

    print(
        f"Training complete. Mean reconstruction error {mean_error:.6f}, "
        f"anomalies detected: {len(anomaly_indices)} (threshold {threshold:.6f})."
    )


if __name__ == "__main__":
    main()
