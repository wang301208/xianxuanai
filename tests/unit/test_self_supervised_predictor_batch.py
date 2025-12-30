from __future__ import annotations

from BrainSimulationSystem.learning.self_supervised import SelfSupervisedConfig, SelfSupervisedPredictor


def test_self_supervised_predictor_observe_batch_reports_batch_metrics() -> None:
    cfg = SelfSupervisedConfig(
        max_observation_dim=8,
        latent_dim=4,
        action_embedding_dim=0,
        history_size=8,
        normalize_inputs=False,
        preview_length=4,
    )
    model = SelfSupervisedPredictor(cfg)
    summary = model.observe_batch(
        [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 3.0}],
        metadata=[{"timestamp": 1.0}, {"timestamp": 2.0}],
    )
    assert summary["batch"] == 2.0
    assert summary["reconstruction_loss"] is not None


def test_self_supervised_predictor_learning_rate_clamps() -> None:
    cfg = SelfSupervisedConfig(
        max_observation_dim=8,
        latent_dim=4,
        action_embedding_dim=0,
        normalize_inputs=False,
        lr_min=1e-4,
        lr_max=1e-3,
    )
    model = SelfSupervisedPredictor(cfg)
    model.set_learning_rates(reconstruction_lr=1.0, prediction_lr=1.0)
    rates = model.learning_rates()
    assert rates["learning_rate"] == 1e-3
    assert rates["prediction_learning_rate"] == 1e-3

