from collections.abc import Callable

import pytest
from omegaconf import DictConfig

DEFAULT_CALLBACK_TARGETS = {
    "best_checkpoint": "lightning.pytorch.callbacks.ModelCheckpoint",
    "early_stopping": "lightning.pytorch.callbacks.EarlyStopping",
    "ema_weight_averaging": "icenet_mp.callbacks.EMAWeightAveragingCallback",
    "learning_rate": "lightning.pytorch.callbacks.LearningRateMonitor",
    "media_logging": "icenet_mp.callbacks.MediaLoggingCallback",
    "metric_summary": "icenet_mp.callbacks.MetricSummaryCallback",
}

PERSISTENCE_CALLBACK_TARGETS = {
    "learning_rate": "lightning.pytorch.callbacks.LearningRateMonitor",
    "media_logging": "icenet_mp.callbacks.MediaLoggingCallback",
    "metric_summary": "icenet_mp.callbacks.MetricSummaryCallback",
    "unconditional_checkpoint": "icenet_mp.callbacks.UnconditionalCheckpoint",
}


class TestTrainCallbacks:
    """Regression tests for icenet-mp's train.callbacks composition."""

    def test_default_train_callbacks(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        config = compose_config("sample")

        assert set(config.train.callbacks.keys()) == set(DEFAULT_CALLBACK_TARGETS)
        for name, target in DEFAULT_CALLBACK_TARGETS.items():
            assert config.train.callbacks[name]._target_ == target

    def test_persistence_overrides_callbacks(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        """00_persistence swaps in unconditional_checkpoint over best/early-stopping."""
        config = compose_config("baseline/00_persistence")

        assert set(config.train.callbacks.keys()) == set(PERSISTENCE_CALLBACK_TARGETS)
        for name, target in PERSISTENCE_CALLBACK_TARGETS.items():
            assert config.train.callbacks[name]._target_ == target


class TestTrainMultistage:
    """Regression tests for icenet-mp's train.multistage composition."""

    def test_multistage_default_present(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        config = compose_config("sample")

        assert config.train.multistage.finetune.optimizer.lr == pytest.approx(5e-4)
