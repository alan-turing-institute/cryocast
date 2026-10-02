from collections.abc import Callable

import pytest
from omegaconf import DictConfig


class TestSchedulerHorizon:
    """Regression coverage for the default cosine scheduler horizon."""

    @pytest.mark.parametrize("max_epochs", [3, 17, 25], ids=["short", "mid", "long"])
    def test_cosine_scheduler_tracks_configured_training_horizon(
        self, compose_config: Callable[..., DictConfig], max_epochs: int
    ) -> None:
        """T_max follows the configured maximum epoch count rather than going stale."""
        config = compose_config(overrides=[f"train.trainer.max_epochs={max_epochs}"])

        assert (
            config.train.scheduler._target_
            == "torch.optim.lr_scheduler.CosineAnnealingLR"
        )
        assert config.train.scheduler.T_max == max_epochs
