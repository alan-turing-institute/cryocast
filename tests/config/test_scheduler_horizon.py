from importlib.resources import files

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


class TestSchedulerHorizon:
    """Regression coverage for the default cosine scheduler horizon."""

    CONFIG_DIR = str(files("icenet_mp.config"))

    def _compose(self, overrides: list[str] | None = None) -> DictConfig:
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(config_dir=self.CONFIG_DIR, version_base=None):
                return compose(config_name="sample", overrides=overrides or [])
        finally:
            GlobalHydra.instance().clear()

    def test_cosine_scheduler_tracks_configured_training_horizon(self) -> None:
        """T_max follows the configured maximum epoch count."""
        config = self._compose(["train.trainer.max_epochs=17"])

        assert (
            config.train.scheduler._target_
            == "torch.optim.lr_scheduler.CosineAnnealingLR"
        )
        assert config.train.scheduler.T_max == 17

    def test_cosine_scheduler_horizon_changes_with_max_epochs(self) -> None:
        """Changing max_epochs changes T_max rather than leaving it stale."""
        short_run = self._compose(["train.trainer.max_epochs=3"])
        long_run = self._compose(["train.trainer.max_epochs=25"])

        assert short_run.train.scheduler.T_max == 3
        assert long_run.train.scheduler.T_max == 25
