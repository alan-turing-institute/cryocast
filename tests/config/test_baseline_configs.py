import re
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

BASELINE_DIR = Path(str(files("icenet_mp.config"))) / "baseline"
BASELINE_CONFIGS = sorted(
    p.stem for p in BASELINE_DIR.glob("*.yaml") if not p.name.endswith(".local.yaml")
)
NUMERIC_PREFIX = re.compile(r"^\d+_")


def expected_model_name(config_name: str) -> str:
    """Derive a baseline's expected model.name from its `override /model:` convention."""
    return NUMERIC_PREFIX.sub("", config_name).replace("_", "-")


class TestBaselineConfigs:
    """Regression tests for icenet-mp baseline configs."""

    @pytest.mark.parametrize("config_name", BASELINE_CONFIGS)
    def test_baseline_configs_compose(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config(f"baseline/{config_name}")

        assert config.model.name == expected_model_name(config_name)
        assert config.model._target_.startswith("icenet_mp.models.")
        assert "data" in config
        assert "predict" in config
        assert "train" in config
        assert "evaluate" in config

    @pytest.mark.parametrize("config_name", BASELINE_CONFIGS)
    def test_baselines_accept_standard_data_override(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config(
            f"baseline/{config_name}", overrides=["data=sample_north"]
        )

        assert config.data.split.train
        assert config.data.split.test
        assert config.data.split.validate

    def test_dc_unet_dc_lowers_default_learning_rate(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        """Guards the dc_unet_dc lr overrides: the default lr causes ResBlock instability."""
        config = compose_config("baseline/dc_unet_dc")

        assert config.train.optimizer.lr == pytest.approx(5e-4)
        assert config.train.multistage.finetune.optimizer.lr == pytest.approx(1e-4)
