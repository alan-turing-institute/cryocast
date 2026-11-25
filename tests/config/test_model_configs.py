from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

from cryocast.types import RolloutSpace, RolloutStrategy

MODEL_DIR = Path(str(files("cryocast.config"))) / "model"
EPD_CONFIGS = sorted(
    p.stem
    for p in MODEL_DIR.glob("*.yaml")
    if "_target_: cryocast.models.EncodeProcessDecode" in p.read_text()
)


def rollout_strategy(model: DictConfig) -> RolloutStrategy:
    """Build the RolloutStrategy that EncodeProcessDecode builds from `model`."""
    return RolloutStrategy.from_options(
        model.get("rollout_space", RolloutSpace.LATENT),
        model.get("skip_anchor"),
        model.get("relaxation_tau"),
    )


class TestModelConfigs:
    """Tests for the EncodeProcessDecode model configs."""

    @pytest.mark.parametrize("config_name", EPD_CONFIGS, ids=EPD_CONFIGS)
    def test_rollout_stabilisation_is_off_by_default(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        model = compose_config(overrides=[f"model={config_name}"]).model
        strategy = rollout_strategy(model)

        assert "skip_anchor" not in model
        assert "relaxation_tau" not in model
        assert strategy.anchor == RolloutStrategy.default_anchor(strategy.space)
        assert strategy.relaxation_tau is None
