import re
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, open_dict

from cryocast.types import RolloutSpace, RolloutStrategy, SkipAnchorType

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
        model.get("relaxation"),
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
        assert "relaxation" not in model
        assert strategy.anchor == RolloutStrategy.default_anchor(strategy.space)
        assert strategy.relaxation is None

    @pytest.mark.parametrize("config_name", EPD_CONFIGS, ids=EPD_CONFIGS)
    def test_uncommented_rollout_options_enable_stabilisation(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        text = (MODEL_DIR / f"{config_name}.yaml").read_text()
        _, found, block = text.partition("# Rollout stabilisation.")
        assert found, f"{config_name}.yaml has no commented rollout options"
        uncommented = re.findall(r"^# ( *\w+:.*)$", block, flags=re.MULTILINE)
        model = compose_config(overrides=[f"model={config_name}"]).model
        with open_dict(model):
            model.merge_with(yaml.safe_load("\n".join(uncommented)))
        strategy = rollout_strategy(model)

        assert strategy.anchor == SkipAnchorType.PREVIOUS
        assert strategy.relaxation is not None
        assert strategy.relaxation.tau == pytest.approx(30.0)
        assert strategy.relaxation.apply_in_training
