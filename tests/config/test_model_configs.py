from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

MODEL_DIR = Path(str(files("cryocast.config"))) / "model"
EPD_CONFIGS = sorted(
    p.stem
    for p in MODEL_DIR.glob("*.yaml")
    if "_target_: cryocast.models.EncodeProcessDecode" in p.read_text()
)


class TestModelConfigs:
    """Tests for the EncodeProcessDecode model configs."""

    @pytest.mark.parametrize("config_name", EPD_CONFIGS, ids=EPD_CONFIGS)
    def test_rollout_stabilisation_is_off_by_default(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        model = compose_config(overrides=[f"model={config_name}"]).model

        assert "skip_anchor" not in model
        assert "relaxation_tau" not in model
