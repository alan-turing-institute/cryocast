import re
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

PREDICT_DIR = Path(str(files("icenet_mp.config"))) / "predict"
PREDICT_CONFIGS = sorted(
    p.stem for p in PREDICT_DIR.glob("*.yaml") if not p.name.endswith(".local.yaml")
)
LEAD_TIME_SUFFIX = re.compile(r"-(\d+)d$")


class TestPredictConfigs:
    """Regression tests for icenet-mp predict configs."""

    @pytest.mark.parametrize("config_name", PREDICT_CONFIGS)
    def test_predict_configs_compose(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config("sample", overrides=[f"predict={config_name}"])

        assert config.predict.n_forecast_steps > 0
        assert config.predict.n_history_steps > 0
        assert config.predict.target.group_name
        assert "ice_conc" in config.predict.target.variables

    @pytest.mark.parametrize("config_name", PREDICT_CONFIGS)
    def test_predict_lead_time_matches_filename(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        match = LEAD_TIME_SUFFIX.search(config_name)
        assert match, f"{config_name!r} does not end in '-<N>d'"
        expected_lead_days = int(match.group(1))

        config = compose_config("sample", overrides=[f"predict={config_name}"])
        assert config.predict.n_forecast_steps == expected_lead_days
