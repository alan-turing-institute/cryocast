import re
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

WINDOW_DIR = Path(str(files("icenet_mp.config"))) / "window"
WINDOW_CONFIGS = sorted(
    p.stem for p in WINDOW_DIR.glob("*.yaml") if not p.name.endswith(".local.yaml")
)
FORECAST_HISTORY = re.compile(r"^forecast-(\d+)-history-(\d+)$")


class TestWindowConfigs:
    """Regression tests for icenet-mp window configs."""

    @pytest.mark.parametrize("config_name", WINDOW_CONFIGS)
    def test_window_configs_compose(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config("sample", overrides=[f"window={config_name}"])

        assert config.window.batch_size > 0
        assert config.window.n_forecast_steps > 0
        assert config.window.n_history_steps > 0

    @pytest.mark.parametrize("config_name", WINDOW_CONFIGS)
    def test_window_filename_matches_forecast_and_history_steps(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        match = FORECAST_HISTORY.match(config_name)
        assert match, f"{config_name!r} does not match 'forecast-<N>-history-<M>'"
        expected_forecast_steps = int(match.group(1))
        expected_history_steps = int(match.group(2))

        config = compose_config("sample", overrides=[f"window={config_name}"])
        assert config.window.n_forecast_steps == expected_forecast_steps
        assert config.window.n_history_steps == expected_history_steps
