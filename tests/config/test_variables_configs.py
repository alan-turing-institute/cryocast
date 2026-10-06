from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

VARIABLES_DIR = Path(str(files("icenet_mp.config"))) / "variables"
VARIABLES_CONFIGS = sorted(
    p.stem for p in VARIABLES_DIR.glob("*.yaml") if not p.name.endswith(".local.yaml")
)


class TestVariablesConfigs:
    """Regression tests for icenet-mp variables configs."""

    @pytest.mark.parametrize("config_name", VARIABLES_CONFIGS)
    def test_variables_configs_compose(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config("sample", overrides=[f"variables={config_name}"])

        assert config.variables.input
        assert config.variables.target
        for group_name, variable_names in config.variables.target.items():
            assert group_name
            assert "ice_conc" in variable_names

    @pytest.mark.parametrize("config_name", VARIABLES_CONFIGS)
    def test_target_groups_are_also_input_groups(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config("sample", overrides=[f"variables={config_name}"])

        for group_name in config.variables.target:
            assert group_name in config.variables.input
