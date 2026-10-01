from collections.abc import Callable, Iterator
from importlib.resources import files

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

CONFIG_DIR = str(files("icenet_mp.config"))


@pytest.fixture(autouse=True)
def _clear_global_hydra() -> Iterator[None]:
    """Reset Hydra's global singleton state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def compose_config() -> Callable[..., DictConfig]:
    """Return a callable that composes a config from the icenet-mp config directory."""

    def _compose(
        config_name: str = "sample", overrides: list[str] | None = None
    ) -> DictConfig:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name=config_name, overrides=overrides or [])

    return _compose
