import datetime

import numpy as np
import pytest
from omegaconf import DictConfig


@pytest.fixture
def cfg_common_data_module() -> DictConfig:
    """Test configuration for a CommonDataModule."""
    return DictConfig(
        {
            "base_path": "/mock/base/path",
            "data": {
                "datasets": {"ds1": {"name": "mock", "group_as": "group1"}},
                "split": {
                    "predict": [{"start": None, "end": None}],
                    "test": [{"start": "2020-01-01", "end": "2020-12-31"}],
                    "train": [
                        {"start": None, "end": "2019-12-31"},
                        {"start": "2018-01-01", "end": None},
                    ],
                    "validate": [{"start": "2020-01-01", "end": "2020-03-31"}],
                },
            },
            "variables": {
                "input": {"group1": ["mock_var"]},
                "target": {"group1": ["mock_var"]},
            },
            "window": {
                "batch_size": 2,
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


@pytest.fixture(scope="session")
def dates_as_np(
    dates_as_dt: tuple[datetime.datetime, ...],
) -> tuple[np.datetime64, ...]:
    """Fixture to provide a tuple of numpy datetime64 objects for testing."""
    return tuple(np.datetime64(f"{dt.date()}T12:00:00", "s") for dt in dates_as_dt)


@pytest.fixture(scope="session")
def dates_as_str(dates_as_dt: tuple[datetime.datetime, ...]) -> tuple[str, ...]:
    """Fixture to provide a tuple of date strings for testing."""
    return tuple(dt.strftime(r"%Y-%m-%d") for dt in dates_as_dt)
