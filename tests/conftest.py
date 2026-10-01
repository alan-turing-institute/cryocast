import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import zarr
from omegaconf import DictConfig, OmegaConf


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: ARG001
    """Restrict PyTorch to a single thread to avoid OpenMP issues under pytest."""
    torch.set_num_threads(1)


def build_zarr(
    zarr_path: Path,
    data_dict: dict[str, Any],
    full_dates: list[datetime.datetime] | None = None,
    missing_dates: list[datetime.datetime] | None = None,
) -> Path:
    """Write a minimal anemoi-compatible zarr store without using Create().

    Bypasses the slow anemoi create pipeline (which has ~40s fixed overhead) by writing
    the zarr arrays directly to disk.
    """
    data_dates: list[datetime.datetime] = data_dict["coords"]["time"]["data"]
    full_dates = full_dates if full_dates is not None else data_dates
    variables = list(data_dict["data_vars"].keys())
    lats: list[float] = data_dict["coords"]["lat"]["data"]
    lons: list[float] = data_dict["coords"]["lon"]["data"]

    T, C, H, W = len(full_dates), len(variables), len(lats), len(lons)  # noqa: N806

    # Build (T, C, 1, H * W) data array; missing timesteps stay zero.
    available_dates = {d.date() for d in data_dates}
    data = np.zeros((T, C, 1, H * W), dtype=np.float32)
    data_idx = 0
    for t, d in enumerate(full_dates):
        if d.date() in available_dates:
            for c, var in enumerate(variables):
                data[t, c, 0, :] = np.array(
                    data_dict["data_vars"][var]["data"][data_idx], dtype=np.float32
                ).ravel()
            data_idx += 1

    # Compute per-variable statistics from available timesteps only.
    flat = data[[d.date() in available_dates for d in full_dates], :, 0, :]
    means = flat.mean(axis=(0, 2)).astype(np.float64)
    stdevs = flat.std(axis=(0, 2)).astype(np.float64)
    minimums = flat.min(axis=(0, 2)).astype(np.float64)
    maximums = flat.max(axis=(0, 2)).astype(np.float64)

    # Create lat/lon grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    latitudes = lat_grid.ravel().astype(np.float64)
    longitudes = lon_grid.ravel().astype(np.float64)

    # Convert dates into the format needed by Anemoi
    full_dates_anemoi = np.array(
        [np.datetime64(d, "s") for d in full_dates],
        dtype="datetime64[s]",
    )
    missing_dates_anemoi = [
        d.isoformat(timespec="seconds") for d in (missing_dates or [])
    ]

    zarr_path.mkdir(parents=True, exist_ok=True)
    z = zarr.open_group(str(zarr_path), mode="w")
    z.create_dataset("data", data=data, chunks=(1, C, 1, H * W))
    z.create_dataset("dates", data=full_dates_anemoi)
    z.create_dataset("latitudes", data=latitudes)
    z.create_dataset("longitudes", data=longitudes)
    z.create_dataset("mean", data=means)
    z.create_dataset("stdev", data=stdevs)
    z.create_dataset("minimum", data=minimums)
    z.create_dataset("maximum", data=maximums)
    z.attrs.update(
        {
            "field_shape": [H, W],
            "frequency": "24h",
            "variables": variables,
            "missing_dates": missing_dates_anemoi,
            "flatten_grid": True,
            "ensemble_dimension": 2,
        }
    )
    return zarr_path


# Multi-year synthetic data for climatology tests. Values depend on channel, month,
# day and year so that monthly means are sensitive to exactly which dates (and which
# years) contribute.
CLIMATOLOGY_START = datetime.datetime(2017, 1, 1)
CLIMATOLOGY_END = datetime.datetime(2020, 1, 5)
CLIMATOLOGY_MISSING = [datetime.datetime(2017, 3, 15)]
CLIMATOLOGY_VARIABLES = ["ice_conc", "ice_thickness"]


def climatology_value(channel: int, dt: datetime.datetime) -> float:
    """Return the synthetic value for a channel and date.

    The value varies by channel (100 * channel), month (10 * (month - 1)), day
    (0.01 * day) and year (year - 2017), so monthly means depend on the dates and
    years that contribute.
    """
    return 100.0 * channel + 10.0 * (dt.month - 1) + 0.01 * dt.day + (dt.year - 2017)


def make_climatology_data_dict(
    start: datetime.datetime,
    end: datetime.datetime,
    missing: list[datetime.datetime],
    variables: list[str] = CLIMATOLOGY_VARIABLES,
) -> dict[str, Any]:
    """Build a build_zarr data dict covering start to end with the given missing dates."""
    dates = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]
    missing_set = {d.date() for d in missing}
    available = [d for d in dates if d.date() not in missing_set]
    data_vars = {
        var: {
            "dims": ("time", "lat", "lon"),
            "attrs": {},
            "data": [
                [[climatology_value(c, d)] * 2 for _ in range(2)] for d in available
            ],
        }
        for c, var in enumerate(variables)
    }
    return {
        "coords": {
            "lat": {"dims": "lat", "attrs": {}, "data": [-89.0, -90.0]},
            "lon": {"dims": "lon", "attrs": {}, "data": [44.0, 45.0]},
            "time": {"dims": ("time",), "attrs": {}, "data": available},
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": len(available)},
        "data_vars": data_vars,
    }


@pytest.fixture
def cfg_common_data_module() -> DictConfig:
    """Test configuration for a CommonDataModule."""
    return DictConfig(
        {
            "base_path": "/mock/base/path",
            "data": {
                "datasets": {"ds1": {"name": "mock", "group_as": "group1"}},
                "split": {
                    "batch_size": 2,
                    "predict": [{"start": None, "end": None}],
                    "test": [{"start": "2020-01-01", "end": "2020-12-31"}],
                    "train": [
                        {"start": None, "end": "2019-12-31"},
                        {"start": "2018-01-01", "end": None},
                    ],
                    "validate": [{"start": "2020-01-01", "end": "2020-03-31"}],
                },
            },
            "predict": {
                "target": {"group_name": "group1"},
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


@pytest.fixture
def cfg_decoder() -> DictConfig:
    """Test configuration for a decoder."""
    return DictConfig({"_target_": "icenet_mp.models.decoders.NaiveLinearDecoder"})


@pytest.fixture
def cfg_encoders() -> DictConfig:
    """Test configuration for an encoder."""
    return DictConfig(
        {
            "latent_space": (64, 64),
            "test-input": {
                "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            },
            "target": {
                "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            },
        }
    )


@pytest.fixture
def cfg_input_space() -> DictConfig:
    """Test configuration for an input space."""
    return DictConfig(
        {
            "channels": 4,
            "name": "test-input",
            "shape": (16, 16),
        }
    )


@pytest.fixture
def cfg_model_service() -> DictConfig:
    """Test configuration for a ModelService."""
    return DictConfig(
        {
            "data": {
                "datasets": {
                    "mock-dataset-1": {
                        "name": "mock_dataset",
                        "group_as": "mock-dataset-group-1",
                    },
                    "mock-dataset-2": {
                        "name": "mock_dataset",
                        "group_as": "mock-dataset-group-2",
                    },
                },
                "split": {
                    "batch_size": 2,
                    "predict": [{"start": None, "end": None}],
                    "test": [{"start": "2019-01-01", "end": "2019-01-31"}],
                    "train": [
                        {"start": "2017-01-01", "end": "2017-12-31"},
                        {"start": "2018-02-01", "end": "2018-12-31"},
                    ],
                    "validate": [{"start": "2018-01-01", "end": "2018-01-31"}],
                },
            },
            "evaluate": {"callbacks": {}},
            "hemisphere": "north",
            "loss": {"_target_": "torch.nn.HuberLoss", "delta": 0.5},
            "model": {
                "_target_": "MockModel",
                "name": "mock-model",
            },
            "predict": {
                "target": {"group_name": "mock-dataset-group-1"},
                "n_forecast_steps": 2,
                "n_history_steps": 3,
            },
            "reporting": {
                "loggers": {},
                "metrics": ["accuracy", "mae"],
            },
            "train": {
                "callbacks": {},
                "optimizer": {},
                "scheduler": {},
                "lr_scheduler": {},
                "trainer": {},
            },
        }
    )


@pytest.fixture
def cfg_optimizer() -> DictConfig:
    """Test configuration for an optimizer."""
    return DictConfig({"_target_": "torch.optim.AdamW", "lr": 5e-4})


@pytest.fixture
def cfg_output_space() -> DictConfig:
    """Test configuration for an output space."""
    return DictConfig(
        {
            "channels": 1,
            "name": "target",
            "shape": (16, 16),
        }
    )


@pytest.fixture
def cfg_processor() -> DictConfig:
    """Test configuration for a processor."""
    return DictConfig({"_target_": "icenet_mp.models.processors.NullProcessor"})


@pytest.fixture
def cfg_scheduler() -> DictConfig:
    """Test configuration for a scheduler."""
    return DictConfig(
        {
            "_target_": "torch.optim.lr_scheduler.LinearLR",
            "start_factor": 0.2,
            "end_factor": 0.8,
        }
    )


@pytest.fixture
def cfg_lr_scheduler() -> DictConfig:
    """Test configuration for a scheduler's Lightning `lr_scheduler_config` wrapper."""
    return DictConfig({"frequency": 1, "interval": "epoch"})


@pytest.fixture
def cfg_metrics() -> list[str]:
    """Test configuration for a model's `metrics` list."""
    return [
        "accuracy",
        "mae",
        "rmse",
        "sieerror",
        "iiee",
        "diiee",
        "centroid_error",
        "fss_neighbourhood_size_1",
        "fss_neighbourhood_size_5",
        "fss_neighbourhood_size_15",
        "ssim",
    ]


@pytest.fixture(scope="session")
def dates_as_dt() -> tuple[datetime.datetime, ...]:
    """Fixture to provide a tuple of datetime objects for testing."""
    return (
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        datetime.datetime(2020, 1, 2, 0, 0, 0),
        datetime.datetime(2020, 1, 3, 0, 0, 0),
        datetime.datetime(2020, 1, 4, 0, 0, 0),
        datetime.datetime(2020, 1, 5, 0, 0, 0),
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


@pytest.fixture(scope="session")
def mock_data(dates_as_dt: tuple[datetime.datetime, ...]) -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset for testing."""
    return {
        "coords": {
            "lat": {
                "dims": ("lat"),
                "attrs": {"units": "degrees_north", "standard_name": "latitude"},
                "data": [-89, -90],
            },
            "lon": {
                "dims": ("lon"),
                "attrs": {"units": "degrees_east", "standard_name": "longitude"},
                "data": [44, 45],
            },
            "time": {
                "dims": ("time",),
                "attrs": {"standard_name": "time"},
                "data": list(dates_as_dt),
            },
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 5},
        "data_vars": {
            "ice_conc": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[0.5, 1.0], [0.4, 0.0]],
                    [[0.4, 0.9], [0.3, 0.1]],
                    [[0.3, 0.8], [0.2, 0.2]],
                    [[0.2, 0.7], [0.1, 0.3]],
                    [[0.1, 0.6], [0.0, 0.4]],
                ],
            },
            "ice_thickness": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[1.5, 1.0], [1.6, 0.0]],
                    [[1.5, 0.9], [1.6, 0.1]],
                    [[1.5, 0.9], [1.6, 0.1]],
                    [[1.5, 0.9], [1.6, 0.2]],
                    [[1.5, 0.8], [1.6, 0.2]],
                ],
            },
            "temperature": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[273.0, 274.0], [275.0, 276.0]],
                    [[273.5, 274.5], [275.5, 276.5]],
                    [[274.0, 275.0], [276.0, 277.0]],
                    [[274.5, 275.5], [276.5, 277.5]],
                    [[275.0, 276.0], [277.0, 278.0]],
                ],
            },
        },
    }


@pytest.fixture(scope="session")
def mock_data_constant_values(
    dates_as_dt: tuple[datetime.datetime, ...],
) -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset with constant data for testing."""
    return {
        "coords": {
            "lat": {
                "dims": "lat",
                "attrs": {"units": "degrees_north", "standard_name": "latitude"},
                "data": [-89, -90],
            },
            "lon": {
                "dims": "lon",
                "attrs": {"units": "degrees_east", "standard_name": "longitude"},
                "data": [44, 45],
            },
            "time": {
                "dims": ("time",),
                "attrs": {"standard_name": "time"},
                "data": list(dates_as_dt)[:2],
            },
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 2},
        "data_vars": {
            "constant": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
            }
        },
    }


@pytest.fixture(scope="session")
def mock_data_missing_dates(
    dates_as_dt: tuple[datetime.datetime, ...],
) -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset with missing dates for testing."""
    return {
        "coords": {
            "lat": {
                "dims": ("lat"),
                "attrs": {"units": "degrees_north", "standard_name": "latitude"},
                "data": [-89, -90],
            },
            "lon": {
                "dims": ("lon"),
                "attrs": {"units": "degrees_east", "standard_name": "longitude"},
                "data": [44, 45],
            },
            "time": {
                "dims": ("time",),
                "attrs": {"standard_name": "time"},
                "data": [dates_as_dt[0], dates_as_dt[2], dates_as_dt[4]],
            },
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 3},
        "data_vars": {
            "ice_conc": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[0.5, 1.0], [0.4, 0.0]],
                    [[0.3, 0.8], [0.2, 0.2]],
                    [[0.1, 0.6], [0.0, 0.4]],
                ],
            }
        },
    }


@pytest.fixture(scope="session")
def mock_data_non_normalized_times(
    mock_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset for testing."""
    output = dict(**mock_data)
    output["coords"]["time"]["data"] = [
        datetime.datetime(2020, 1, 1, 3, 47, 42),
        datetime.datetime(2020, 1, 2, 3, 47, 42),
        datetime.datetime(2020, 1, 3, 3, 47, 42),
        datetime.datetime(2020, 1, 4, 3, 47, 42),
        datetime.datetime(2020, 1, 5, 3, 47, 42),
    ]
    return output


@pytest.fixture(scope="session")
def mock_data_status_flag() -> dict[str, dict[str, Any]]:
    """Fixture to create a mock status_flag dataset with a known land/inactive pattern.

    Cell (0,0): status_flag=1 (land bit set every timestep) -> land.
    Cell (0,1): status_flag=0 -> sea, active.
    Cell (1,0): status_flag=128 (inactive bit set every timestep) -> inactive.
    Cell (1,1): status_flag=0 -> sea, active.
    """
    dates = [
        datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i) for i in range(3)
    ]
    return {
        "coords": {
            "lat": {"dims": "lat", "attrs": {}, "data": [80.0, 85.0]},
            "lon": {"dims": "lon", "attrs": {}, "data": [0.0, 90.0]},
            "time": {"dims": ("time",), "attrs": {}, "data": dates},
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 3},
        "data_vars": {
            "status_flag": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[1, 0], [128, 0]],
                    [[1, 0], [128, 0]],
                    [[1, 0], [128, 0]],
                ],
            }
        },
    }


@pytest.fixture(scope="session")
def mock_data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture to create a temporary directory for mock data files."""
    return tmp_path_factory.mktemp("data", numbered=False)


@pytest.fixture(scope="session")
def mock_dataset(mock_data_path: Path, mock_data: dict[str, dict[str, Any]]) -> Path:
    """Fixture to create a mock file for testing."""
    return build_zarr(mock_data_path / "anemoi" / "mock_dataset.zarr", mock_data)


@pytest.fixture(scope="session")
def mock_dataset_constant_values(
    mock_data_path: Path,
    mock_data_constant_values: dict[str, dict[str, Any]],
) -> Path:
    """Fixture to create a mock file with constant data for testing."""
    return build_zarr(
        mock_data_path / "anemoi" / "mock_dataset_constant_data.zarr",
        mock_data_constant_values,
    )


@pytest.fixture(scope="session")
def mock_dataset_missing_dates(
    dates_as_dt: tuple[datetime.datetime, ...],
    mock_data_path: Path,
    mock_data_missing_dates: dict[str, dict[str, Any]],
) -> Path:
    """Fixture to create a mock file with missing dates for testing."""
    return build_zarr(
        mock_data_path / "anemoi" / "mock_dataset_missing_dates.zarr",
        mock_data_missing_dates,
        full_dates=mock_data_missing_dates["coords"]["time"]["data"],
        missing_dates=[dates_as_dt[1], dates_as_dt[3]],
    )


@pytest.fixture(scope="session")
def climatology_zarr(mock_data_path: Path) -> Path:
    """A multi-year synthetic zarr with values varying by channel, month, day and year.

    Covers 2017-01-01 to 2020-01-05 with one missing date (2017-03-15), so that
    calendar-day means are sensitive to both the dates and the years that contribute.
    None of 2017-2019 are leap years, so 29 February is never present.
    """
    data = make_climatology_data_dict(
        CLIMATOLOGY_START, CLIMATOLOGY_END, CLIMATOLOGY_MISSING
    )
    full_dates = [
        CLIMATOLOGY_START + datetime.timedelta(days=i)
        for i in range((CLIMATOLOGY_END - CLIMATOLOGY_START).days + 1)
    ]
    return build_zarr(
        mock_data_path / "data" / "anemoi" / "sic_south.zarr",
        data,
        full_dates=full_dates,
        missing_dates=CLIMATOLOGY_MISSING,
    )


@pytest.fixture(scope="session")
def mock_dataset_non_normalized_times(
    mock_data_path: Path,
    mock_data_non_normalized_times: dict[str, dict[str, Any]],
) -> Path:
    """Fixture to create a mock file with non-normalized times for testing."""
    return build_zarr(
        mock_data_path / "anemoi" / "mock_dataset_non_normalized_times.zarr",
        mock_data_non_normalized_times,
    )


@pytest.fixture
def cfg_loss() -> DictConfig:
    """Test configuration for a loss function."""
    return OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5})
