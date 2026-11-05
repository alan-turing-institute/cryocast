from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from omegaconf import DictConfig

from cryocast.data.combined_dataset import MISSING_INPUT_VALUE, CombinedDataset
from cryocast.data.common_data_module import CommonDataModule
from cryocast.data.single_dataset import SingleDataset


def test_single_dataset_fills_unavailable_dates(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    """Fill unavailable dates while preserving available conditioning values."""
    dataset = SingleDataset(name="conditioning", input_files=[mock_dataset])
    dataset.dates = [date for date in dates_as_np if date != dates_as_np[1]]

    values = dataset.get_tchw_with_fill(dates_as_np[:3])

    assert values.shape == (3, 3, 2, 2)
    assert np.all(values[1] == MISSING_INPUT_VALUE)
    assert np.any(values[0] != MISSING_INPUT_VALUE)
    assert np.any(values[2] != MISSING_INPUT_VALUE)


def test_missing_conditioning_data_does_not_remove_target_windows(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    """Keep valid target windows when only conditioning dates are missing."""
    target = SingleDataset(name="target", input_files=[mock_dataset])
    conditioning = SingleDataset(name="conditioning", input_files=[mock_dataset])
    conditioning.dates = list(dates_as_np[2:])

    combined = CombinedDataset(
        datasets=[target, conditioning],
        target_group_name="target",
        target_variables=["ice_conc"],
        allow_missing_inputs=True,
        n_history_steps=2,
        n_forecast_steps=1,
    )

    assert combined.dates == list(dates_as_np[:3])
    batch = combined[0]
    assert np.all(batch["conditioning"] == MISSING_INPUT_VALUE)
    assert np.any(batch["target"] != MISSING_INPUT_VALUE)


def test_missing_target_history_still_removes_window(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    """Remove a window when mandatory target history is unavailable."""
    target = SingleDataset(name="target", input_files=[mock_dataset])
    target.dates = [date for date in dates_as_np if date != dates_as_np[1]]
    conditioning = SingleDataset(name="conditioning", input_files=[mock_dataset])

    combined = CombinedDataset(
        datasets=[target, conditioning],
        target_group_name="target",
        target_variables=["ice_conc"],
        allow_missing_inputs=True,
        n_history_steps=2,
        n_forecast_steps=1,
    )

    assert dates_as_np[0] not in combined.dates
    assert dates_as_np[2] in combined.dates


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_missing_inputs_requires_explicit_boolean(
    cfg_common_data_module: DictConfig, value: str | int | None
) -> None:
    """Malformed config must not silently enable the opt-in loader path."""
    cfg_common_data_module.data.allow_missing_inputs = value
    with pytest.raises(TypeError, match=r"allow_missing_inputs.*boolean"):
        CommonDataModule(cfg_common_data_module)


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_inputs_accepts_boolean_config(
    cfg_common_data_module: DictConfig, *, enabled: bool
) -> None:
    """Boolean flags are preserved, and the omitted setting stays strict."""
    assert CommonDataModule(cfg_common_data_module).allow_missing_inputs is False
    cfg_common_data_module.data.allow_missing_inputs = enabled
    assert CommonDataModule(cfg_common_data_module).allow_missing_inputs is enabled


def test_fill_preserves_values_across_native_missing_dates(
    mock_dataset_missing_dates: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    """Native Anemoi date gaps cannot shift subsequent observations."""
    dataset = SingleDataset("conditioning", [mock_dataset_missing_dates])
    result = dataset.get_tchw_with_fill(dates_as_np)
    store = zarr.open_group(str(mock_dataset_missing_dates), mode="r")
    offset = store["minimum"][:].astype(np.float32).reshape(1, 1, 1)
    scale = (
        (1.0 / (store["maximum"][:] - store["minimum"][:]))
        .astype(np.float32)
        .reshape(1, 1, 1)
    )
    assert result.shape == (5, 1, 2, 2)
    assert result.dtype == np.float32
    for index in (0, 2, 4):
        expected = (store["data"][index].reshape(1, 2, 2) - offset) * scale
        np.testing.assert_array_equal(result[index], expected)
    assert np.all(result[[1, 3]] == MISSING_INPUT_VALUE)


@pytest.mark.parametrize("indices", [(0, 1, 2), (4, 0, 4, 2)])
def test_fill_matches_complete_reads_including_request_order(
    mock_dataset: Path,
    dates_as_np: tuple[np.datetime64, ...],
    indices: tuple[int, ...],
) -> None:
    """Both the fast path and arbitrary repeated-date reads preserve values."""
    dataset = SingleDataset("conditioning", [mock_dataset], variables=["temperature"])
    dates = [dates_as_np[i].astype("datetime64[D]") for i in indices]
    np.testing.assert_array_equal(
        dataset.get_tchw_with_fill(dates), dataset.get_tchw(dates)
    )


def test_all_missing_and_empty_reads_keep_selected_channel_shape(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    """No conditioning observation is required to form a correctly shaped fill."""
    dataset = SingleDataset("conditioning", [mock_dataset], variables=["temperature"])
    outside = [dates_as_np[0] - np.timedelta64(1, "D")]
    result = dataset.get_tchw_with_fill(outside, fill_value=-2.0)
    assert result.shape == (1, 1, 2, 2)
    assert result.dtype == np.float32
    assert np.all(result == -2.0)
    empty = dataset.get_tchw_with_fill([])
    assert empty.shape == (0, 1, 2, 2)
    assert empty.dtype == np.float32


def test_fill_does_not_hide_storage_errors(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    """Only absent dates are filled; broken reads must still fail visibly."""
    dataset = SingleDataset("conditioning", [mock_dataset])
    dates = [dates_as_np[0], dates_as_np[-1] + np.timedelta64(1, "D")]
    with (
        patch.object(dataset, "_chw_raw", side_effect=OSError("broken store")),
        pytest.raises(OSError, match="broken store"),
    ):
        dataset.get_tchw_with_fill(dates)


def test_missing_forecast_targets_are_never_filled(
    mock_dataset_missing_dates: Path,
    mock_dataset: Path,
    dates_as_np: tuple[np.datetime64, ...],
) -> None:
    """Missing target truth rejects every window, even with complete conditioning."""
    target = SingleDataset("sic", [mock_dataset_missing_dates])
    conditioning = SingleDataset("weather", [mock_dataset])
    combined = CombinedDataset(
        [target, conditioning], "sic", ["ice_conc"], allow_missing_inputs=True
    )
    assert dates_as_np[0] in target.dates
    assert dates_as_np[1] not in target.dates
    with pytest.raises(ValueError, match="no valid dates"):
        _ = combined.dates


@pytest.mark.parametrize("enabled", [False, True])
def test_opt_in_preserves_complete_batches_and_climatology(
    mock_dataset: Path, *, enabled: bool
) -> None:
    """Enabling the feature changes neither complete data nor climatology."""
    target = SingleDataset("sic", [mock_dataset])
    weather = SingleDataset("weather", [mock_dataset], variables=["temperature"])
    climatology = np.arange(366 * 4, dtype=np.float32).reshape(366, 1, 2, 2)
    strict = CombinedDataset(
        [target, weather],
        "sic",
        ["ice_conc"],
        n_history_steps=2,
        n_forecast_steps=2,
        climatology=climatology,
    )
    candidate = CombinedDataset(
        [target, weather],
        "sic",
        ["ice_conc"],
        allow_missing_inputs=enabled,
        n_history_steps=2,
        n_forecast_steps=2,
        climatology=climatology,
    )
    assert candidate.dates == strict.dates
    for i in range(len(strict)):
        expected, actual = strict[i], candidate[i]
        assert actual.keys() == expected.keys()
        assert "climatology" in actual
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("enabled", [False, True])
def test_data_module_applies_policy_to_every_loader(
    cfg_common_data_module: DictConfig,
    mock_dataset: Path,
    mock_dataset_missing_dates: Path,
    dates_as_np: tuple[np.datetime64, ...],
    *,
    enabled: bool,
) -> None:
    """Training, validation, test and prediction use the same explicit policy."""
    cfg_common_data_module.data.allow_missing_inputs = enabled
    cfg_common_data_module.window.n_history_steps = 1
    cfg_common_data_module.variables = {
        "input": {"sic": ["ice_conc"], "weather": ["ice_conc"]},
        "target": {"sic": ["ice_conc"]},
    }
    cfg_common_data_module.data.datasets = {
        "sic": {"name": "sic", "group_as": "sic"},
        "weather": {"name": "weather", "group_as": "weather"},
    }
    for stage in ("train", "validate", "test", "predict"):
        cfg_common_data_module.data.split[stage] = [{"start": None, "end": None}]
    module = CommonDataModule(cfg_common_data_module)
    module.dataset_groups.clear()
    module.dataset_groups.update(
        {
            "sic": [mock_dataset],
            "weather": [mock_dataset_missing_dates],
        }
    )
    module.climatology = None
    expected_dates = list(dates_as_np[:4]) if enabled else list(dates_as_np[:4:2])
    for loader in (
        module.train_dataloader(shuffle=False),
        module.val_dataloader(),
        module.test_dataloader(),
        module.predict_dataloader(),
    ):
        assert isinstance(loader.dataset, CombinedDataset)
        assert loader.dataset.allow_missing_inputs is enabled
        assert loader.dataset.dates == expected_dates
        batches = list(loader)
        assert batches
        assert all(np.isfinite(batch["target"].numpy()).all() for batch in batches)
        if enabled:
            assert np.all(batches[0]["weather"][1].numpy() == MISSING_INPUT_VALUE)
