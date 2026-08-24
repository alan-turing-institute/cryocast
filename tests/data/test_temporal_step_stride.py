from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from icenet_mp.data.combined_dataset import CombinedDataset
from icenet_mp.data.common_data_module import CommonDataModule
from icenet_mp.data.single_dataset import SingleDataset


def test_weekly_stride_spaces_history_and_forecast_dates(
    mock_dataset: Path,
    dates_as_np: tuple[np.datetime64, ...],
) -> None:
    """Space model timesteps by the configured multiple of native data frequency."""
    dataset = SingleDataset(name="dataset", input_files=[mock_dataset])
    # Extend the fixture's daily timeline so a weekly-spaced window is available.
    start = dates_as_np[0]
    dataset.dates = [start + np.timedelta64(day, "D") for day in range(22)]

    combined = CombinedDataset(
        datasets=[dataset],
        target_group_name="dataset",
        target_variables=["ice_conc"],
        n_history_steps=2,
        n_forecast_steps=2,
        step_stride=7,
    )

    assert combined.get_history_steps(start) == [
        start,
        start + np.timedelta64(7, "D"),
    ]
    assert combined.get_forecast_steps(start) == [
        start + np.timedelta64(14, "D"),
        start + np.timedelta64(21, "D"),
    ]


def test_strided_getitem_reads_exact_requested_dates(
    mock_dataset: Path,
    dates_as_np: tuple[np.datetime64, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use date-based reads rather than contiguous slices for strided windows."""
    dataset = SingleDataset(name="dataset", input_files=[mock_dataset])
    start = dates_as_np[0]
    dataset.dates = [start + np.timedelta64(day, "D") for day in range(5)]
    combined = CombinedDataset(
        datasets=[dataset],
        target_group_name="dataset",
        target_variables=["ice_conc"],
        n_history_steps=2,
        n_forecast_steps=1,
        step_stride=2,
    )

    requested: list[list[np.datetime64]] = []
    original_get_tchw = SingleDataset.get_tchw

    def record_dates(self: SingleDataset, dates: list[np.datetime64]) -> np.ndarray:
        requested.append(list(dates))
        return original_get_tchw(self, dates)

    monkeypatch.setattr(SingleDataset, "get_tchw", record_dates)
    _ = combined[0]

    assert requested[0] == [start, start + np.timedelta64(2, "D")]
    assert requested[1] == [start + np.timedelta64(4, "D")]


def test_temporal_stride_rejects_nonpositive_values(mock_dataset: Path) -> None:
    """Reject zero or negative temporal spacing."""
    dataset = SingleDataset(name="dataset", input_files=[mock_dataset])

    with pytest.raises(ValueError, match="step_stride must be at least 1"):
        CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset",
            target_variables=["ice_conc"],
            step_stride=0,
        )


def test_common_data_module_reads_predict_step_stride(
    cfg_common_data_module: DictConfig,
) -> None:
    """Expose temporal spacing through the prediction configuration."""
    cfg_common_data_module["predict"]["step_stride"] = 7

    data_module = CommonDataModule(cfg_common_data_module)

    assert data_module.step_stride == 7


def test_common_data_module_defaults_to_daily_stride(
    cfg_common_data_module: DictConfig,
) -> None:
    """Preserve current daily behaviour when no stride is configured."""
    cfg_common_data_module["predict"].pop("step_stride", None)

    data_module = CommonDataModule(cfg_common_data_module)

    assert data_module.step_stride == 1
