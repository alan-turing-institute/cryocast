"""Integration tests for combining year-round targets with seasonal (gappy) inputs.

Freeze-season-only products such as CS2SMOS sea-ice thickness have no data in summer.
These tests write a gappy input dataset as a real anemoi zarr store with missing dates
and combine it with a year-round SIC dataset, both as a required input (windows that
touch the gap are dropped) and with `data/allow_missing=sit-cs2smos` (gap days are
filled and no windows are lost).
"""

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cryocast.data.combined_dataset import CombinedDataset
from cryocast.data.single_dataset import SingleDataset
from tests.conftest import build_zarr

N_DAYS = 12
GAP_DAYS = (5, 6, 7)  # day-of-month with no thickness data ("summer")
FILL_VALUE = -1.0
ALL_DATES = [dt.datetime(2020, 1, day, 12) for day in range(1, N_DAYS + 1)]
SEASONAL_DATES = [d for d in ALL_DATES if d.day not in GAP_DAYS]
GAP_DATES = [d for d in ALL_DATES if d.day in GAP_DAYS]


def _data_dict(
    dates: list[dt.datetime], variable: str, offset: float
) -> dict[str, Any]:
    return {
        "coords": {
            "lat": {"data": [80.0, 85.0]},
            "lon": {"data": [0.0, 90.0]},
            "time": {"data": dates},
        },
        "data_vars": {
            variable: {
                "data": [
                    np.full((2, 2), offset + d.day, dtype=np.float32) for d in dates
                ]
            }
        },
    }


@pytest.fixture(scope="module")
def sic_zarr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Year-round SIC store."""
    return build_zarr(
        tmp_path_factory.mktemp("seasonal") / "sic.zarr",
        _data_dict(ALL_DATES, "ice_conc", offset=0.0),
    )


@pytest.fixture(scope="module")
def sit_zarr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seasonal thickness store with a gap recorded as missing dates."""
    return build_zarr(
        tmp_path_factory.mktemp("seasonal") / "sit.zarr",
        _data_dict(SEASONAL_DATES, "analysis_sea_ice_thickness", offset=100.0),
        full_dates=ALL_DATES,
        missing_dates=GAP_DATES,
    )


def _combine(
    sic_zarr: Path, sit_zarr: Path | None = None, *, fill_value: float | None = None
) -> CombinedDataset:
    """Combine SIC with optional SIT, targeting SIC, as CommonDataModule does."""
    datasets = [SingleDataset(name="sic-osisaf", input_files=[sic_zarr])]
    if sit_zarr is not None:
        datasets.append(
            SingleDataset(
                name="sit-cs2smos", input_files=[sit_zarr], fill_value=fill_value
            )
        )
    return CombinedDataset(
        datasets=datasets,
        target_group_name="sic-osisaf",
        target_variables=["ice_conc"],
        n_history_steps=2,
        n_forecast_steps=2,
    )


def _start_days(combined: CombinedDataset) -> list[int]:
    return [d.astype("datetime64[D]").item().day for d in combined.dates]


class TestSeasonalInputs:
    def test_seasonal_dataset_excludes_missing_dates(self, sit_zarr: Path) -> None:
        sit = SingleDataset(name="sit-cs2smos", input_files=[sit_zarr])
        assert len(sit.dates) == len(SEASONAL_DATES)

    def test_required_seasonal_input_drops_gap_windows(
        self, sic_zarr: Path, sit_zarr: Path
    ) -> None:
        combined = _combine(sic_zarr, sit_zarr)
        # History (d, d+1) must avoid days 5-7; forecasts (d+2, d+3) come from the
        # year-round target so may enter the gap, and must end by day 12.
        assert _start_days(combined) == [1, 2, 3, 8, 9]

    def test_allow_missing_keeps_gap_windows(
        self, sic_zarr: Path, sit_zarr: Path
    ) -> None:
        """With a fill value the seasonal input no longer restricts the windows."""
        combined = _combine(sic_zarr, sit_zarr, fill_value=FILL_VALUE)
        assert _start_days(combined) == _start_days(_combine(sic_zarr))
        assert _start_days(combined) == list(range(1, 10))

    def test_allow_missing_fills_gap_days(self, sic_zarr: Path, sit_zarr: Path) -> None:
        combined = _combine(sic_zarr, sit_zarr, fill_value=FILL_VALUE)
        sit = combined.inputs[1]
        # Window starting on day 4: history days 4 (present) and 5 (gap)
        sample = combined[combined.dates.index(np.datetime64("2020-01-04T12:00:00"))]
        history = sample["sit-cs2smos"]
        np.testing.assert_allclose(
            history[0], sit.get_tchw([np.datetime64("2020-01-04T12:00:00")])[0]
        )
        np.testing.assert_array_equal(history[1], FILL_VALUE)
        assert not np.any(history[0] == FILL_VALUE)
