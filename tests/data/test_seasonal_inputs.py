"""Integration tests for combining year-round targets with seasonal (gappy) inputs.

Freeze-season-only products such as CS2SMOS sea-ice thickness have no data in summer.
These tests check that a gappy input dataset, written as a real anemoi zarr store with
missing dates, is combined with a year-round SIC dataset so that only windows whose
history steps are fully covered by every input are used.
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
    *zarrs: tuple[str, Path],
    date_ranges: list[dict[str, str | None]] | None = None,
) -> CombinedDataset:
    """Combine (group name, zarr) pairs, targeting SIC, as CommonDataModule does."""
    return CombinedDataset(
        datasets=[
            SingleDataset(
                name=name,
                input_files=[path],
                normalise=False,
                date_ranges=date_ranges or [{"start": None, "end": None}],
            )
            for name, path in zarrs
        ],
        target_group_name="sic-ssmis",
        target_variables=["ice_conc"],
        n_history_steps=2,
        n_forecast_steps=2,
    )


def _start_days(combined: CombinedDataset) -> list[int]:
    return [d.astype("datetime64[D]").item().day for d in combined.dates]


class TestSeasonalInputs:
    def test_seasonal_dataset_excludes_missing_dates(self, sit_zarr: Path) -> None:
        sit = SingleDataset(name="sit-cs2smos", input_files=[sit_zarr], normalise=False)
        assert len(sit.dates) == len(SEASONAL_DATES)

    def test_windows_with_history_in_gap_are_dropped(
        self, sic_zarr: Path, sit_zarr: Path
    ) -> None:
        combined = _combine(("sic-ssmis", sic_zarr), ("sit-cs2smos", sit_zarr))
        # History (d, d+1) must avoid days 5-7; forecasts (d+2, d+3) come from the
        # year-round target so may enter the gap, and must end by day 12.
        assert _start_days(combined) == [1, 2, 3, 8, 9]

    def test_sic_only_inputs_keep_gap_windows(self, sic_zarr: Path) -> None:
        """Without the seasonal input and with no split, gap windows are used."""
        combined = _combine(("sic-ssmis", sic_zarr))
        assert _start_days(combined) == list(range(1, 10))

    def test_seasonal_split_gives_identical_windows_with_and_without_sit(
        self, sic_zarr: Path, sit_zarr: Path
    ) -> None:
        """Listing each season as its own split period aligns SIC-only and SIC+SIT.

        This is how the sample_sit_winters split makes a SIC-only baseline see the
        same windows as a SIC+SIT model.
        """
        seasons: list[dict[str, str | None]] = [
            {"start": "2020-01-01", "end": "2020-01-04"},
            {"start": "2020-01-08", "end": "2020-01-12"},
        ]
        sic_only = _combine(("sic-ssmis", sic_zarr), date_ranges=seasons)
        sic_sit = _combine(
            ("sic-ssmis", sic_zarr), ("sit-cs2smos", sit_zarr), date_ranges=seasons
        )
        # History and forecast steps must both lie inside a season: day 2 is excluded
        # because its forecast (days 4-5) runs past the end of the first season.
        assert _start_days(sic_only) == _start_days(sic_sit) == [1, 8, 9]

    def test_samples_contain_aligned_seasonal_inputs(
        self, sic_zarr: Path, sit_zarr: Path
    ) -> None:
        combined = _combine(("sic-ssmis", sic_zarr), ("sit-cs2smos", sit_zarr))
        # The first window after the gap starts on day 8
        sample = combined[combined.dates.index(np.datetime64("2020-01-08T12:00:00"))]
        np.testing.assert_allclose(sample["sit-cs2smos"][:, 0, 0, 0], [108.0, 109.0])
        np.testing.assert_allclose(sample["sic-ssmis"][:, 0, 0, 0], [8.0, 9.0])
        np.testing.assert_allclose(sample["target"][:, 0, 0, 0], [10.0, 11.0])
