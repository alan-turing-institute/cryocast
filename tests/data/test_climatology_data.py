"""Tests for the climatology (calendar-day mean) data plumbing."""

import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from omegaconf import DictConfig

from icenet_mp.data import CombinedDataset, CommonDataModule, SingleDataset
from icenet_mp.data.calendar_day import (
    FEBRUARY_28_INDEX,
    FEBRUARY_29_INDEX,
    N_CALENDAR_DAYS,
    calendar_day_index,
)
from tests.conftest import (
    CLIMATOLOGY_END,
    CLIMATOLOGY_MISSING,
    CLIMATOLOGY_START,
    CLIMATOLOGY_VARIABLES,
)

# Union-of-training-periods arrangement: all of 2017 and 2018 plus the first half of
# 2019, so the 2019 second half is excluded from the climatology averaging period. None
# of these years are leap years, so 29 February is never present in the period.
TRAIN_PERIODS: list[dict[str, str | None]] = [
    {"start": "2017-01-01", "end": "2018-12-31"},
    {"start": "2019-01-01", "end": "2019-06-30"},
]


def _all_dates() -> list[datetime.datetime]:
    """Return every calendar day covered by the climatology zarr."""
    return [
        CLIMATOLOGY_START + datetime.timedelta(days=i)
        for i in range((CLIMATOLOGY_END - CLIMATOLOGY_START).days + 1)
    ]


def _available_dates() -> list[datetime.datetime]:
    """Return the dates present in the climatology zarr (missing dates excluded)."""
    missing = {d.date() for d in CLIMATOLOGY_MISSING}
    return [d for d in _all_dates() if d.date() not in missing]


def _period_dates(periods: list[dict[str, str | None]]) -> list[datetime.datetime]:
    """Return the available dates falling within any of the given ISO-bounded periods."""
    dates: list[datetime.datetime] = []
    for date in _available_dates():
        day = date.strftime("%Y-%m-%d")
        for period in periods:
            start = period.get("start")
            end = period.get("end")
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            dates.append(date)
            break
    return dates


def _normalised_rows(zarr_path: Path, dates: list[datetime.datetime]) -> np.ndarray:
    """Return [n, C, H, W] float32 rows replicating SingleDataset.normalise.

    The per-channel min/max statistics are read from the zarr (float64), the scale is
    computed in float64 and cast to float32, and the normalisation arithmetic is done
    in float32, exactly as SingleDataset does.
    """
    store = zarr.open(str(zarr_path), mode="r")
    raw = store["data"][:]
    height, width = store.attrs["field_shape"]
    n_channels = raw.shape[1]
    minimum = store["minimum"][:].astype(np.float64)
    maximum = store["maximum"][:].astype(np.float64)
    scale = (1.0 / (maximum - minimum)).astype(np.float32).reshape(n_channels, 1, 1)
    offset = minimum.astype(np.float32).reshape(n_channels, 1, 1)
    full_index = {d.date(): i for i, d in enumerate(_all_dates())}
    rows = []
    for date in dates:
        row = raw[full_index[date.date()]].reshape(n_channels, 1, height, width)[:, 0]
        rows.append((row - offset) * scale)
    return np.stack(rows, axis=0)


def _expected_daily_means(
    zarr_path: Path, period_dates: list[datetime.datetime]
) -> dict[str, np.ndarray]:
    """Return per-variable [366, H, W] float64 tables of calendar-day means.

    Mirrors the 29 February fallback: if the period has no 29 February date, that slot
    is set to the 28 February mean, exactly like ``CommonDataModule.climatology``.
    """
    store = zarr.open(str(zarr_path), mode="r")
    variables = list(store.attrs["variables"])
    rows = _normalised_rows(zarr_path, period_dates)
    by_day: dict[int, list[np.ndarray]] = defaultdict(list)
    for date, row in zip(period_dates, rows, strict=True):
        by_day[calendar_day_index(np.datetime64(date))].append(row)
    expected: dict[str, np.ndarray] = {}
    for variable in variables:
        channel = variables.index(variable)
        table = np.zeros((N_CALENDAR_DAYS, *rows.shape[2:]), dtype=np.float64)
        for day_index, day_rows in by_day.items():
            table[day_index] = (
                np.stack([row[channel] for row in day_rows], axis=0)
                .astype(np.float64)
                .mean(axis=0)
            )
        if FEBRUARY_29_INDEX not in by_day:
            table[FEBRUARY_29_INDEX] = table[FEBRUARY_28_INDEX]
        expected[variable] = table
    return expected


def _cfg(
    base_path: Path,
    train_periods: list[dict[str, str | None]],
    target_variables: list[str] = CLIMATOLOGY_VARIABLES,
) -> DictConfig:
    """Build a CommonDataModule config pointing at the climatology zarr."""
    open_period: list[dict[str, Any]] = [{"start": None, "end": None}]
    return DictConfig(
        {
            "base_path": str(base_path),
            "data": {
                "datasets": {"sic": {"name": "sic_south", "group_as": "sic"}},
                "split": {
                    "batch_size": 2,
                    "predict": open_period,
                    "test": open_period,
                    "train": train_periods,
                    "validate": open_period,
                },
            },
            "predict": {
                "target": {"group_name": "sic", "variables": target_variables},
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


class TestCommonDataModuleClimatology:
    """Tests for the CommonDataModule.climatology calendar-day-mean table."""

    def test_table_shape_and_calendar_day_means(self, climatology_zarr: Path) -> None:
        """Each calendar day holds the mean over its dates in the train-period union."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_cfg(base_path, TRAIN_PERIODS))

        table = dm.climatology
        assert table.shape == (366, 2, 2, 2)
        assert table.dtype == np.float32

        expected = _expected_daily_means(climatology_zarr, _period_dates(TRAIN_PERIODS))
        for channel, variable in enumerate(dm.target_variables):
            np.testing.assert_allclose(
                table[:, channel], expected[variable], rtol=0, atol=1e-6
            )

    def test_uses_union_of_train_periods(self, climatology_zarr: Path) -> None:
        """Dates outside the train-period union (e.g. 15 July 2019) are excluded."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_cfg(base_path, TRAIN_PERIODS))

        table = dm.climatology
        variable = CLIMATOLOGY_VARIABLES[0]
        channel = CLIMATOLOGY_VARIABLES.index(variable)
        july_15 = calendar_day_index(np.datetime64("2000-07-15"))

        expected = _expected_daily_means(climatology_zarr, _period_dates(TRAIN_PERIODS))
        # The correct 15 July mean only includes 2017 and 2018 (the union ends
        # 2019-06-30).
        np.testing.assert_allclose(
            table[july_15, channel], expected[variable][july_15], atol=1e-6
        )

        # A mean that (incorrectly) included 15 July 2019 would differ noticeably,
        # because the synthetic values carry a per-year offset.
        july_15_all_years = [
            d
            for d in _available_dates()
            if d.month == 7 and d.day == 15 and d.year <= 2019
        ]
        wrong = (
            _normalised_rows(climatology_zarr, july_15_all_years)
            .astype(np.float64)
            .mean(axis=0)
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                table[july_15, channel], wrong[channel], atol=1e-6
            )

    def test_missing_dates_excluded_from_means(self, climatology_zarr: Path) -> None:
        """A date missing from the dataset never contributes to its calendar-day mean."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_cfg(base_path, TRAIN_PERIODS))

        table = dm.climatology
        # 15 March includes the missing 2017-03-15; a mean over *all* years' 15 March
        # (using the missing day's zero-filled row) would differ from the table.
        variable = CLIMATOLOGY_VARIABLES[0]
        channel = CLIMATOLOGY_VARIABLES.index(variable)
        march_15 = calendar_day_index(np.datetime64("2000-03-15"))
        expected = _expected_daily_means(climatology_zarr, _period_dates(TRAIN_PERIODS))
        np.testing.assert_allclose(
            table[march_15, channel], expected[variable][march_15], atol=1e-6
        )

        zero_filled = zarr.open(str(climatology_zarr), mode="r")["data"][:]
        march_15_days = [
            d for d in _all_dates() if d.month == 3 and d.day == 15 and d.year <= 2019
        ]
        full_index = {d.date(): i for i, d in enumerate(_all_dates())}
        wrong = zero_filled[[full_index[d.date()] for d in march_15_days], channel]
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                table[march_15, channel], wrong.mean(axis=0), atol=1e-6
            )

    def test_missing_calendar_day_raises(self, climatology_zarr: Path) -> None:
        """A calendar day with no available dates in the period raises ValueError."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _cfg(base_path, [{"start": "2017-01-01", "end": "2017-01-31"}])
        )
        with pytest.raises(ValueError, match="calendar day 02-01"):
            _ = dm.climatology

    def test_missing_29_february_falls_back_to_28_february(
        self, climatology_zarr: Path
    ) -> None:
        """A period spanning only non-leap years copies 28 February onto 29 February."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_cfg(base_path, TRAIN_PERIODS))

        table = dm.climatology
        np.testing.assert_array_equal(
            table[FEBRUARY_29_INDEX], table[FEBRUARY_28_INDEX]
        )

    def test_raises_when_no_dates_in_train_periods(
        self, climatology_zarr: Path
    ) -> None:
        """If no available dates fall in the training periods, climatology raises."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _cfg(base_path, [{"start": "2030-01-01", "end": "2030-12-31"}])
        )
        with pytest.raises(ValueError, match="none of the configured training periods"):
            _ = dm.climatology

    def test_time_component_bounds_match_day_precision(
        self, climatology_zarr: Path
    ) -> None:
        """Bounds carrying a time component behave like day-precision bounds.

        A start bound of ``2017-01-01T12:00:00`` must still include 2017-01-01, so the
        full table is identical to the one built from plain day-precision bounds.
        """
        base_path = climatology_zarr.parents[2]
        timed_periods: list[dict[str, str | None]] = [
            {"start": "2017-01-01T12:00:00", "end": "2018-12-31T12:00:00"},
            {"start": "2019-01-01T00:00:00", "end": "2019-06-30T23:59:59"},
        ]
        dm = CommonDataModule(_cfg(base_path, timed_periods))

        table = dm.climatology
        expected = _expected_daily_means(climatology_zarr, _period_dates(TRAIN_PERIODS))
        for channel, variable in enumerate(dm.target_variables):
            np.testing.assert_allclose(
                table[:, channel], expected[variable], rtol=0, atol=1e-6
            )

    def test_dataloaders_include_climatology(self, climatology_zarr: Path) -> None:
        """Every split's dataloader batches contain a correctly-shaped climatology key."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_cfg(base_path, TRAIN_PERIODS))
        for name in ("train", "val", "test", "predict"):
            loader = getattr(dm, f"{name}_dataloader")()
            batch = next(iter(loader))
            assert "climatology" in batch
            # shape: batch x n_forecast_steps x C_target x H x W
            assert batch["climatology"].shape == (2, 1, 2, 2, 2)

    def test_dataloaders_degrade_gracefully_when_climatology_unavailable(
        self, climatology_zarr: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A train window missing a calendar day must not break other models' loaders.

        ``CommonDataModule.climatology`` itself still raises (see
        ``test_missing_calendar_day_raises``), but building a dataloader is a shared
        code path used by every model, not just the Climatology baseline, so it must
        fall back to omitting the ``climatology`` batch key with a warning instead of
        crashing.
        """
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _cfg(base_path, [{"start": "2017-01-01", "end": "2017-01-31"}])
        )
        with caplog.at_level("WARNING"):
            loader = dm.train_dataloader()
            batch = next(iter(loader))
        assert "climatology" not in batch
        assert "Climatology baseline unavailable" in caplog.text


class TestCombinedDatasetClimatology:
    """Tests for the CombinedDataset climatology key and accessor."""

    @staticmethod
    def _combined(
        climatology_zarr: Path,
        climatology: np.ndarray | None,
    ) -> CombinedDataset:
        ds = SingleDataset(
            name="sic_south",
            input_files=[climatology_zarr],
            date_ranges=[{"start": "2019-12-28", "end": "2020-01-05"}],
        )
        return CombinedDataset(
            datasets=[ds],
            target_group_name="sic_south",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=2,
            climatology=climatology,
        )

    @staticmethod
    def _table() -> np.ndarray:
        """A [366, C, H, W] table whose value encodes the calendar-day index."""
        table = np.zeros((N_CALENDAR_DAYS, 1, 2, 2), dtype=np.float32)
        for index in range(N_CALENDAR_DAYS):
            table[index] = 100.0 * index + 0.5
        return table

    def test_getitem_includes_climatology(self, climatology_zarr: Path) -> None:
        """Batches gain a climatology key with the calendar day of each forecast step."""
        table = self._table()
        combined = self._combined(climatology_zarr, table)
        idx = combined.dates.index(np.datetime64("2019-12-30T12:00:00", "s"))

        batch = combined[idx]
        assert set(batch.keys()) == {"sic_south", "target", "climatology"}
        assert batch["climatology"].shape == (2, 1, 2, 2)
        # Forecast steps are 2019-12-31 and 2020-01-01.
        dec_31 = calendar_day_index(np.datetime64("2019-12-31"))
        jan_1 = calendar_day_index(np.datetime64("2020-01-01"))
        np.testing.assert_array_equal(batch["climatology"][0], table[dec_31])
        np.testing.assert_array_equal(batch["climatology"][1], table[jan_1])

    def test_getitem_without_climatology_is_unchanged(
        self, climatology_zarr: Path
    ) -> None:
        """Without a climatology table the batch keys and values are exactly as before."""
        combined = self._combined(climatology_zarr, None)
        idx = combined.dates.index(np.datetime64("2019-12-30T12:00:00", "s"))

        batch = combined[idx]
        assert set(batch.keys()) == {"sic_south", "target"}
        assert "climatology" not in batch
        np.testing.assert_array_equal(
            batch["target"],
            combined.target.get_tchw(
                [
                    np.datetime64("2019-12-31T12:00:00", "s"),
                    np.datetime64("2020-01-01T12:00:00", "s"),
                ]
            ),
        )

    def test_climatology_for(self, climatology_zarr: Path) -> None:
        """climatology_for returns the calendar-day fields for the forecast steps."""
        table = self._table()
        combined = self._combined(climatology_zarr, table)
        start = np.datetime64("2019-12-31T12:00:00", "s")

        result = combined.climatology_for(start)
        assert result is not None
        # Forecast steps are 2020-01-01 and 2020-01-02.
        jan_1 = calendar_day_index(np.datetime64("2020-01-01"))
        jan_2 = calendar_day_index(np.datetime64("2020-01-02"))
        np.testing.assert_array_equal(result[0], table[jan_1])
        np.testing.assert_array_equal(result[1], table[jan_2])

    def test_climatology_for_none(self, climatology_zarr: Path) -> None:
        """climatology_for returns None when no climatology table was provided."""
        combined = self._combined(climatology_zarr, None)
        assert (
            combined.climatology_for(np.datetime64("2019-12-31T12:00:00", "s")) is None
        )
