import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from omegaconf import DictConfig

from icenet_mp.data.calendar_day_climatology import CalendarDayClimatology
from icenet_mp.data.common_data_module import CommonDataModule
from icenet_mp.utils import mask_dir
from tests.conftest import (
    CLIMATOLOGY_END,
    CLIMATOLOGY_MISSING,
    CLIMATOLOGY_START,
    CLIMATOLOGY_VARIABLES,
)

FEB_29 = CalendarDayClimatology.day_index(np.datetime64("2000-02-29"))
N_DAYS = len(CalendarDayClimatology.DAY_INDEX)

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


def _zarr_array(zarr_path: Path, name: str) -> np.ndarray:
    """Read a named array from the climatology zarr as a NumPy array."""
    return np.asarray(zarr.open_group(str(zarr_path), mode="r")[name])


def _normalised_rows(zarr_path: Path, dates: list[datetime.datetime]) -> np.ndarray:
    """Return [n, C, H, W] float32 rows replicating SingleDataset.normalise.

    The per-channel min/max statistics are read from the zarr (float64), the scale is
    computed in float64 and cast to float32, and the normalisation arithmetic is done
    in float32, exactly as SingleDataset does.
    """
    raw = _zarr_array(zarr_path, "data")
    height, width = zarr.open_group(str(zarr_path), mode="r").attrs["field_shape"]
    n_channels = raw.shape[1]
    minimum = _zarr_array(zarr_path, "minimum").astype(np.float64)
    maximum = _zarr_array(zarr_path, "maximum").astype(np.float64)
    scale = (1.0 / (maximum - minimum)).astype(np.float32).reshape(n_channels, 1, 1)
    offset = minimum.astype(np.float32).reshape(n_channels, 1, 1)
    full_index = {d.date(): i for i, d in enumerate(_all_dates())}
    rows = []
    for date in dates:
        row = raw[full_index[date.date()]].reshape(n_channels, 1, height, width)[:, 0]
        rows.append((row - offset) * scale)
    return np.stack(rows, axis=0)


def _expected_statistics(
    zarr_path: Path, period_dates: list[datetime.datetime]
) -> tuple[np.ndarray, np.ndarray]:
    """Return [366, C, H, W] float64 smoothed calendar-day mean and std tables.

    Each calendar day is a triangular-weighted average over the period's normalised
    rows within ``CalendarDayClimatology.HALF_WINDOW`` calendar days.
    """
    rows = _normalised_rows(zarr_path, period_dates).astype(np.float64)
    slots = np.array(
        [CalendarDayClimatology.day_index(np.datetime64(d)) for d in period_dates]
    )
    targets = np.arange(N_DAYS).reshape(-1, 1)
    distance = np.abs(slots - targets)
    distance = np.minimum(distance, N_DAYS - distance)
    weights = np.maximum(0.0, 1.0 - distance / (CalendarDayClimatology.HALF_WINDOW + 1))
    totals = weights.sum(axis=1).reshape(-1, 1, 1, 1)
    mean = np.tensordot(weights, rows, axes=1) / totals
    variance = np.tensordot(weights, rows**2, axes=1) / totals - mean**2
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _climatology_cfg(
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


def _climatology(dm: CommonDataModule) -> CalendarDayClimatology:
    """Return the data module's climatology, failing the test if it is unavailable."""
    climatology = dm.climatology
    assert climatology is not None
    return climatology


class TestCommonDataModule:
    def test_mask_directory_derived_from_target_dataset(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """The mask dir is built from the root and the SIC target dataset's name."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.mask_directory == Path("/mock/base/path/data/masks/mock")

    def test_null_preserved_as_none(self, cfg_common_data_module: DictConfig) -> None:
        """Python None (YAML null) must not be stringified to 'None'."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.predict_periods == [{"start": None, "end": None}]

    def test_string_values_unchanged(self, cfg_common_data_module: DictConfig) -> None:
        """Date strings must pass through without modification."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.test_periods == [{"start": "2020-01-01", "end": "2020-12-31"}]
        assert dm.val_periods == [{"start": "2020-01-01", "end": "2020-03-31"}]

    def test_mixed_none_and_string_in_same_period(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """A period with one None bound and one date string normalises both correctly."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.train_periods == [
            {"start": None, "end": "2019-12-31"},
            {"start": "2018-01-01", "end": None},
        ]

    def test_all_four_split_types_normalised(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """None propagates correctly through every split type."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.predict_periods[0]["start"] is None
        assert dm.predict_periods[0]["end"] is None
        assert dm.test_periods[0]["start"] == "2020-01-01"
        assert dm.train_periods[0]["start"] is None
        assert dm.train_periods[1]["end"] is None
        assert dm.val_periods[0]["end"] == "2020-03-31"

    def test_missing_target_group_explains_checkpoint_mismatch(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """A missing evaluation target should explain how to align dataset groups."""
        available_group = next(
            iter(cfg_common_data_module["data"]["datasets"].values())
        )["group_as"]
        cfg_common_data_module["predict"]["target"]["group_name"] = "missing-target"

        with pytest.raises(ValueError, match="missing-target") as exc_info:
            CommonDataModule(cfg_common_data_module)

        message = str(exc_info.value)
        assert str(available_group) in message
        assert "group_as" in message
        assert "predict.target.group_name" in message


class TestTargetMaskDir:
    """B2: choosing the mask when the target group holds multiple datasets."""

    @staticmethod
    def _cfg(base_path: str, datasets: dict) -> DictConfig:
        none_period = [{"start": None, "end": None}]
        return DictConfig(
            {
                "base_path": base_path,
                "data": {
                    "datasets": datasets,
                    "split": {
                        "batch_size": 2,
                        "predict": none_period,
                        "test": none_period,
                        "train": none_period,
                        "validate": none_period,
                    },
                },
                "predict": {
                    "target": {"group_name": "sic"},
                    "n_forecast_steps": 1,
                    "n_history_steps": 1,
                },
            }
        )

    def test_picks_first_dataset_with_an_existing_mask(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With several datasets in the group, use the first that has a mask on disk."""
        cfg = self._cfg(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},  # no mask on disk
                "ds2": {"name": "sic_b", "group_as": "sic"},  # has a mask
            },
        )
        mdir = mask_dir(tmp_path, "sic_b")
        mdir.mkdir(parents=True)
        np.save(mdir / "active_mask.npy", np.ones((4, 4), dtype=np.uint8))

        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            chosen = dm.mask_directory
        # Picked sic_b (the available one), not sic_a (the first listed).
        assert chosen == mask_dir(tmp_path, "sic_b")
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)

    def test_falls_back_to_first_when_no_masks_exist(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No dataset has a mask: fall back to the first (old behaviour) and warn."""
        cfg = self._cfg(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},
                "ds2": {"name": "sic_b", "group_as": "sic"},
            },
        )
        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            chosen = dm.mask_directory
        assert chosen == mask_dir(tmp_path, "sic_a")
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)


class TestCommonDataModuleClimatology:
    """Tests for the CommonDataModule.climatology calendar-day statistics."""

    def test_table_shape_and_calendar_day_means(self, climatology_zarr: Path) -> None:
        """Each calendar day holds the mean over its dates in the train-period union."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))

        table = _climatology(dm).mean
        assert table.shape == (366, 2, 2, 2)
        assert table.dtype == np.float32

        expected, _ = _expected_statistics(
            climatology_zarr, _period_dates(TRAIN_PERIODS)
        )
        np.testing.assert_allclose(table, expected, rtol=0, atol=1e-6)

    def test_uses_union_of_train_periods(self, climatology_zarr: Path) -> None:
        """Dates outside the train-period union (e.g. July 2019) are excluded."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))

        table = _climatology(dm).mean
        july_15 = CalendarDayClimatology.day_index(np.datetime64("2000-07-15"))

        # The correct 15 July mean only includes 2017 and 2018 (the union ends
        # 2019-06-30).
        expected, _ = _expected_statistics(
            climatology_zarr, _period_dates(TRAIN_PERIODS)
        )
        np.testing.assert_allclose(table[july_15], expected[july_15], atol=1e-6)

        # A mean that (incorrectly) included July 2019 would differ noticeably,
        # because the synthetic values carry a per-year offset.
        wrong, _ = _expected_statistics(
            climatology_zarr, [d for d in _available_dates() if d.year <= 2019]
        )
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            np.testing.assert_allclose(table[july_15], wrong[july_15], atol=1e-6)

    def test_missing_dates_excluded_from_means(self, climatology_zarr: Path) -> None:
        """A date missing from the dataset never contributes to its calendar-day mean."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))

        table = _climatology(dm).mean
        # 15 March is next to the missing 2017-03-15; a mean that included the missing
        # day's zero-filled row would differ from the table.
        march_15 = CalendarDayClimatology.day_index(np.datetime64("2000-03-15"))
        period_dates = _period_dates(TRAIN_PERIODS)
        expected, _ = _expected_statistics(climatology_zarr, period_dates)
        np.testing.assert_allclose(table[march_15], expected[march_15], atol=1e-6)

        wrong, _ = _expected_statistics(
            climatology_zarr, sorted(period_dates + CLIMATOLOGY_MISSING)
        )
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            np.testing.assert_allclose(table[march_15], wrong[march_15], atol=1e-6)

    def test_missing_calendar_day_returns_none(
        self, climatology_zarr: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A calendar day with no available dates in the period gives no climatology."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _climatology_cfg(base_path, [{"start": "2017-01-01", "end": "2017-01-31"}])
        )
        with caplog.at_level("WARNING"):
            assert dm.climatology is None
        # January data covers 25 December to 7 February: 45 of 366 calendar days
        assert "321 calendar days have pixels with no finite values" in caplog.text

    def test_29_february_smoothed_without_leap_years(
        self, climatology_zarr: Path
    ) -> None:
        """A period with no leap years still fills 29 February from its neighbours."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))

        climatology = _climatology(dm)
        assert climatology.n_dates[FEB_29] == 0
        expected, _ = _expected_statistics(
            climatology_zarr, _period_dates(TRAIN_PERIODS)
        )
        np.testing.assert_allclose(
            climatology.mean[FEB_29],
            expected[FEB_29],
            rtol=0,
            atol=1e-6,
        )

    def test_std_matches_calendar_day_spread(self, climatology_zarr: Path) -> None:
        """Std is the weighted population std over each calendar day's window."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))

        climatology = _climatology(dm)
        std = climatology.std
        assert std.shape == climatology.mean.shape
        assert std.dtype == np.float32

        _, expected = _expected_statistics(
            climatology_zarr, _period_dates(TRAIN_PERIODS)
        )
        np.testing.assert_allclose(std, expected, rtol=0, atol=1e-6)

    def test_returns_none_when_no_dates_in_train_periods(
        self, climatology_zarr: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If no available dates fall in the training periods, there is no climatology."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _climatology_cfg(base_path, [{"start": "2030-01-01", "end": "2030-12-31"}])
        )
        with caplog.at_level("WARNING"):
            assert dm.climatology is None
        assert "none of the configured training periods" in caplog.text

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
        dm = CommonDataModule(_climatology_cfg(base_path, timed_periods))

        table = _climatology(dm).mean
        expected, _ = _expected_statistics(
            climatology_zarr, _period_dates(TRAIN_PERIODS)
        )
        np.testing.assert_allclose(table, expected, rtol=0, atol=1e-6)

    def test_dataloaders_include_climatology(self, climatology_zarr: Path) -> None:
        """Every split's dataloader batches contain a correctly-shaped climatology key."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))
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

        ``CommonDataModule.climatology`` is None in this case (see
        ``test_missing_calendar_day_returns_none``). Building a dataloader is a shared
        code path used by every model, not just the Climatology baseline, so it must
        omit the ``climatology`` batch key with a warning instead of crashing.
        """
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _climatology_cfg(base_path, [{"start": "2017-01-01", "end": "2017-01-31"}])
        )
        with caplog.at_level("WARNING"):
            loader = dm.train_dataloader()
            batch = next(iter(loader))
        assert "climatology" not in batch
        assert "Cannot build climatology" in caplog.text
