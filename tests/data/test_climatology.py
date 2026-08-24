from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from icenet_mp.data import generate_daily_climatology, save_daily_climatology


class _FakeDataset:
    """Minimal SingleDataset-compatible object for climatology unit tests."""

    def __init__(
        self,
        values: dict[np.datetime64, np.ndarray],
        *,
        variable: str = "ice_conc",
    ) -> None:
        self.name = "sic-test"
        self.variable_names = [variable]
        self.dates = sorted(values)
        self.space = SimpleNamespace(shape=next(iter(values.values())).shape)
        self._values = values

    def subset(self, *, variables: list[str]) -> "_FakeDataset":
        if variables != self.variable_names:
            raise ValueError
        return self

    def get_tchw(self, dates: list[np.datetime64]) -> np.ndarray:
        return np.stack([self._values[date][None, ...] for date in dates], axis=0)


def _dataset() -> _FakeDataset:
    return _FakeDataset(
        {
            np.datetime64("2000-01-01"): np.array(
                [[1.0, np.nan], [3.0, 4.0]], dtype=np.float32
            ),
            np.datetime64("2000-02-29"): np.array(
                [[5.0, 6.0], [7.0, 8.0]], dtype=np.float32
            ),
            np.datetime64("2001-01-01"): np.array(
                [[3.0, 2.0], [5.0, 6.0]], dtype=np.float32
            ),
        }
    )


def test_daily_climatology_averages_calendar_days_and_ignores_nonfinite() -> None:
    """Average each calendar day pixel-wise over valid reference observations."""
    result = generate_daily_climatology(
        _dataset(),
        variable="ice_conc",
        reference_end_year=2001,
        years=2,
    )
    jan_1 = result.month_day.tolist().index("01-01")

    np.testing.assert_allclose(
        result.values[jan_1],
        np.array([[2.0, 2.0], [4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.sample_count[jan_1],
        np.array([[2, 1], [2, 2]], dtype=np.int32),
    )


def test_daily_climatology_keeps_february_29_separate() -> None:
    """Represent leap day explicitly and average it over leap years only."""
    result = generate_daily_climatology(
        _dataset(),
        variable="ice_conc",
        reference_end_year=2001,
        years=2,
    )
    leap_day = result.month_day.tolist().index("02-29")

    np.testing.assert_allclose(
        result.values[leap_day],
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(result.sample_count[leap_day], np.ones((2, 2)))


def test_daily_climatology_requires_every_reference_year() -> None:
    """Reject a requested climatology period if an entire reference year is absent."""
    dataset = _FakeDataset(
        {np.datetime64("2000-01-01"): np.ones((2, 2), dtype=np.float32)}
    )

    with pytest.raises(ValueError, match="missing years:.*2001"):
        generate_daily_climatology(
            dataset,
            variable="ice_conc",
            reference_end_year=2001,
            years=2,
        )


def test_daily_climatology_validates_variable_and_window() -> None:
    """Reject unknown variables and non-positive reference windows."""
    dataset = _dataset()

    with pytest.raises(ValueError, match="Variable 'unknown'"):
        generate_daily_climatology(
            dataset,
            variable="unknown",
            reference_end_year=2001,
            years=2,
        )
    with pytest.raises(ValueError, match="years must be greater than 0"):
        generate_daily_climatology(
            dataset,
            variable="ice_conc",
            reference_end_year=2001,
            years=0,
        )


def test_save_daily_climatology_preserves_data_and_provenance(tmp_path: Path) -> None:
    """Persist the baseline values, counts, calendar labels and reference metadata."""
    result = generate_daily_climatology(
        _dataset(),
        variable="ice_conc",
        reference_end_year=2001,
        years=2,
    )

    path = save_daily_climatology(result, tmp_path / "baseline")

    assert path.suffix == ".npz"
    assert path.exists()
    with np.load(path) as saved:
        np.testing.assert_allclose(saved["climatology"], result.values, equal_nan=True)
        np.testing.assert_array_equal(saved["sample_count"], result.sample_count)
        np.testing.assert_array_equal(saved["month_day"], result.month_day)
        assert saved["dataset_name"].item() == "sic-test"
        assert saved["variable"].item() == "ice_conc"
        assert saved["reference_start_year"].item() == 2000
        assert saved["reference_end_year"].item() == 2001
