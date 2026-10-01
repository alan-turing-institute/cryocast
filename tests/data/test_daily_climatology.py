from types import SimpleNamespace
from typing import cast

import numpy as np

from icenet_mp.data import SingleDataset, generate_daily_climatology
from icenet_mp.data.calendar_day import CALENDAR_DAY_LABELS


class _FakeDataset:
    """Minimal SingleDataset-compatible object for climatology unit tests."""

    def __init__(self, values: dict[np.datetime64, np.ndarray]) -> None:
        self.dates = sorted(values)
        self.space = SimpleNamespace(chw=next(iter(values.values())).shape)
        self._values = values

    def get_tchw(self, dates: list[np.datetime64]) -> np.ndarray:
        return np.stack([self._values[date] for date in dates], axis=0)


def _dataset() -> SingleDataset:
    return cast(
        "SingleDataset",
        _FakeDataset(
            {
                np.datetime64("2000-01-01"): np.array(
                    [[[1.0, np.nan], [3.0, 4.0]]], dtype=np.float32
                ),
                np.datetime64("2000-02-29"): np.array(
                    [[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32
                ),
                np.datetime64("2001-01-01"): np.array(
                    [[[3.0, 2.0], [5.0, 6.0]]], dtype=np.float32
                ),
            }
        ),
    )


def test_daily_climatology_averages_calendar_days_and_ignores_nonfinite() -> None:
    """Average each calendar day pixel-wise over valid reference observations."""
    dataset = _dataset()
    result = generate_daily_climatology(dataset, dataset.dates)
    jan_1 = CALENDAR_DAY_LABELS.index("01-01")

    np.testing.assert_allclose(
        result.values[jan_1, 0],
        np.array([[2.0, 2.0], [4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.sample_count[jan_1, 0],
        np.array([[2, 1], [2, 2]], dtype=np.int32),
    )


def test_daily_climatology_keeps_february_29_separate() -> None:
    """Represent leap day explicitly and average it over leap years only."""
    dataset = _dataset()
    result = generate_daily_climatology(dataset, dataset.dates)
    leap_day = CALENDAR_DAY_LABELS.index("02-29")

    np.testing.assert_allclose(
        result.values[leap_day, 0],
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(result.sample_count[leap_day, 0], np.ones((2, 2)))


def test_daily_climatology_only_uses_given_dates() -> None:
    """Average only over the dates passed in, not every date in the dataset."""
    dataset = _dataset()
    result = generate_daily_climatology(dataset, [np.datetime64("2001-01-01")])
    jan_1 = CALENDAR_DAY_LABELS.index("01-01")

    np.testing.assert_allclose(
        result.values[jan_1, 0],
        np.array([[3.0, 2.0], [5.0, 6.0]], dtype=np.float32),
    )
    assert np.isnan(result.values[CALENDAR_DAY_LABELS.index("02-29")]).all()


def test_daily_climatology_averages_each_channel() -> None:
    """Keep channels separate when the dataset has more than one target variable."""
    dataset = cast(
        "SingleDataset",
        _FakeDataset(
            {
                np.datetime64("2000-01-01"): np.array(
                    [[[1.0]], [[10.0]]], dtype=np.float32
                ),
                np.datetime64("2001-01-01"): np.array(
                    [[[3.0]], [[30.0]]], dtype=np.float32
                ),
            }
        ),
    )
    result = generate_daily_climatology(dataset, dataset.dates)
    jan_1 = CALENDAR_DAY_LABELS.index("01-01")

    np.testing.assert_allclose(
        result.values[jan_1], np.array([[[2.0]], [[20.0]]], dtype=np.float32)
    )
