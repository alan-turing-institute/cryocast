from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from icenet_mp.data import CalendarDayClimatology, SingleDataset

JAN_1 = CalendarDayClimatology.day_index(np.datetime64("2000-01-01"))


class _FakeDataset:
    """Minimal SingleDataset-compatible object for climatology unit tests."""

    def __init__(self, values: dict[np.datetime64, np.ndarray]) -> None:
        self.dates = sorted(values)
        self.space = SimpleNamespace(chw=next(iter(values.values())).shape)
        self._values = values

    def get_tchw(self, dates: list[np.datetime64]) -> np.ndarray:
        return np.stack([self._values[date] for date in dates], axis=0)


def _fake(values: dict[str, list]) -> SingleDataset:
    return cast(
        "SingleDataset",
        _FakeDataset(
            {
                np.datetime64(date): np.array(field, dtype=np.float32)
                for date, field in values.items()
            }
        ),
    )


def _dataset() -> SingleDataset:
    return _fake(
        {
            "2000-01-01": [[[1.0, np.nan], [3.0, 4.0]]],
            "2000-02-29": [[[5.0, 6.0], [7.0, 8.0]]],
            "2001-01-01": [[[3.0, 2.0], [5.0, 6.0]]],
        }
    )


def test_daily_climatology_averages_calendar_days_and_ignores_nonfinite() -> None:
    """Average each calendar day pixel-wise over valid observations."""
    dataset = _dataset()
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    np.testing.assert_allclose(
        result.mean[JAN_1, 0], np.array([[2.0, 2.0], [4.0, 5.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        result.std[JAN_1, 0], np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    )
    assert result.n_dates[JAN_1] == 2


def test_daily_climatology_std_matches_numpy() -> None:
    """Return the population standard deviation of each calendar day."""
    samples = [0.1, 0.4, 0.35, 0.9]
    dataset = _fake({f"{2001 + i}-01-01": [[[x]]] for i, x in enumerate(samples)})
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    np.testing.assert_allclose(result.std[JAN_1, 0, 0, 0], np.std(samples), rtol=1e-6)


def test_daily_climatology_keeps_february_29_separate() -> None:
    """Represent leap day explicitly and average it over leap years only."""
    dataset = _dataset()
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    np.testing.assert_allclose(
        result.mean[CalendarDayClimatology.FEBRUARY_29, 0],
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    )
    assert result.n_dates[CalendarDayClimatology.FEBRUARY_29] == 1


def test_daily_climatology_fills_february_29_from_neighbours() -> None:
    """Use the average of 28 February and 1 March when no leap day is present."""
    dataset = _fake(
        {
            "2001-02-28": [[[2.0]]],
            "2002-02-28": [[[4.0]]],
            "2001-03-01": [[[6.0]]],
            "2002-03-01": [[[6.0]]],
        }
    )
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    assert result.mean[CalendarDayClimatology.FEBRUARY_28, 0, 0, 0] == 3.0
    assert result.mean[CalendarDayClimatology.MARCH_1, 0, 0, 0] == 6.0
    assert result.mean[CalendarDayClimatology.FEBRUARY_29, 0, 0, 0] == 4.5
    assert result.std[CalendarDayClimatology.FEBRUARY_29, 0, 0, 0] == 0.5
    assert result.n_dates[CalendarDayClimatology.FEBRUARY_29] == 0


def test_daily_climatology_marks_all_nan_pixels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Leave a pixel NaN, and warn per day, when it has no finite values on a day."""
    dataset = _fake(
        {
            "2001-01-01": [[[np.nan, np.nan, 1.0]]],
            "2002-01-01": [[[np.nan, np.nan, 3.0]]],
            "2001-01-02": [[[4.0, 5.0, 6.0]]],
        }
    )
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    assert np.isnan(result.mean[JAN_1, 0, 0, :2]).all()
    assert np.isnan(result.std[JAN_1, 0, 0, :2]).all()
    assert result.mean[JAN_1, 0, 0, 2] == 2.0
    # Two NaN pixels on one calendar day count as one day
    assert "There are 1 calendar days with pixels" in caplog.text


def test_daily_climatology_only_uses_given_dates() -> None:
    """Average only over the dates passed in, not every date in the dataset."""
    dataset = _dataset()
    result = CalendarDayClimatology.from_dataset(dataset, [np.datetime64("2001-01-01")])

    np.testing.assert_allclose(
        result.mean[JAN_1, 0], np.array([[3.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    )
    assert result.n_dates.sum() == 1


def test_daily_climatology_leaves_empty_days_nan() -> None:
    """Leave calendar days with no dates as NaN, with zero in ``n_dates``."""
    dataset = _dataset()
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)
    jan_2 = CalendarDayClimatology.day_index(np.datetime64("2000-01-02"))

    assert np.isnan(result.mean[jan_2]).all()
    assert result.n_dates[jan_2] == 0


def test_daily_climatology_averages_each_channel() -> None:
    """Keep channels separate when the dataset has more than one target variable."""
    dataset = _fake(
        {"2000-01-01": [[[1.0]], [[10.0]]], "2001-01-01": [[[3.0]], [[30.0]]]}
    )
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    np.testing.assert_allclose(
        result.mean[JAN_1], np.array([[[2.0]], [[20.0]]], dtype=np.float32)
    )
