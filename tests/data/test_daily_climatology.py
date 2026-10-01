from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from icenet_mp.data import SingleDataset, calendar_day_statistics
from icenet_mp.data.calendar_day import (
    FEBRUARY_28_INDEX,
    FEBRUARY_29_INDEX,
    MARCH_1_INDEX,
    calendar_day_index,
)

JAN_1 = calendar_day_index(np.datetime64("2000-01-01"))


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


def test_calendar_day_statistics_averages_calendar_days_and_ignores_nonfinite() -> None:
    """Average each calendar day pixel-wise over valid observations."""
    dataset = _dataset()
    result = calendar_day_statistics(dataset, dataset.dates)

    np.testing.assert_allclose(
        result.mean[JAN_1, 0], np.array([[2.0, 2.0], [4.0, 5.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        result.std[JAN_1, 0], np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    )
    assert result.n_dates[JAN_1] == 2


def test_calendar_day_statistics_std_matches_numpy() -> None:
    """Return the population standard deviation of each calendar day."""
    samples = [0.1, 0.4, 0.35, 0.9]
    dataset = _fake({f"{2001 + i}-01-01": [[[x]]] for i, x in enumerate(samples)})
    result = calendar_day_statistics(dataset, dataset.dates)

    np.testing.assert_allclose(result.std[JAN_1, 0, 0, 0], np.std(samples), rtol=1e-6)


def test_calendar_day_statistics_keeps_february_29_separate() -> None:
    """Represent leap day explicitly and average it over leap years only."""
    dataset = _dataset()
    result = calendar_day_statistics(dataset, dataset.dates)

    np.testing.assert_allclose(
        result.mean[FEBRUARY_29_INDEX, 0],
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    )
    assert result.n_dates[FEBRUARY_29_INDEX] == 1


def test_calendar_day_statistics_fills_february_29_from_neighbours() -> None:
    """Use the average of 28 February and 1 March when no leap day is present."""
    dataset = _fake(
        {
            "2001-02-28": [[[2.0]]],
            "2002-02-28": [[[4.0]]],
            "2001-03-01": [[[6.0]]],
            "2002-03-01": [[[6.0]]],
        }
    )
    result = calendar_day_statistics(dataset, dataset.dates)

    assert result.mean[FEBRUARY_28_INDEX, 0, 0, 0] == 3.0
    assert result.mean[MARCH_1_INDEX, 0, 0, 0] == 6.0
    assert result.mean[FEBRUARY_29_INDEX, 0, 0, 0] == 4.5
    assert result.std[FEBRUARY_29_INDEX, 0, 0, 0] == 0.5
    assert result.n_dates[FEBRUARY_29_INDEX] == 0


def test_calendar_day_statistics_marks_all_nan_pixels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Leave a pixel NaN, and warn, when it has no finite values on a calendar day."""
    dataset = _fake({"2001-01-01": [[[np.nan, 1.0]]], "2002-01-01": [[[np.nan, 3.0]]]})
    result = calendar_day_statistics(dataset, dataset.dates)

    assert np.isnan(result.mean[JAN_1, 0, 0, 0])
    assert np.isnan(result.std[JAN_1, 0, 0, 0])
    assert result.mean[JAN_1, 0, 0, 1] == 2.0
    assert "1 calendar-day pixels have no finite values" in caplog.text


def test_calendar_day_statistics_only_uses_given_dates() -> None:
    """Average only over the dates passed in, not every date in the dataset."""
    dataset = _dataset()
    result = calendar_day_statistics(dataset, [np.datetime64("2001-01-01")])

    np.testing.assert_allclose(
        result.mean[JAN_1, 0], np.array([[3.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    )
    assert result.n_dates.sum() == 1


def test_calendar_day_statistics_leaves_empty_days_nan() -> None:
    """Leave calendar days with no dates as NaN, with zero in ``n_dates``."""
    dataset = _dataset()
    result = calendar_day_statistics(dataset, dataset.dates)
    jan_2 = calendar_day_index(np.datetime64("2000-01-02"))

    assert np.isnan(result.mean[jan_2]).all()
    assert result.n_dates[jan_2] == 0


def test_calendar_day_statistics_averages_each_channel() -> None:
    """Keep channels separate when the dataset has more than one target variable."""
    dataset = _fake(
        {"2000-01-01": [[[1.0]], [[10.0]]], "2001-01-01": [[[3.0]], [[30.0]]]}
    )
    result = calendar_day_statistics(dataset, dataset.dates)

    np.testing.assert_allclose(
        result.mean[JAN_1], np.array([[[2.0]], [[20.0]]], dtype=np.float32)
    )
