from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from icenet_mp.data import CalendarDayClimatology, SingleDataset

JAN_1 = CalendarDayClimatology.day_index(np.datetime64("2000-01-01"))
FEB_28 = CalendarDayClimatology.day_index(np.datetime64("2000-02-28"))
FEB_29 = CalendarDayClimatology.day_index(np.datetime64("2000-02-29"))
MAR_1 = CalendarDayClimatology.day_index(np.datetime64("2000-03-01"))

N_DAYS = len(CalendarDayClimatology.DAY_INDEX)


def _from_dataset(
    monkeypatch: pytest.MonkeyPatch,
    dataset: SingleDataset,
    dates: list[np.datetime64],
    *,
    half_window: int,
) -> CalendarDayClimatology:
    """Run ``from_dataset`` with the given smoothing half-width."""
    monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", half_window)
    return CalendarDayClimatology.from_dataset(dataset, dates)


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


def test_daily_climatology_averages_calendar_days_and_ignores_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Average each calendar day pixel-wise over valid observations."""
    dataset = _dataset()
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=0)

    np.testing.assert_allclose(
        result.mean[JAN_1, 0], np.array([[2.0, 2.0], [4.0, 5.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        result.std[JAN_1, 0], np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    )
    assert result.n_dates[JAN_1] == 2


def test_daily_climatology_std_matches_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the population standard deviation of each calendar day."""
    samples = [0.1, 0.4, 0.35, 0.9]
    dataset = _fake({f"{2001 + i}-01-01": [[[x]]] for i, x in enumerate(samples)})
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=0)

    np.testing.assert_allclose(result.std[JAN_1, 0, 0, 0], np.std(samples), rtol=1e-6)


def test_daily_climatology_keeps_february_29_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Represent leap day explicitly and average it over leap years only."""
    dataset = _dataset()
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=0)

    np.testing.assert_allclose(
        result.mean[FEB_29, 0],
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    )
    assert result.n_dates[FEB_29] == 1


def test_daily_climatology_aligns_calendar_days_across_leap_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group by month/day, so later dates share a slot in leap and non-leap years."""
    dataset = _fake(
        {
            "2000-03-01": [[[1.0]]],
            "2001-03-01": [[[3.0]]],
            "2000-12-31": [[[5.0]]],
            "2001-12-31": [[[7.0]]],
        }
    )
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=0)
    dec_31 = CalendarDayClimatology.day_index(np.datetime64("2001-12-31"))

    # Day-of-year would split each pair (2000 is a leap year), month/day does not
    assert result.n_dates[MAR_1] == 2
    assert result.mean[MAR_1, 0, 0, 0] == 2.0
    assert dec_31 == N_DAYS - 1
    assert result.n_dates[dec_31] == 2
    assert result.mean[dec_31, 0, 0, 0] == 6.0


def test_daily_climatology_marks_all_nan_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave a pixel NaN when it has no finite values on a calendar day."""
    values: dict[str, list] = {
        str(day): [[[4.0, 5.0, 6.0]]]
        for day in np.arange(
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            dtype="datetime64[D]",
        )
    }
    values["2000-01-01"] = [[[np.nan, np.nan, 1.0]]]
    values["2001-01-01"] = [[[np.nan, np.nan, 3.0]]]
    dataset = _fake(values)
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=0)

    assert np.isnan(result.mean[JAN_1, 0, 0, :2]).all()
    assert np.isnan(result.std[JAN_1, 0, 0, :2]).all()
    assert result.mean[JAN_1, 0, 0, 2] == 2.0


def test_daily_climatology_only_uses_given_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Average only over the dates passed in, not every date in the dataset."""
    dataset = _dataset()
    result = _from_dataset(
        monkeypatch, dataset, [np.datetime64("2001-01-01")], half_window=0
    )

    np.testing.assert_allclose(
        result.mean[JAN_1, 0], np.array([[3.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    )
    assert result.n_dates.sum() == 1


def test_daily_climatology_leaves_empty_days_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave calendar days with no dates as NaN, with zero in ``n_dates``."""
    dataset = _dataset()
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=0)
    jan_2 = CalendarDayClimatology.day_index(np.datetime64("2000-01-02"))

    assert np.isnan(result.mean[jan_2]).all()
    assert result.n_dates[jan_2] == 0


def test_daily_climatology_averages_each_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep channels separate when the dataset has more than one target variable."""
    dataset = _fake(
        {"2000-01-01": [[[1.0]], [[10.0]]], "2001-01-01": [[[3.0]], [[30.0]]]}
    )
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=0)

    np.testing.assert_allclose(
        result.mean[JAN_1], np.array([[[2.0]], [[20.0]]], dtype=np.float32)
    )


def test_daily_climatology_smoothing_fills_february_29_from_non_leap_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """29 February is a slot between 28 February and 1 March in every year."""
    dataset = _fake({"2001-02-28": [[[0.0]]], "2001-03-01": [[[3.0]]]})
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=1)

    # 28 February and 1 March are two slots apart, outside each other's 1-day window
    assert result.n_dates[FEB_29] == 0
    assert result.mean[FEB_28, 0, 0, 0] == 0.0
    assert result.mean[FEB_29, 0, 0, 0] == pytest.approx(1.5)
    assert result.std[FEB_29, 0, 0, 0] == pytest.approx(1.5)
    assert result.mean[MAR_1, 0, 0, 0] == 3.0


def test_daily_climatology_smoothing_wraps_around_the_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """31 December and 1 January are neighbouring slots."""
    dataset = _fake({"2001-12-31": [[[0.0]]], "2002-01-01": [[[3.0]]]})
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=1)
    dec_31 = CalendarDayClimatology.day_index(np.datetime64("2001-12-31"))

    assert result.mean[dec_31, 0, 0, 0] == pytest.approx(1.0)
    assert result.mean[JAN_1, 0, 0, 0] == pytest.approx(2.0)


def test_daily_climatology_smoothing_is_nan_only_without_data_in_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A calendar day is NaN only if no date falls within ``half_window`` days of it."""
    dataset = _fake({"2001-06-01": [[[1.0]]]})
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=7)
    june_1 = CalendarDayClimatology.day_index(np.datetime64("2001-06-01"))

    assert not np.isnan(result.mean[june_1 - 7 : june_1 + 8]).any()
    assert np.isnan(result.mean[june_1 - 8]).all()
    assert np.isnan(result.mean[june_1 + 8]).all()


def test_daily_climatology_smoothing_matches_explicit_weighted_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match an explicit weighted average over every date, by calendar-slot distance."""
    half_window = 5
    rng = np.random.default_rng(0)
    days = np.arange(
        np.datetime64("1999-01-01"), np.datetime64("2003-01-01"), dtype="datetime64[D]"
    )
    values = {day: rng.random((1, 1, 2)).astype(np.float32) for day in days}
    dataset = cast("SingleDataset", _FakeDataset(values))
    result = _from_dataset(monkeypatch, dataset, dataset.dates, half_window=half_window)

    stacked = np.stack([values[day] for day in days]).astype(np.float64)
    slots = np.array([CalendarDayClimatology.day_index(day) for day in days])
    for target in range(N_DAYS):
        distance = np.abs(slots - target)
        distance = np.minimum(distance, N_DAYS - distance)
        weights = np.maximum(0.0, 1.0 - distance / (half_window + 1))
        expected = np.tensordot(weights, stacked, axes=1) / weights.sum()
        np.testing.assert_allclose(result.mean[target], expected, rtol=1e-5)


def _linear_february_march(years: list[int]) -> SingleDataset:
    """Return February and March of each year, valued at days since 1 January."""
    values: dict[np.datetime64, np.ndarray] = {}
    for year in years:
        start = np.datetime64(f"{year}-01-01")
        for day in np.arange(
            np.datetime64(f"{year}-02-01"),
            np.datetime64(f"{year}-04-01"),
            dtype="datetime64[D]",
        ):
            offset = float((day - start).astype(int))
            values[day] = np.full((1, 1, 1), offset, dtype=np.float32)
    return cast("SingleDataset", _FakeDataset(values))


def test_daily_climatology_default_window_fills_february_29_from_non_leap_years() -> (
    None
):
    """Non-leap years alone give 29 February a value halfway between its neighbours."""
    dataset = _linear_february_march([2001, 2002, 2003])
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    # The window around 29 February is symmetric over a linear trend: 28 February is
    # day 58 and 1 March is day 59, so 29 February sits at 58.5
    assert result.n_dates[FEB_29] == 0
    assert result.mean[FEB_29, 0, 0, 0] == pytest.approx(58.5)


def test_daily_climatology_default_window_keeps_leap_year_february_29() -> None:
    """In leap years 29 February is a real day between 28 February and 1 March."""
    dataset = _linear_february_march([2000, 2004])
    result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

    # In a leap year 28 February, 29 February and 1 March are days 58, 59 and 60
    for index, expected in [
        (FEB_28, 58.0),
        (FEB_29, 59.0),
        (MAR_1, 60.0),
    ]:
        assert result.mean[index, 0, 0, 0] == pytest.approx(expected)
