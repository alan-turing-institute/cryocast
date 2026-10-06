from collections.abc import Mapping
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import ArrayLike

from cryocast.data import CalendarDayClimatology, SingleDataset
from cryocast.types import DataSpace

JAN_1 = CalendarDayClimatology.day_index(np.datetime64("2000-01-01"))
FEB_28 = CalendarDayClimatology.day_index(np.datetime64("2000-02-28"))
FEB_29 = CalendarDayClimatology.day_index(np.datetime64("2000-02-29"))
MAR_1 = CalendarDayClimatology.day_index(np.datetime64("2000-03-01"))
DEC_31 = CalendarDayClimatology.day_index(np.datetime64("2001-12-31"))

N_DAYS = len(CalendarDayClimatology.DAY_INDEX)


def _mock_dataset(values: Mapping[str, ArrayLike]) -> MagicMock:
    """Return a SingleDataset stand-in serving a [C, H, W] field for each ISO date."""
    fields = {
        np.datetime64(date): np.asarray(field, dtype=np.float32)
        for date, field in values.items()
    }
    channels, height, width = next(iter(fields.values())).shape
    dataset = MagicMock(spec=SingleDataset)
    dataset.dates = sorted(fields)
    dataset.space = DataSpace(channels=channels, name="mock", shape=(height, width))
    dataset.get_tchw.side_effect = lambda dates: np.stack(
        [fields[date] for date in dates], axis=0
    )
    return dataset


class TestCalendarDayClimatology:
    # Two years of 1 January (with a NaN pixel) and one leap-year 29 February
    FIELDS: ClassVar[dict[str, list]] = {
        "2000-01-01": [[[1.0, np.nan], [3.0, 4.0]]],
        "2000-02-29": [[[5.0, 6.0], [7.0, 8.0]]],
        "2001-01-01": [[[3.0, 2.0], [5.0, 6.0]]],
    }

    def test_day_index_has_one_slot_per_leap_year_day(self) -> None:
        """Index every month/day label of a leap year, in calendar order."""
        assert N_DAYS == 366
        assert list(CalendarDayClimatology.DAY_INDEX.values()) == list(range(N_DAYS))

    @pytest.mark.parametrize(
        ("day", "expected"),
        [
            (np.datetime64("2001-01-01"), 0),
            (np.datetime64("2000-02-28"), 58),
            (np.datetime64("2000-02-29"), 59),
            (np.datetime64("2000-03-01"), 60),
            (np.datetime64("2001-03-01"), 60),
            (np.datetime64("2001-12-31"), 365),
            (np.datetime64("2020-01-01T12:00:00", "s"), 0),
            (np.datetime64("2019-12-31T23:59:59", "s"), 365),
        ],
        ids=[
            "jan-01",
            "feb-28",
            "feb-29-leap",
            "mar-01-leap",
            "mar-01-non-leap",
            "dec-31-non-leap",
            "jan-01-seconds",
            "dec-31-seconds",
        ],
    )
    def test_day_index(self, day: np.datetime64, expected: int) -> None:
        """Map dates to month/day slots, ignoring year, leap status and time of day."""
        assert CalendarDayClimatology.day_index(day) == expected

    def test_averages_calendar_days_and_ignores_nonfinite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Average each calendar day pixel-wise over valid observations."""
        dataset = _mock_dataset(self.FIELDS)
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        np.testing.assert_allclose(result.mean[JAN_1, 0], [[2.0, 2.0], [4.0, 5.0]])
        np.testing.assert_allclose(result.std[JAN_1, 0], [[1.0, 0.0], [1.0, 1.0]])
        assert result.n_dates[JAN_1] == 2

    def test_std_matches_numpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return the population standard deviation of each calendar day."""
        samples = [0.1, 0.4, 0.35, 0.9]
        dataset = _mock_dataset(
            {f"{2001 + i}-01-01": [[[x]]] for i, x in enumerate(samples)}
        )
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        np.testing.assert_allclose(
            result.std[JAN_1, 0, 0, 0], np.std(samples), rtol=1e-6
        )

    def test_keeps_february_29_separate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Represent leap day explicitly and average it over leap years only."""
        dataset = _mock_dataset(self.FIELDS)
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        np.testing.assert_allclose(
            result.mean[FEB_29, 0],
            [[5.0, 6.0], [7.0, 8.0]],
        )
        assert result.n_dates[FEB_29] == 1

    def test_aligns_calendar_days_across_leap_years(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Group by month/day, so later dates share a slot in leap and non-leap years."""
        dataset = _mock_dataset(
            {
                "2000-03-01": [[[1.0]]],
                "2001-03-01": [[[3.0]]],
                "2000-12-31": [[[5.0]]],
                "2001-12-31": [[[7.0]]],
            }
        )
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        # Day-of-year would split each pair (2000 is a leap year), month/day does not
        assert result.n_dates[MAR_1] == 2
        assert result.mean[MAR_1, 0, 0, 0] == 2.0
        assert result.n_dates[DEC_31] == 2
        assert result.mean[DEC_31, 0, 0, 0] == 6.0

    def test_marks_all_nan_pixels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Leave a pixel NaN when it has no finite values on a calendar day."""
        dataset = _mock_dataset(
            {
                "2000-01-01": [[[np.nan, np.nan, 1.0]]],
                "2001-01-01": [[[np.nan, 4.0, 3.0]]],
            }
        )
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        np.testing.assert_allclose(result.mean[JAN_1, 0, 0], [np.nan, 4.0, 2.0])
        np.testing.assert_allclose(result.std[JAN_1, 0, 0], [np.nan, 0.0, 1.0])

    def test_only_uses_given_dates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Average only over the dates passed in, not every date in the dataset."""
        dataset = _mock_dataset(self.FIELDS)
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(
            dataset, [np.datetime64("2001-01-01")]
        )

        np.testing.assert_allclose(result.mean[JAN_1, 0], [[3.0, 2.0], [5.0, 6.0]])
        assert result.n_dates.sum() == 1

    def test_leaves_empty_days_nan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Leave calendar days with no dates as NaN, with zero in ``n_dates``."""
        dataset = _mock_dataset(self.FIELDS)
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)
        jan_2 = CalendarDayClimatology.day_index(np.datetime64("2000-01-02"))

        assert np.isnan(result.mean[jan_2]).all()
        assert result.n_dates[jan_2] == 0

    def test_averages_each_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep channels separate when the dataset has more than one target variable."""
        dataset = _mock_dataset(
            {"2000-01-01": [[[1.0]], [[10.0]]], "2001-01-01": [[[3.0]], [[30.0]]]}
        )
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 0)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        np.testing.assert_allclose(result.mean[JAN_1], [[[2.0]], [[20.0]]])

    def test_smoothing_fills_february_29_from_non_leap_years(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """29 February is a slot between 28 February and 1 March in every year."""
        dataset = _mock_dataset({"2001-02-28": [[[0.0]]], "2001-03-01": [[[3.0]]]})
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 1)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        # 28 February and 1 March are two slots apart, outside each other's 1-day window
        assert result.n_dates[FEB_29] == 0
        assert result.mean[FEB_28, 0, 0, 0] == 0.0
        assert result.mean[FEB_29, 0, 0, 0] == pytest.approx(1.5)
        assert result.std[FEB_29, 0, 0, 0] == pytest.approx(1.5)
        assert result.mean[MAR_1, 0, 0, 0] == 3.0

    def test_smoothing_wraps_around_the_year(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """31 December and 1 January are neighbouring slots."""
        dataset = _mock_dataset({"2001-12-31": [[[0.0]]], "2002-01-01": [[[3.0]]]})
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 1)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        assert result.mean[DEC_31, 0, 0, 0] == pytest.approx(1.0)
        assert result.mean[JAN_1, 0, 0, 0] == pytest.approx(2.0)

    def test_smoothing_is_nan_only_without_data_in_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A calendar day is NaN only if no date falls within ``half_window`` days."""
        dataset = _mock_dataset({"2001-06-01": [[[1.0]]]})
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", 7)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)
        june_1 = CalendarDayClimatology.day_index(np.datetime64("2001-06-01"))

        assert not np.isnan(result.mean[june_1 - 7 : june_1 + 8]).any()
        assert np.isnan(result.mean[june_1 - 8]).all()
        assert np.isnan(result.mean[june_1 + 8]).all()

    def test_smoothing_matches_explicit_weighted_average(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Match an explicit weighted average over every date, by calendar distance."""
        half_window = 5
        rng = np.random.default_rng(0)
        days = np.arange(
            np.datetime64("1999-01-01"),
            np.datetime64("2003-01-01"),
            dtype="datetime64[D]",
        )
        values = {str(day): rng.random((1, 1, 2), dtype=np.float32) for day in days}
        dataset = _mock_dataset(values)
        monkeypatch.setattr(CalendarDayClimatology, "HALF_WINDOW", half_window)
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        stacked = np.stack(list(values.values())).astype(np.float64)
        slots = np.array([CalendarDayClimatology.day_index(day) for day in days])
        for target in range(N_DAYS):
            distance = np.abs(slots - target)
            distance = np.minimum(distance, N_DAYS - distance)
            weights = np.maximum(0.0, 1.0 - distance / (half_window + 1))
            expected = np.tensordot(weights, stacked, axes=1) / weights.sum()
            np.testing.assert_allclose(result.mean[target], expected, rtol=1e-5)

    @pytest.mark.parametrize(
        ("years", "index", "expected"),
        [
            # 28 February is day 58 and 1 March is day 59, so 29 February sits between
            ([2001, 2002, 2003], FEB_29, 58.5),
            # In leap years 28 February, 29 February and 1 March are days 58, 59, 60
            ([2000, 2004], FEB_28, 58.0),
            ([2000, 2004], FEB_29, 59.0),
            ([2000, 2004], MAR_1, 60.0),
        ],
        ids=["non-leap-feb-29", "leap-feb-28", "leap-feb-29", "leap-mar-01"],
    )
    def test_default_window_around_february_29(
        self, years: list[int], index: int, expected: float
    ) -> None:
        """The default window keeps 29 February between its neighbours.

        Over a linear trend the symmetric window leaves each day at its own value, and
        fills 29 February halfway between 28 February and 1 March in non-leap years.
        """
        # February and March of each year, valued at days since 1 January
        dataset = _mock_dataset(
            {
                str(day): [[[(day - day.astype("datetime64[Y]")).astype(float)]]]
                for year in years
                for day in np.arange(
                    np.datetime64(f"{year}-02-01"),
                    np.datetime64(f"{year}-04-01"),
                    dtype="datetime64[D]",
                )
            }
        )
        result = CalendarDayClimatology.from_dataset(dataset, dataset.dates)

        assert result.mean[index, 0, 0, 0] == pytest.approx(expected)
