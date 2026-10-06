"""Calendar-day climatology: indexing and pixel-wise statistics."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np

from cryocast.types import ArrayTCHW

if TYPE_CHECKING:
    from .single_dataset import SingleDataset


@dataclass(frozen=True)
class CalendarDayClimatology:
    """Pixel-wise statistics for each calendar day, in the dataset's own units.

    Dates are grouped by month/day label (``"MM-DD"``) rather than ordinal day-of-year,
    so 29 February keeps its own slot and every other calendar day stays aligned
    between leap and non-leap years.

    Attributes:
        mean: [366, C, H, W] mean field for each calendar day.
        std: [366, C, H, W] population standard deviation for each calendar day.
        n_dates: [366] number of dates falling on each calendar day.

    """

    # Map each month/day label to its index, enumerating the days of 2000 (a leap year)
    DAY_INDEX: ClassVar[dict[str, int]] = {
        str(day)[5:]: index
        for index, day in enumerate(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2001-01-01"),
                dtype="datetime64[D]",
            )
        )
    }
    # Triangular window half-width. Smooths day-to-day variability keeping seasonality.
    HALF_WINDOW: ClassVar[int] = 7

    mean: ArrayTCHW
    std: ArrayTCHW
    n_dates: np.ndarray

    @classmethod
    def day_index(cls, day: np.datetime64) -> int:
        """Return the 0-365 calendar-day index (month/day label) for a date."""
        label = np.datetime_as_string(day.astype("datetime64[D]"), unit="D")[5:]
        return cls.DAY_INDEX[label]

    @classmethod
    def from_dataset(
        cls,
        dataset: "SingleDataset",
        dates: Sequence[np.datetime64],
    ) -> Self:
        """Compute smoothed pixel-wise mean and standard deviation for each calendar day.

        The data is read once, reducing it to a count, sum and sum of squares of the
        finite values on each calendar day. These are then smoothed around the
        366-day calendar with triangular weights, falling linearly from 1 at the centre
        to ``1 / (HALF_WINDOW + 1)`` at each end, so each calendar day draws on every
        date within ``HALF_WINDOW`` calendar days of it, wrapping from 31 December to 1
        January. Every year is treated as having a 29 February slot: that slot draws
        on 28 February and 1 March from every year, and in non-leap years those two
        days are treated as two days apart rather than one.

        Sums are accumulated in float64. Non-finite values are excluded pixel by
        pixel, so a pixel is NaN only if it has no finite values within the window;
        the caller decides whether that is an error.

        Args:
            dataset: The dataset to read fields from.
            dates: The dates to average over.

        """
        by_day: dict[int, list[np.datetime64]] = defaultdict(list)
        for day in dates:
            by_day[cls.day_index(day)].append(day)

        # Reduce: count, sum and sum of squares of the finite values on each day
        counts, totals, squares = (
            np.zeros((len(cls.DAY_INDEX), *dataset.space.chw), dtype=np.float64)
            for _ in range(3)
        )
        n_dates = np.zeros(len(cls.DAY_INDEX), dtype=np.int64)
        for index, day_dates in by_day.items():
            fields = dataset.get_tchw(day_dates).astype(np.float64)
            valid = np.isfinite(fields)
            values = np.where(valid, fields, 0.0)
            counts[index] = valid.sum(axis=0)
            totals[index] = values.sum(axis=0)
            squares[index] = (values**2).sum(axis=0)
            n_dates[index] = len(day_dates)

        # Smooth: weighted sums over neighbouring calendar days, wrapping at year end
        def smooth(moment: np.ndarray) -> np.ndarray:
            smoothed = np.zeros_like(moment)
            for offset in range(-cls.HALF_WINDOW, cls.HALF_WINDOW + 1):
                weight = 1.0 - abs(offset) / (cls.HALF_WINDOW + 1)
                smoothed += weight * np.roll(moment, -offset, axis=0)
            return smoothed

        weights = smooth(counts)
        # Pixels with no finite values in the window give 0/0, i.e. NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = smooth(totals) / weights
            variance = smooth(squares) / weights - mean**2

        return cls(
            mean=mean.astype(np.float32),
            std=np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
            n_dates=n_dates,
        )
