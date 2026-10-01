"""Calendar-day climatology: indexing and pixel-wise statistics."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np

from icenet_mp.types import ArrayTCHW

if TYPE_CHECKING:
    from .single_dataset import SingleDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalendarDayClimatology:
    """Pixel-wise statistics for each calendar day, in the dataset's own units.

    Dates are grouped by month/day label (``"MM-DD"``) rather than ordinal day-of-year,
    so 29 February keeps its own slot and every other calendar day stays aligned
    between leap and non-leap years.

    Attributes:
        mean: [366, C, H, W] mean field for each calendar day.
        std: [366, C, H, W] population standard deviation for each calendar day.
        n_dates: [366] number of dates contributing to each calendar day.

    """

    # Map each month/day label to its index, enumerating the days of 2000 (a leap year)
    _INDEX: ClassVar[dict[str, int]] = {
        str(day)[5:]: index
        for index, day in enumerate(
            np.arange(
                np.datetime64("2000-01-01"),
                np.datetime64("2001-01-01"),
                dtype="datetime64[D]",
            )
        )
    }
    LABELS: ClassVar[tuple[str, ...]] = tuple(_INDEX)
    N_DAYS: ClassVar[int] = len(_INDEX)
    FEBRUARY_28: ClassVar[int] = _INDEX["02-28"]
    FEBRUARY_29: ClassVar[int] = _INDEX["02-29"]
    MARCH_1: ClassVar[int] = _INDEX["03-01"]

    mean: ArrayTCHW
    std: ArrayTCHW
    n_dates: np.ndarray

    @classmethod
    def day_index(cls, day: np.datetime64) -> int:
        """Return the 0-365 calendar-day index (month/day label) for a date."""
        label = np.datetime_as_string(day.astype("datetime64[D]"), unit="D")[5:]
        return cls._INDEX[label]

    @classmethod
    def from_dataset(
        cls, dataset: "SingleDataset", dates: Sequence[np.datetime64]
    ) -> Self:
        """Compute pixel-wise mean and standard deviation for each calendar day.

        Sums are accumulated in float64. Non-finite values are excluded pixel by
        pixel, and a pixel with no finite values on a calendar day is NaN.

        29 February is averaged over leap years only. If no date falls on 29
        February, that slot instead takes the average of the 28 February and 1 March
        statistics. Any other calendar day with no dates is left as NaN; ``n_dates``
        records which days these are so the caller can decide whether that is an
        error.

        Args:
            dataset: The dataset to read fields from.
            dates: The dates to average over.

        """
        by_day: dict[int, list[np.datetime64]] = defaultdict(list)
        for day in dates:
            by_day[cls.day_index(day)].append(day)

        shape = (cls.N_DAYS, *dataset.space.chw)
        mean = np.full(shape, np.nan, dtype=np.float32)
        std = np.full(shape, np.nan, dtype=np.float32)
        n_dates = np.zeros(cls.N_DAYS, dtype=np.int64)
        for index, day_dates in by_day.items():
            fields = dataset.get_tchw(day_dates).astype(np.float64)
            valid = np.isfinite(fields)
            count = valid.sum(axis=0)
            values = np.where(valid, fields, 0.0)
            # Pixels with no finite values give 0/0, i.e. NaN
            with np.errstate(divide="ignore", invalid="ignore"):
                day_mean = values.sum(axis=0) / count
                day_variance = (values**2).sum(axis=0) / count - day_mean**2
            mean[index] = day_mean
            std[index] = np.sqrt(np.maximum(day_variance, 0.0))
            n_dates[index] = len(day_dates)

        if n_dates[cls.FEBRUARY_29] == 0:
            logger.info(
                "Climatology: no 29 February dates in the averaging period; using the "
                "average of the 28 February and 1 March statistics instead."
            )
            for table in (mean, std):
                table[cls.FEBRUARY_29] = (
                    table[cls.FEBRUARY_28] + table[cls.MARCH_1]
                ) / 2

        if n_empty := int(np.isnan(mean[n_dates > 0]).sum()):
            logger.warning(
                "Climatology: %d calendar-day pixels have no finite values and are NaN.",
                n_empty,
            )
        return cls(mean=mean, std=std, n_dates=n_dates)
