"""Calendar-day statistics used to build the climatology baseline."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from icenet_mp.types import ArrayTCHW

from .calendar_day import (
    FEBRUARY_28_INDEX,
    FEBRUARY_29_INDEX,
    MARCH_1_INDEX,
    N_CALENDAR_DAYS,
    calendar_day_index,
)
from .single_dataset import SingleDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalendarDayStatistics:
    """Pixel-wise statistics for each calendar day, in the dataset's own units.

    Attributes:
        mean: [366, C, H, W] mean field for each calendar day.
        std: [366, C, H, W] population standard deviation for each calendar day.
        n_dates: [366] number of dates contributing to each calendar day.

    """

    mean: ArrayTCHW
    std: ArrayTCHW
    n_dates: np.ndarray


def calendar_day_statistics(
    dataset: SingleDataset, dates: Sequence[np.datetime64]
) -> CalendarDayStatistics:
    """Compute pixel-wise mean and standard deviation for each calendar day.

    Dates are grouped by calendar day (month/day label) rather than ordinal
    day-of-year so dates after February remain aligned in leap and non-leap years.
    Sums are accumulated in float64. Non-finite values are excluded pixel by pixel,
    and a pixel with no finite values on a calendar day is NaN.

    29 February is averaged over leap years only. If no date falls on 29 February,
    that slot instead takes the average of the 28 February and 1 March statistics.
    Any other calendar day with no dates is left as NaN; ``n_dates`` records which
    days these are so the caller can decide whether that is an error.

    Args:
        dataset: The dataset to read fields from.
        dates: The dates to average over.

    """
    by_day: dict[int, list[np.datetime64]] = defaultdict(list)
    for day in dates:
        by_day[calendar_day_index(day)].append(day)

    shape = (N_CALENDAR_DAYS, *dataset.space.chw)
    mean = np.full(shape, np.nan, dtype=np.float32)
    std = np.full(shape, np.nan, dtype=np.float32)
    n_dates = np.zeros(N_CALENDAR_DAYS, dtype=np.int64)
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

    if n_dates[FEBRUARY_29_INDEX] == 0:
        logger.info(
            "Climatology: no 29 February dates in the averaging period; using the "
            "average of the 28 February and 1 March statistics instead."
        )
        for table in (mean, std):
            table[FEBRUARY_29_INDEX] = (
                table[FEBRUARY_28_INDEX] + table[MARCH_1_INDEX]
            ) / 2

    if n_empty := int(np.isnan(mean[n_dates > 0]).sum()):
        logger.warning(
            "Climatology: %d calendar-day pixels have no finite values and are NaN.",
            n_empty,
        )
    return CalendarDayStatistics(mean=mean, std=std, n_dates=n_dates)
