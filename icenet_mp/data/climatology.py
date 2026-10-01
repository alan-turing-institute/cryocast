"""Utilities for generating daily climatology baselines from SIC datasets."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from icenet_mp.types import ArrayTCHW

from .calendar_day import N_CALENDAR_DAYS, calendar_day_index
from .single_dataset import SingleDataset


@dataclass(frozen=True)
class DailyClimatology:
    """A calendar-day climatology and the number of samples behind each pixel."""

    values: ArrayTCHW
    sample_count: np.ndarray


def generate_daily_climatology(
    dataset: SingleDataset, dates: Sequence[np.datetime64]
) -> DailyClimatology:
    """Generate a pixel-wise daily climatology over the given dates.

    Dates are grouped by calendar day (month/day label) rather than ordinal
    day-of-year so dates after February remain aligned in leap and non-leap years.
    The 29 February field is averaged over leap years only. Non-finite pixels are
    excluded independently, and a pixel with no finite samples is NaN.
    """
    channels, height, width = dataset.space.chw
    shape = (N_CALENDAR_DAYS, channels, height, width)
    sums = np.zeros(shape, dtype=np.float64)
    sample_count = np.zeros(shape, dtype=np.int32)

    for date in dates:
        values = dataset.get_tchw([date])[0].astype(np.float64, copy=False)
        valid = np.isfinite(values)
        index = calendar_day_index(date)
        sums[index][valid] += values[valid]
        sample_count[index][valid] += 1

    climatology = np.full(shape, np.nan, dtype=np.float32)
    np.divide(
        sums,
        sample_count,
        out=climatology,
        where=sample_count > 0,
        casting="unsafe",
    )
    return DailyClimatology(values=climatology, sample_count=sample_count)
