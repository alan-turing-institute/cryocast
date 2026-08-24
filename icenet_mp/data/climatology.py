"""Utilities for generating daily climatology baselines from SIC datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .single_dataset import SingleDataset


@dataclass(frozen=True)
class DailyClimatology:
    """A calendar-day climatology and the number of samples behind each pixel."""

    values: np.ndarray
    sample_count: np.ndarray
    month_day: np.ndarray
    dataset_name: str
    variable: str
    reference_start_year: int
    reference_end_year: int


def _calendar_days() -> np.ndarray:
    """Return month/day labels for a leap-year calendar, including 29 February."""
    start = np.datetime64("2000-01-01")
    return np.array(
        [
            np.datetime_as_string(start + np.timedelta64(day, "D"), unit="D")[5:]
            for day in range(366)
        ]
    )


def generate_daily_climatology(
    dataset: SingleDataset,
    *,
    variable: str,
    reference_end_year: int,
    years: int = 35,
) -> DailyClimatology:
    """Generate a pixel-wise daily climatology over a complete reference-year window.

    Calendar dates are grouped by month/day rather than ordinal day-of-year so dates
    after February remain aligned in leap and non-leap years. The 29 February field is
    averaged over leap years only. Non-finite pixels are excluded independently.
    """
    if years <= 0:
        msg = "years must be greater than 0."
        raise ValueError(msg)
    if variable not in dataset.variable_names:
        msg = f"Variable {variable!r} not found in dataset {dataset.name!r}."
        raise ValueError(msg)

    source = dataset.subset(variables=[variable])
    reference_start_year = reference_end_year - years + 1
    selected_dates = [
        date
        for date in source.dates
        if reference_start_year
        <= int(np.datetime_as_string(date, unit="Y"))
        <= reference_end_year
    ]
    observed_years = {
        int(np.datetime_as_string(date, unit="Y")) for date in selected_dates
    }
    expected_years = set(range(reference_start_year, reference_end_year + 1))
    missing_years = sorted(expected_years - observed_years)
    if missing_years:
        msg = (
            "Cannot generate a complete climatology reference period; missing years: "
            f"{missing_years}."
        )
        raise ValueError(msg)

    month_day = _calendar_days()
    day_indices = {label: idx for idx, label in enumerate(month_day.tolist())}
    height, width = source.space.shape
    sums = np.zeros((366, height, width), dtype=np.float64)
    sample_count = np.zeros((366, height, width), dtype=np.int32)

    for date in selected_dates:
        label = np.datetime_as_string(date, unit="D")[5:]
        values = source.get_tchw([date])[0, 0].astype(np.float64, copy=False)
        valid = np.isfinite(values)
        index = day_indices[label]
        sums[index][valid] += values[valid]
        sample_count[index][valid] += 1

    climatology = np.full((366, height, width), np.nan, dtype=np.float32)
    np.divide(
        sums,
        sample_count,
        out=climatology,
        where=sample_count > 0,
        casting="unsafe",
    )
    return DailyClimatology(
        values=climatology,
        sample_count=sample_count,
        month_day=month_day,
        dataset_name=dataset.name,
        variable=variable,
        reference_start_year=reference_start_year,
        reference_end_year=reference_end_year,
    )


def save_daily_climatology(climatology: DailyClimatology, path: Path) -> Path:
    """Save a daily climatology and its provenance to a compressed NumPy archive."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        climatology=climatology.values,
        sample_count=climatology.sample_count,
        month_day=climatology.month_day,
        dataset_name=np.array(climatology.dataset_name),
        variable=np.array(climatology.variable),
        reference_start_year=np.array(climatology.reference_start_year),
        reference_end_year=np.array(climatology.reference_end_year),
    )
    return path
