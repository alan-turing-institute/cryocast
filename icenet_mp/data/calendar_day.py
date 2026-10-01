"""Calendar-day (month/day) indexing shared by the climatology table and its lookups.

Dates are grouped by month/day label (``"MM-DD"``) rather than ordinal day-of-year, so
29 February keeps its own slot and every other calendar day stays aligned between leap
and non-leap years.
"""

import numpy as np

N_CALENDAR_DAYS = 366

# 2000 is a leap year; used only to enumerate the ordered calendar-day labels below.
_LEAP_YEAR_ORIGIN = np.datetime64("2000-01-01")

CALENDAR_DAY_LABELS: list[str] = [
    str(
        np.datetime_as_string(_LEAP_YEAR_ORIGIN + np.timedelta64(offset, "D"), unit="D")
    )[5:]
    for offset in range(N_CALENDAR_DAYS)
]

_CALENDAR_DAY_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(CALENDAR_DAY_LABELS)
}

FEBRUARY_28_INDEX = _CALENDAR_DAY_INDEX["02-28"]
FEBRUARY_29_INDEX = _CALENDAR_DAY_INDEX["02-29"]


def calendar_day_index(day: np.datetime64) -> int:
    """Return the 0-365 calendar-day index (month/day label) for a date."""
    label = np.datetime_as_string(day.astype("datetime64[D]"), unit="D")[5:]
    return _CALENDAR_DAY_INDEX[label]
