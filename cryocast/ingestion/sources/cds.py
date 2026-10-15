"""Copernicus Climate Data Store source for Anemoi recipes."""

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, ClassVar

from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data import FieldList, from_source
from earthkit.data.core.fieldlist import MultiFieldList
from typing_extensions import override

logger = logging.getLogger(__name__)


@source_registry.register("cds")
class CDSSource(Source):
    """Download exact requested dates from a CDS dataset."""

    date_keys: ClassVar[frozenset[str]] = frozenset({"year", "month", "day"})

    def __init__(
        self,
        context: Context,
        *,
        dataset: str,
        request: dict[str, Any],
        time_from_dates: bool = False,
    ) -> None:
        """Initialise the source with a CDS dataset and request template.

        ``year``, ``month`` and ``day`` are supplied from Anemoi's requested dates,
        preventing a recipe chunk from accidentally downloading data outside that
        chunk. Set ``time_from_dates`` for sub-daily products whose valid time should
        also follow the recipe dates.
        """
        if not dataset:
            msg = "CDS dataset name must not be empty."
            raise ValueError(msg)
        reserved = self.date_keys.intersection(request)
        if reserved:
            msg = (
                "CDS request must not set date fields managed by the recipe: "
                f"{sorted(reserved)}."
            )
            raise ValueError(msg)
        if time_from_dates and "time" in request:
            msg = "CDS request must not set 'time' when time_from_dates=True."
            raise ValueError(msg)

        self.context: Context = context
        self.dataset = dataset
        self.request = deepcopy(request)
        self.time_from_dates = time_from_dates

    @override
    def execute(self, argument: list[datetime] | GroupOfDates) -> FieldList:
        """Download the requested dates, split into exact year/month requests."""
        requested_dates = sorted(argument)
        if not requested_dates:
            return MultiFieldList([])

        dates_by_request: dict[tuple[int, int, str | None], list[datetime]] = (
            defaultdict(list)
        )
        for date in requested_dates:
            request_time = date.strftime("%H:%M") if self.time_from_dates else None
            dates_by_request[(date.year, date.month, request_time)].append(date)

        field_lists: list[FieldList] = []
        for (year, month, request_time), dates in sorted(dates_by_request.items()):
            request = deepcopy(self.request)
            request.update(
                {
                    "year": [f"{year:04d}"],
                    "month": [f"{month:02d}"],
                    "day": sorted({date.strftime("%d") for date in dates}),
                }
            )
            if request_time is not None:
                request["time"] = [request_time]

            logger.info(
                "Requesting %s from CDS for %04d-%02d (%d date(s)).",
                self.dataset,
                year,
                month,
                len(dates),
            )
            self.context.trace("🌍", f"cds {self.dataset} {request}")
            field_lists.append(
                from_source(
                    "cds",
                    self.dataset,
                    request=request,
                    prompt=False,
                )
            )

        return MultiFieldList(field_lists)
