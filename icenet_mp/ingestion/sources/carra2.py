import logging
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import cdsapi
from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data import from_source
from earthkit.data.core.fieldlist import FieldList, MultiFieldList
from typing_extensions import override

logger = logging.getLogger(__name__)


@source_registry.register("carra2")
class CARRA2Source(Source):
    """Download CARRA2 pan-Arctic reanalysis fields from the CDS.

    CARRA2 is exposed through the Climate Data Store as ``reanalysis-pan-carra``.
    Requests are made one analysis time at a time so the returned fields match the
    exact dates requested by Anemoi rather than a Cartesian product of years, months,
    days and times.
    """

    def __init__(  # noqa: PLR0913
        self,
        context: Context,
        *,
        variables: list[str],
        level_type: str = "single_levels",
        level_location: list[str] | None = None,
        product_type: str = "analysis",
        data_format: str = "grib",
        area: list[float] | None = None,
        dataset: str = "reanalysis-pan-carra",
    ) -> None:
        """Initialise the CARRA2 source."""
        if not variables:
            msg = "At least one CARRA2 variable must be requested."
            raise ValueError(msg)
        if data_format not in {"grib", "netcdf"}:
            msg = "CARRA2 data_format must be either 'grib' or 'netcdf'."
            raise ValueError(msg)
        if area is not None and len(area) != 4:  # noqa: PLR2004
            msg = "CARRA2 area must contain [north, west, south, east]."
            raise ValueError(msg)

        self.context = context
        self.variables = variables
        self.level_type = level_type
        self.level_location = level_location
        self.product_type = product_type
        self.data_format = data_format
        self.area = area
        self.dataset = dataset

    def _request_for_date(
        self, date: datetime
    ) -> dict[str, str | list[str] | list[float]]:
        """Build a CDS request for one exact CARRA2 valid time."""
        request: dict[str, str | list[str] | list[float]] = {
            "level_type": self.level_type,
            "variable": self.variables,
            "product_type": self.product_type,
            "time": [date.strftime("%H:%M")],
            "year": [date.strftime("%Y")],
            "month": [date.strftime("%m")],
            "day": [date.strftime("%d")],
            "data_format": self.data_format,
        }
        if self.level_location is not None:
            request["level_location"] = self.level_location
        if self.area is not None:
            request["area"] = self.area
        return request

    @override
    def execute(self, argument: list[datetime] | GroupOfDates) -> FieldList:
        """Download requested CARRA2 dates and return them as an Earthkit FieldList."""
        field_lists: list[FieldList] = []
        extension = "grib" if self.data_format == "grib" else "nc"

        with TemporaryDirectory() as tmpdir:
            client = cdsapi.Client()
            for date in argument:
                target = Path(tmpdir) / f"carra2-{date:%Y%m%d%H%M}.{extension}"
                request = self._request_for_date(date)
                logger.info(
                    "Downloading CARRA2 %s with %d variable(s).",
                    date.isoformat(),
                    len(self.variables),
                )
                client.retrieve(self.dataset, request, str(target))
                field_lists.append(from_source("file", str(target)))

        return MultiFieldList(field_lists)
