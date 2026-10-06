"""Anemoi source for deterministic synthetic sea-ice trajectories."""

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import xarray as xr
from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data import FieldList
from typing_extensions import override

from cryocast.synthetic.trajectories import generate_default_dataset


@source_registry.register("synthetic")
class SyntheticSource(Source):
    """Generate deterministic moving-shape data for an Anemoi recipe."""

    def __init__(
        self,
        context: Context,
        *,
        dynamics: str,
        grid_size: int,
        trajectory_start_dates: Sequence[str],
        variable_name: str = "ice_conc",
    ) -> None:
        """Configure a synthetic dataset source."""
        self.context = context
        self.variable_name = variable_name
        self.dataset = generate_default_dataset(
            dynamics=dynamics,
            grid_size=grid_size,
            start_dates=(
                [datetime.fromisoformat(date) for date in trajectory_start_dates]
            ),
        )
        self.frames_by_date = {
            date.date(): frame
            for date, frame in zip(self.dataset.dates, self.dataset.frames, strict=True)
        }

    @override
    def execute(self, argument: list[datetime] | GroupOfDates) -> FieldList:
        """Return requested available frames as an Anemoi field list."""
        available_dates = sorted(
            date for date in argument if date.date() in self.frames_by_date
        )
        frames = np.stack(
            [self.frames_by_date[date.date()] for date in available_dates]
        )
        grid_size = frames.shape[-1]
        latitudes, longitudes = np.meshgrid(
            np.linspace(-90, 90, grid_size),
            np.linspace(-180, 180, grid_size),
            indexing="ij",
        )
        dataset = xr.Dataset(
            data_vars={
                self.variable_name: (("time", "x_pos", "y_pos"), frames),
            },
            coords={
                "time": (
                    "time",
                    np.asarray(available_dates, dtype="datetime64[ns]"),
                    {
                        "standard_name": "time",
                        "units": "seconds since 1970-01-01 00:00:00",
                        "calendar": "standard",
                    },
                ),
                "lat": (
                    ("x_pos", "y_pos"),
                    latitudes,
                    {"standard_name": "latitude", "units": "degrees_north"},
                ),
                "lon": (
                    ("x_pos", "y_pos"),
                    longitudes,
                    {"standard_name": "longitude", "units": "degrees_east"},
                ),
            },
        )
        return load_one(
            "🔬", self.context, [date.isoformat() for date in available_dates], dataset
        )
