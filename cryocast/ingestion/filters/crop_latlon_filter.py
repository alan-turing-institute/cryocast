"""Crop gridded fields to a rectangular latitude/longitude region."""

from typing import TYPE_CHECKING, cast

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.fields import (
    WrappedField,
    new_field_from_numpy,
    new_fieldlist_from_list,
)
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry
from earthkit.data import Field

from cryocast.geotools import GeographicField, GeographicGrid

if TYPE_CHECKING:
    from collections.abc import Sequence

N_SPATIAL_DIMS = 2
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0


class ExplicitGeographicGrid(GeographicGrid):
    """A 2D grid whose latitude/longitude coordinates are supplied explicitly."""

    def __init__(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        *,
        resolution: str = "native",
    ) -> None:
        """Initialise an explicit curvilinear latitude/longitude grid."""
        if latitudes.shape != longitudes.shape or latitudes.ndim != N_SPATIAL_DIMS:
            msg = "Explicit grid latitude/longitude arrays must have equal 2D shapes."
            raise ValueError(msg)
        self.latitudes_ = np.asarray(latitudes)
        self.longitudes_ = np.asarray(longitudes)
        # x/y are only used by the inherited Geography helpers. The original native
        # projection is intentionally not claimed after cropping; lat/lon coordinates
        # are authoritative for this curvilinear subset.
        self.x_ = self.longitudes_
        self.y_ = self.latitudes_
        self.resolution_ = resolution
        self.native_crs = "EPSG:4326"


@filter_registry.register("crop-latlon")
class CropLatLonFilter(Filter):
    """Crop a structured field list to the smallest native-grid box covering an ROI."""

    def __init__(
        self,
        *,
        north: float,
        west: float,
        south: float,
        east: float,
        resolution: str = "native",
    ) -> None:
        """Initialise a latitude/longitude region in north/west/south/east order."""
        if not MIN_LATITUDE <= south < north <= MAX_LATITUDE:
            msg = f"Invalid latitude bounds: north={north}, south={south}."
            raise ValueError(msg)
        if (
            not MIN_LONGITUDE <= west <= MAX_LONGITUDE
            or not MIN_LONGITUDE <= east <= MAX_LONGITUDE
        ):
            msg = f"Longitude bounds must be within [-180, 180]: west={west}, east={east}."
            raise ValueError(msg)
        self.north = north
        self.west = west
        self.south = south
        self.east = east
        self.resolution = resolution

    @staticmethod
    def _wrap_longitudes(longitudes: np.ndarray) -> np.ndarray:
        """Normalise longitudes to [-180, 180)."""
        return (longitudes + 180.0) % 360.0 - 180.0

    def _crop_slices(self, field: Field) -> tuple[slice, slice, ExplicitGeographicGrid]:
        """Return the minimal structured-grid slices that cover the requested ROI."""
        latitudes, longitudes = field.grid_points()
        latitudes = np.asarray(latitudes).reshape(field.shape)
        longitudes = self._wrap_longitudes(np.asarray(longitudes).reshape(field.shape))

        latitude_mask = (latitudes >= self.south) & (latitudes <= self.north)
        if self.west <= self.east:
            longitude_mask = (longitudes >= self.west) & (longitudes <= self.east)
        else:
            # Support regions crossing the antimeridian.
            longitude_mask = (longitudes >= self.west) | (longitudes <= self.east)
        inside = latitude_mask & longitude_mask
        row_indices, column_indices = np.nonzero(inside)
        if row_indices.size == 0:
            msg = (
                "Requested ROI contains no grid points: "
                f"north={self.north}, west={self.west}, "
                f"south={self.south}, east={self.east}."
            )
            raise ValueError(msg)

        rows = slice(int(row_indices.min()), int(row_indices.max()) + 1)
        columns = slice(int(column_indices.min()), int(column_indices.max()) + 1)
        geography = ExplicitGeographicGrid(
            latitudes[rows, columns],
            longitudes[rows, columns],
            resolution=self.resolution,
        )
        return rows, columns, geography

    def forward(self, data: ekd.FieldList | pd.DataFrame) -> ekd.FieldList:
        """Crop every field using one consistent native-grid index box."""
        if not isinstance(data, ekd.FieldList):
            msg = f"Expected data to be a FieldList, but got {type(data)}."
            raise TypeError(msg)
        fields = list(cast("Sequence[Field]", data))
        if not fields:
            return new_fieldlist_from_list([])

        source_shape = fields[0].shape
        rows, columns, geography = self._crop_slices(fields[0])
        cropped_fields = []
        for field in fields:
            if field.shape != source_shape:
                msg = (
                    "All fields must share one structured grid for crop-latlon; "
                    f"expected {source_shape}, got {field.shape}."
                )
                raise ValueError(msg)
            cropped_fields.append(
                new_field_from_numpy(
                    field.to_numpy(flatten=True).reshape(source_shape)[rows, columns],
                    template=WrappedField(GeographicField(field, geography)),
                )
            )
        return new_fieldlist_from_list(cropped_fields)
