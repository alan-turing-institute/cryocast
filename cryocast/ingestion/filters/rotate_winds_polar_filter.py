"""Filter that rotates eastward/northward wind components onto a projected grid's axes."""

from collections.abc import Iterator
from typing import ClassVar

import earthkit.data as ekd
import numpy as np
from anemoi.transform.filters.fields.matching import (
    MatchingFieldsFilter,
    MatchingSpec,
)
from numpy.typing import NDArray
from pyproj import Proj

from .reproject_filter import grid_key


class RotateWindsPolarFilter(MatchingFieldsFilter):
    """Rotate eastward/northward wind components onto the x/y axes of a projected grid.

    Each (u, v) pair is rotated by the grid convergence angle: the angle between true
    north and grid north at each point, as reported by pyproj. On the EASE2 polar
    grids this is +longitude (EPSG:6931) or -longitude (EPSG:6932). This is a pure
    rotation, so wind speed is unchanged.

    We do not use anemoi-transform's `rotate_winds` here. It maps a small step along
    the wind vector through the projection, which equals a pure rotation only on
    conformal projections. EASE2 grids are equal-area, not conformal, so that method
    also applies the projection's shape distortion (about 2 degrees at 60N, more
    towards the grid corners) and returns NaN for calm cells.

    Input components must be eastward/northward (e.g. ERA5 u/v) on fields whose grid
    points are in `crs`, e.g. after the `reproject` filter.
    """

    MATCHING = MatchingSpec(
        select="param",
        forward=("x_wind", "y_wind"),
        backward=("x_wind", "y_wind"),
    )

    # Convergence angles (radians) for each grid. We cache at class level because each
    # GroupOfDates in a dataset creates a new instance of the filter.
    convergence_cached: ClassVar[dict[tuple[str, str], NDArray[np.float64]]] = {}

    def __init__(self, *, x_wind: str, y_wind: str, crs: str) -> None:
        """Initialise the filter.

        Parameters
        ----------
        x_wind : str
            Parameter name of the eastward wind component, e.g. "10u".
        y_wind : str
            Parameter name of the northward wind component, e.g. "10v".
        crs : str
            CRS of the grid the fields are on, e.g. "EPSG:6931".

        """
        # x_wind and y_wind must be explicit parameters: MatchingFieldsFilter checks
        # this constructor's signature for them when the class is defined.
        self.x_wind = x_wind
        self.y_wind = y_wind
        self.crs = crs
        super().__init__()

    def convergence(self, field: ekd.Field) -> NDArray[np.float64]:
        """Return the grid convergence angle in radians at each point of the field.

        Parameters
        ----------
        field : ekd.Field
            A field on the grid of interest.

        Returns
        -------
        NDArray[np.float64]
            Angle from true north to grid north at each point, with the field's shape.

        """
        lats, lons = field.grid_points()
        key = (self.crs, grid_key(np.stack((lats, lons), axis=-1)))
        if key not in RotateWindsPolarFilter.convergence_cached:
            factors = Proj(self.crs).get_factors(lons, lats)
            RotateWindsPolarFilter.convergence_cached[key] = np.deg2rad(
                np.asarray(factors.meridian_convergence)
            ).reshape(field.shape)
        return RotateWindsPolarFilter.convergence_cached[key]

    def rotate(
        self, x_wind: ekd.Field, y_wind: ekd.Field, angle: NDArray[np.float64]
    ) -> Iterator[ekd.Field]:
        """Rotate a pair of wind component fields anticlockwise by `angle`.

        Parameters
        ----------
        x_wind : ekd.Field
            The x wind component field.
        y_wind : ekd.Field
            The y wind component field, on the same grid as `x_wind`.
        angle : NDArray[np.float64]
            Rotation angle in radians at each point, with the fields' shape.

        Yields
        ------
        ekd.Field
            The rotated x and y wind component fields, with the input fields' shape.

        """
        u = x_wind.to_numpy().reshape(x_wind.shape)
        v = y_wind.to_numpy().reshape(y_wind.shape)
        cos, sin = np.cos(angle), np.sin(angle)
        yield self.new_field_from_numpy(
            u * cos - v * sin, template=x_wind, param=x_wind.metadata("param")
        )
        yield self.new_field_from_numpy(
            u * sin + v * cos, template=y_wind, param=y_wind.metadata("param")
        )

    def forward_transform(
        self, x_wind: ekd.Field, y_wind: ekd.Field
    ) -> Iterator[ekd.Field]:
        """Rotate eastward/northward components onto the grid's x/y axes."""
        return self.rotate(x_wind, y_wind, self.convergence(x_wind))

    def backward_transform(
        self, x_wind: ekd.Field, y_wind: ekd.Field
    ) -> Iterator[ekd.Field]:
        """Rotate grid x/y components back to eastward/northward."""
        return self.rotate(x_wind, y_wind, -self.convergence(x_wind))
