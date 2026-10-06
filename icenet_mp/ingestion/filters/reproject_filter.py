import logging
from typing import TYPE_CHECKING, ClassVar, cast

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.fields import (
    WrappedField,
    new_field_from_numpy,
    new_fieldlist_from_list,
)
from anemoi.transform.filter import Filter
from earthkit.data import Field

from icenet_mp.geotools import (
    GeographicField,
    GeographicGrid,
    grid_factory,
    nearest_neighbour_indices,
)
from icenet_mp.types import ArrayHWV, ArrayIndices2D

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class ReprojectFilter(Filter):
    """A filter to reproject fields to a new grid."""

    # We cache at class level because each GroupOfDates in a dataset creates a new
    # instance of the filter.
    nn_indices_cached: ClassVar[
        dict[tuple[int, int, int, int], tuple[ArrayIndices2D, ArrayIndices2D]]
    ] = {}

    def __init__(self, *, crs: str, resolution: str, shape: tuple[int, int]) -> None:
        """Initialise the filter with the output grid parameters."""
        self.output_geography: GeographicGrid = grid_factory.create(
            crs, resolution=resolution, shape=shape
        )

    def nearest_neighbours(
        self, data: ekd.FieldList
    ) -> tuple[ArrayIndices2D, ArrayIndices2D]:
        """Determine the nearest neighbour input cell for each cell in the output grid.

        Returns:
            Tuple of (nn_indices_h, nn_indices_w) where each is a tensor of shape
            [output_height, output_width] containing, for each output cell, the index in
            the H and W dimensions of the nearest neighbour input cell that should be
            used as the source.

        """
        # Check that we can load a field from the
        if (
            field := next((field for field in data if isinstance(field, Field)), None)
        ) is None:
            msg = "No latitudes/longitudes were found in the input data."
            raise ValueError(msg)

        # Get the input grid from the data
        lats, lons = field.grid_points()
        input_latlons: ArrayHWV = np.stack(
            (
                np.clip(lats, -90, 90).reshape(field.shape),
                np.clip(lons, -180, 180).reshape(field.shape),
            ),
            axis=-1,
        )

        # Get the output grid from the output geography
        output_latlons: ArrayHWV = np.stack(
            (self.output_geography.latitudes(), self.output_geography.longitudes()),
            axis=-1,
        )

        # Ensure that nearest neighbour mapping is in the class level cache
        shape_key = (
            int(input_latlons.shape[0]),
            int(input_latlons.shape[1]),
            int(output_latlons.shape[0]),
            int(output_latlons.shape[1]),
        )
        if shape_key not in ReprojectFilter.nn_indices_cached:
            ReprojectFilter.nn_indices_cached[shape_key] = nearest_neighbour_indices(
                input_latlons, output_latlons
            )

        return ReprojectFilter.nn_indices_cached[shape_key]

    def forward(self, data: ekd.FieldList | pd.DataFrame) -> ekd.FieldList:
        """Apply the forward regridding transformation.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.

        """
        if not isinstance(data, ekd.FieldList):
            msg = f"Expected data to be a FieldList, but got {type(data)}."
            raise TypeError(msg)

        return new_fieldlist_from_list(
            [
                new_field_from_numpy(
                    field.to_numpy()[*self.nearest_neighbours(data)],
                    template=WrappedField(
                        GeographicField(field, self.output_geography)
                    ),
                )
                for field in cast("Sequence[Field]", data)
            ]
        )
