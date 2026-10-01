from functools import cache
from typing import TYPE_CHECKING, cast

import earthkit.data as ekd
import pandas as pd
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter

from icenet_mp.geotools import (
    GeographicField,
    GeographicGrid,
    grid_factory,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from earthkit.data import Field


class SetGeographyFilter(Filter):
    """A filter to correct the geography of a field."""

    def __init__(self, *, crs: str, resolution: str) -> None:
        """Initialise a SetGeographyFilter with the specified CRS and resolution."""
        self.crs_ = crs
        self.resolution_ = resolution
        self.cached_geography = cache(self.geography)

    def geography(self, shape: tuple[int, int]) -> GeographicGrid:
        return grid_factory.create(self.crs_, resolution=self.resolution_, shape=shape)

    def forward(self, data: ekd.FieldList | pd.DataFrame) -> ekd.FieldList:
        """Wrap each input field with the configured geography.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data, wrapped with the configured geography.

        """
        if not isinstance(data, ekd.FieldList):
            msg = f"Expected data to be a FieldList, but got {type(data)}."
            raise TypeError(msg)

        return new_fieldlist_from_list(
            [
                GeographicField(field, self.cached_geography(field.shape[-2:]))
                for field in cast("Iterable[Field]", data)
            ]
        )
