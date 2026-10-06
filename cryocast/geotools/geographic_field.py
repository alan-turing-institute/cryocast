from typing import Any, cast

import numpy as np
from earthkit.data import Field
from earthkit.data.core.metadata import Metadata
from numpy.typing import NDArray

from .geographic_grid import GeographicGrid
from .geographic_metadata import GeographicMetadata


class GeographicField(Field):
    def __init__(self, field: Field, geography: GeographicGrid) -> None:
        """Initialise a GeographicField from an EarthKit Field and a GeographicGrid."""
        self._field = field
        self.geo_metadata = GeographicMetadata(
            cast("Metadata", field.metadata()), geography
        )

    def __repr__(self) -> str:
        """Return a string representation of the GeographicField, including the underlying field."""
        return f"{self.__class__.__name__}({self._field!r})"

    @property
    def _metadata(self) -> Metadata:
        """Return the geographic metadata for the field."""
        return self.geo_metadata

    def _values(self, dtype: Any | None = None) -> Any:  # noqa: ANN401
        """Delegate creation of values to the underlying field."""
        return self._field._values(dtype)

    def clone(
        self,
        *,
        values: Any | None = None,  # noqa: ANN401
        metadata: Metadata | None = None,
        **kwargs: Any,
    ) -> "GeographicField":
        return GeographicField(
            self._field.clone(values=values, metadata=metadata, **kwargs),
            self.geo_metadata.geography,
        )

    def message(self) -> bytes:
        """Delegate message generation to the underlying field."""
        return self._field.message()

    def to_latlon(
        self,
        flatten: bool = True,  # noqa: FBT001, FBT002
        dtype: type | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        """Return the latitudes and longitudes for the field.

        We take these from the geography as GeographicField might be a wrapper around a
        differently-sized field. This is done, for example, in ReprojectFilter.
        """
        lats = self.geo_metadata.geography.latitudes()
        lons = self.geo_metadata.geography.longitudes()
        if flatten:
            lats = lats.flatten()
            lons = lons.flatten()
        if dtype is not None:
            lats = lats.astype(dtype)
            lons = lons.astype(dtype)
        if index is not None:
            lats = lats[index]
            lons = lons[index]
        return {"lat": lats, "lon": lons}

    def to_numpy(
        self,
        flatten: bool = False,  # noqa: FBT001, FBT002
        dtype: type | None = None,
        index: int | None = None,
    ) -> NDArray[np.generic]:
        """Delegate creation of numpy array to the underlying field.

        It is important to do this rather than relying on the base class implementation
        since `self._field` might override the base class method.
        """
        return self._field.to_numpy(flatten=flatten, dtype=dtype, index=index)
