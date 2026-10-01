from unittest.mock import MagicMock

import numpy as np
from earthkit.data import Field

from icenet_mp.geotools.geographic_field import GeographicField
from icenet_mp.geotools.geographic_grid import GeographicGrid
from icenet_mp.geotools.geographic_metadata import GeographicMetadata


class TestGeographicField:
    """Unit tests for the GeographicField class."""

    def test_init_wraps_field_metadata_in_geographic_metadata(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """Construction eagerly wraps the underlying field's metadata with the geography."""
        mock_field = MagicMock(spec=Field)

        field = GeographicField(mock_field, minimal_grid)

        mock_field.metadata.assert_called_once()
        assert isinstance(field.geo_metadata, GeographicMetadata)
        assert field.geo_metadata.geography is minimal_grid
        assert field._metadata is field.geo_metadata

    def test_repr_includes_wrapped_field(self, minimal_grid: GeographicGrid) -> None:
        """__repr__ reports the wrapped field's own repr."""
        mock_field = MagicMock(spec=Field)
        field = GeographicField(mock_field, minimal_grid)

        assert repr(field) == f"GeographicField({mock_field!r})"

    def test_values_delegates_to_wrapped_field(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """_values() forwards the dtype argument to the wrapped field."""
        mock_field = MagicMock(spec=Field)
        mock_field._values.return_value = np.array([1.0, 2.0])
        field = GeographicField(mock_field, minimal_grid)

        result = field._values(dtype=np.float32)

        mock_field._values.assert_called_once_with(np.float32)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_message_delegates_to_wrapped_field(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """message() forwards to the wrapped field."""
        mock_field = MagicMock(spec=Field)
        mock_field.message.return_value = b"grib-bytes"
        field = GeographicField(mock_field, minimal_grid)

        assert field.message() == b"grib-bytes"

    def test_to_numpy_delegates_to_wrapped_field(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """to_numpy() forwards flatten/dtype/index to the wrapped field."""
        mock_field = MagicMock(spec=Field)
        mock_field.to_numpy.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        field = GeographicField(mock_field, minimal_grid)

        result = field.to_numpy(flatten=True, dtype=np.float32, index=1)

        mock_field.to_numpy.assert_called_once_with(
            flatten=True, dtype=np.float32, index=1
        )
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_clone_wraps_cloned_field_with_same_geography(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """clone() clones the wrapped field and rewraps it with the original geography."""
        mock_field = MagicMock(spec=Field)
        cloned_inner_field = MagicMock(spec=Field)
        mock_field.clone.return_value = cloned_inner_field
        field = GeographicField(mock_field, minimal_grid)

        new_values = np.array([5.0, 6.0])
        clone = field.clone(values=new_values)

        mock_field.clone.assert_called_once_with(values=new_values, metadata=None)
        assert isinstance(clone, GeographicField)
        assert clone._field is cloned_inner_field
        assert clone.geo_metadata.geography is minimal_grid

    def test_to_latlon_returns_flattened_geography_coordinates_by_default(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """to_latlon() defaults to flattened lat/lon arrays taken from the geography."""
        mock_field = MagicMock(spec=Field)
        field = GeographicField(mock_field, minimal_grid)

        result = field.to_latlon()

        np.testing.assert_allclose(result["lat"], minimal_grid.latitudes().flatten())
        np.testing.assert_allclose(result["lon"], minimal_grid.longitudes().flatten())
        assert result["lat"].ndim == 1

    def test_to_latlon_can_keep_grid_shape_and_convert_dtype(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """to_latlon(flatten=False, dtype=...) preserves grid shape and casts dtype."""
        mock_field = MagicMock(spec=Field)
        field = GeographicField(mock_field, minimal_grid)

        result = field.to_latlon(flatten=False, dtype=np.float32)

        assert result["lat"].shape == minimal_grid.shape()
        assert result["lat"].dtype == np.float32
        assert result["lon"].dtype == np.float32

    def test_to_latlon_can_select_a_single_index(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """to_latlon(index=...) selects a single element from the flattened arrays."""
        mock_field = MagicMock(spec=Field)
        field = GeographicField(mock_field, minimal_grid)

        result = field.to_latlon(index=0)

        assert result["lat"] == minimal_grid.latitudes().flatten()[0]
        assert result["lon"] == minimal_grid.longitudes().flatten()[0]
