from unittest.mock import MagicMock

import numpy as np
import pytest
from earthkit.data.utils.bbox import BoundingBox
from pyproj import Transformer

from icenet_mp.geotools.geographic_grid import GeographicGrid
from icenet_mp.geotools.grid_factory import ease2_grid_helper


class TestGeographicGrid:
    """Unit tests for the GeographicGrid class."""

    def test_resolution_and_shape_return_constructor_values(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """resolution() and shape() report the values derived at construction."""
        assert minimal_grid.resolution() == "1.0"
        assert minimal_grid.shape() == (2, 2)

    def test_coordinate_accessors_support_dtype_conversion(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """Allow callers to request coordinate arrays in a different dtype."""
        assert minimal_grid.x(dtype=np.float32).dtype == np.float32
        assert minimal_grid.y(dtype=np.float32).dtype == np.float32
        assert minimal_grid.latitudes(dtype=np.float32).dtype == np.float32
        assert minimal_grid.longitudes(dtype=np.float32).dtype == np.float32

    def test_latitudes_and_longitudes_pass_through_for_epsg_4326(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """A native EPSG:4326 grid returns its x/y arrays as lon/lat without transforming."""
        np.testing.assert_allclose(minimal_grid.longitudes(), minimal_grid.x())
        np.testing.assert_allclose(minimal_grid.latitudes(), minimal_grid.y())

    def test_latitudes_and_longitudes_transform_and_cache_for_projected_crs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A projected grid transforms x/y to lat/lon once and caches the result."""
        from_crs_spy = MagicMock(wraps=Transformer.from_crs)
        monkeypatch.setattr(Transformer, "from_crs", from_crs_spy)

        _, h_points, w_points = ease2_grid_helper("500p0km", 2, 2)
        grid = GeographicGrid("EPSG:6931", "500p0km", h_points, w_points)

        expected_lat, expected_lon = Transformer.from_crs(
            "EPSG:6931", "EPSG:4326"
        ).transform(grid.x(), grid.y())

        np.testing.assert_allclose(grid.latitudes(), expected_lat)
        np.testing.assert_allclose(grid.longitudes(), expected_lon)
        grid.latitudes()
        grid.longitudes()

        # One call above to build `expected_*`, one for the grid under test.
        assert from_crs_spy.call_count == 2

    def test_mars_area(self, minimal_grid: GeographicGrid) -> None:
        """mars_area() reports the (north, west, south, east) bounds of the grid."""
        assert minimal_grid.mars_area() == (89.5, -0.5, 88.5, 0.5)

    def test_mars_grid(self, minimal_grid: GeographicGrid) -> None:
        """mars_grid() reports the per-step lat/lon spacing implied by the grid extent."""
        lat_step, lon_step = minimal_grid.mars_grid()

        assert lat_step == pytest.approx(0.5)
        assert lon_step == pytest.approx(0.5)

    def test_bounding_box_matches_mars_area(self, minimal_grid: GeographicGrid) -> None:
        """bounding_box() wraps mars_area() in an earthkit BoundingBox."""
        box = minimal_grid.bounding_box()

        assert box == BoundingBox(north=89.5, west=-0.5, south=88.5, east=0.5)

    def test_unimplemented_earthkit_grid_methods_raise(
        self, minimal_grid: GeographicGrid
    ) -> None:
        """Keep unsupported EarthKit geography methods explicit rather than silent."""
        for method in (
            minimal_grid._unique_grid_id,
            minimal_grid.gridspec,
            minimal_grid.grid_spec,
            minimal_grid.projection,
        ):
            with pytest.raises(NotImplementedError):
                method()
