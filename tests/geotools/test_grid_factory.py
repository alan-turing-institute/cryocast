import numpy as np
import pytest

from icenet_mp.geotools import GeographicGrid, GridFactory
from icenet_mp.geotools.grid_factory import (
    ease2_grid_helper,
    epsg_4326n_builder,
    epsg_4326s_builder,
    epsg_6931_builder,
    epsg_6932_builder,
)


class TestGridFactory:
    """Unit tests for the GridFactory class."""

    def test_grid_factory_registers_and_dispatches_builder(self) -> None:
        """Create a grid through the builder registered for a CRS."""
        factory = GridFactory()
        factory.register_crs("EPSG:6931", epsg_6931_builder)

        grid = factory.create("EPSG:6931", resolution="25p0km", shape=(4, 6))

        assert isinstance(grid, GeographicGrid)
        assert grid.native_crs == "EPSG:6931"
        assert grid.resolution() == "25p0km"
        assert grid.shape() == (6, 4)

    def test_grid_factory_rejects_unknown_crs(self) -> None:
        """Reject requests for CRS builders that have not been registered."""
        with pytest.raises(
            ValueError, match="No builder registered for CRS: EPSG:9999"
        ):
            GridFactory().create("EPSG:9999", resolution="25p0km", shape=(4, 4))


class TestEase2GridHelper:
    """Unit tests for the ease2_grid_helper function."""

    def test_normalises_resolution_and_centres_grid(self) -> None:
        """Convert kilometre resolution to centred projected-coordinate arrays."""
        resolution, h_points, w_points = ease2_grid_helper("25p0km", 4, 4)

        assert resolution == "25p0km"
        np.testing.assert_allclose(h_points, [-37500.0, -12500.0, 12500.0, 37500.0])
        np.testing.assert_allclose(w_points, [37500.0, 12500.0, -12500.0, -37500.0])

    def test_accepts_metres_and_rejects_bad_format(self) -> None:
        """Normalise metre input and reject resolutions without an explicit unit."""
        resolution, _, _ = ease2_grid_helper("25000m", 3, 3)
        assert resolution == "25p0km"

        with pytest.raises(ValueError, match="Invalid resolution format"):
            ease2_grid_helper("25000", 3, 3)


class TestBuilderFunctions:
    """Unit tests for the module-level CRS builder functions."""

    def test_wgs84_north_builder_exposes_expected_coordinates(self) -> None:
        """Build a northern regular lat/lon grid without coordinate transformation."""
        grid = epsg_4326n_builder("1p0", (4, 4))

        assert grid.shape() == (4, 4)
        np.testing.assert_allclose(grid.x()[0], [-1.5, -0.5, 0.5, 1.5])
        np.testing.assert_allclose(grid.y()[:, 0], [86.5, 87.5, 88.5, 89.5])
        assert grid.mars_area() == (89.5, -1.5, 86.5, 1.5)

    def test_wgs84_south_builder_exposes_expected_coordinates(self) -> None:
        """Build a southern regular lat/lon grid with the requested shape."""
        grid = epsg_4326s_builder("1p0", (4, 4))

        np.testing.assert_allclose(grid.y()[:, 0], [-3.5, -2.5, -1.5, -0.5])
        assert grid.mars_area() == (-0.5, -1.5, -3.5, 1.5)

    def test_polar_builders_set_expected_crs_and_orientation(self) -> None:
        """Construct north/south EASE2 grids with matching projected coordinates."""
        north = epsg_6931_builder("25p0km", (3, 5))
        south = epsg_6932_builder("25p0km", (3, 5))

        assert north.native_crs == "EPSG:6931"
        assert south.native_crs == "EPSG:6932"
        assert north.shape() == south.shape() == (5, 3)
        np.testing.assert_allclose(north.x(), south.x())
        np.testing.assert_allclose(north.y(), south.y())
