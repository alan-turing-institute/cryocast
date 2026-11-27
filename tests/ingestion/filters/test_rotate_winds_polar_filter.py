import numpy as np
import pytest
from earthkit.data import SimpleFieldList

from cryocast.ingestion.filters.reproject_filter import ReprojectFilter
from cryocast.ingestion.filters.rotate_winds_polar_filter import (
    RotateWindsPolarFilter,
)

from .conftest import make_array_field


def wind_fieldlist(
    u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> SimpleFieldList:
    """Build a FieldList holding a pair of 10u/10v fields on the given grid."""
    lats_flat, lons_flat = lats.ravel().tolist(), lons.ravel().tolist()
    return SimpleFieldList(
        [
            make_array_field(u, lats=lats_flat, lons=lons_flat, param="10u"),
            make_array_field(v, lats=lats_flat, lons=lons_flat, param="10v"),
        ]
    )


def by_param(data: SimpleFieldList) -> dict[str, np.ndarray]:
    """Return the values of each field in a FieldList, keyed by param."""
    return {f.metadata("param"): f.to_numpy() for f in data}


def rotate_filter(crs: str) -> RotateWindsPolarFilter:
    """Return a filter rotating 10u/10v onto the axes of the given grid."""
    return RotateWindsPolarFilter(x_wind="10u", y_wind="10v", crs=crs)


class TestRotateWindsPolarFilter:
    def setup_method(self) -> None:
        ReprojectFilter.nn_indices_cached.clear()
        RotateWindsPolarFilter.convergence_cached.clear()

    def teardown_method(self) -> None:
        ReprojectFilter.nn_indices_cached.clear()
        RotateWindsPolarFilter.convergence_cached.clear()

    @pytest.mark.parametrize(
        ("crs", "lat_top", "lat_bottom", "hemisphere_sign"),
        [("EPSG:6931", 90.0, 60.0, 1.0), ("EPSG:6932", -60.0, -90.0, -1.0)],
    )
    def test_rotates_reprojected_winds_onto_ease2_axes(
        self, crs: str, lat_top: float, lat_bottom: float, hemisphere_sign: float
    ) -> None:
        """Uniform lat/lon winds are rotated onto EASE2 x/y axes after reprojection.

        On EASE2 grids, the rotation angle from (east, north) to (x, y) is the
        longitude in the north and minus the longitude in the south.
        """
        lats_1d = np.linspace(lat_top, lat_bottom, 7)
        lons_1d = np.arange(-180.0, 180.1, 10.0)
        lons, lats = np.meshgrid(lons_1d, lats_1d)
        u = np.full(lats.shape, 3.0)
        v = np.full(lats.shape, 4.0)
        reprojected = ReprojectFilter(
            crs=crs, resolution="500km", shape=(4, 4)
        ).forward(wind_fieldlist(u, v, lats, lons))

        result = by_param(rotate_filter(crs).forward(reprojected))

        output_lons = next(iter(reprojected)).grid_points()[1].reshape(4, 4)
        angle = hemisphere_sign * np.deg2rad(output_lons)
        assert result["10u"].shape == (4, 4)
        assert result["10v"].shape == (4, 4)
        np.testing.assert_allclose(
            result["10u"], 3.0 * np.cos(angle) - 4.0 * np.sin(angle), atol=1e-6
        )
        np.testing.assert_allclose(
            result["10v"], 3.0 * np.sin(angle) + 4.0 * np.cos(angle), atol=1e-6
        )
        np.testing.assert_allclose(np.hypot(result["10u"], result["10v"]), 5.0)

    def test_backward_undoes_forward(self) -> None:
        """Rotating forward then backward returns the original components."""
        lats = np.array([[70.0, 70.0], [75.0, 75.0]])
        lons = np.array([[-120.0, 45.0], [10.0, 170.0]])
        u = np.array([[3.0, -1.0], [0.5, 7.0]])
        v = np.array([[4.0, 2.0], [-6.0, 0.0]])
        rotate = rotate_filter("EPSG:6931")

        result = by_param(
            rotate.backward(rotate.forward(wind_fieldlist(u, v, lats, lons)))
        )

        np.testing.assert_allclose(result["10u"], u, atol=1e-12)
        np.testing.assert_allclose(result["10v"], v, atol=1e-12)

    def test_calm_wind_stays_zero_and_nan_propagates(self) -> None:
        """Calm cells stay zero, and a NaN in either component gives NaN in both."""
        lats = np.array([[70.0, 70.0], [75.0, 75.0]])
        lons = np.array([[30.0, 90.0], [30.0, 90.0]])
        u = np.array([[0.0, np.nan], [1.0, 3.0]])
        v = np.array([[0.0, 1.0], [np.nan, 4.0]])

        result = by_param(
            rotate_filter("EPSG:6931").forward(wind_fieldlist(u, v, lats, lons))
        )

        for param in ("10u", "10v"):
            assert result[param][0, 0] == 0.0
            assert np.isnan(result[param][0, 1])
            assert np.isnan(result[param][1, 0])
            assert not np.isnan(result[param][1, 1])
