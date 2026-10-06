import numpy as np
import pytest
import torch

from cryocast.models.common import GeographicInterpolation


def _source_coordinates() -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    longitudes = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    return latitudes, longitudes


def _target_coordinates() -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.array([[1.5, 1.5], [0.5, 0.5]])
    longitudes = np.array([[0.5, 1.5], [0.5, 1.5]])
    return latitudes, longitudes


class TestGeographicInterpolation:
    """Tests for coordinate-aware bilinear interpolation."""

    def test_bilinear_interpolation_matches_linear_field(self) -> None:
        """A linear field is reproduced exactly at target coordinates."""
        source_lat, source_lon = _source_coordinates()
        target_lat, target_lon = _target_coordinates()
        interpolation = GeographicInterpolation(
            source_latitudes=source_lat.ravel().tolist(),
            source_longitudes=source_lon.ravel().tolist(),
            source_shape=source_lat.shape,
            target_latitudes=target_lat.ravel().tolist(),
            target_longitudes=target_lon.ravel().tolist(),
            target_shape=target_lat.shape,
            source_crs="EPSG:4326",
        )
        values = (source_lat + source_lon) / 4.0
        inputs = torch.from_numpy(values).float().view(1, 1, 3, 3)

        result = interpolation(inputs)

        expected = torch.from_numpy((target_lat + target_lon) / 4.0).float()
        torch.testing.assert_close(result[0, 0], expected)

    def test_rejects_target_outside_source_domain(self) -> None:
        """Extrapolation outside the source grid is rejected."""
        source_lat, source_lon = _source_coordinates()
        target_lat = np.array([[3.0]])
        target_lon = np.array([[1.0]])

        with pytest.raises(ValueError, match="outside"):
            GeographicInterpolation(
                source_latitudes=source_lat.ravel().tolist(),
                source_longitudes=source_lon.ravel().tolist(),
                source_shape=source_lat.shape,
                target_latitudes=target_lat.ravel().tolist(),
                target_longitudes=target_lon.ravel().tolist(),
                target_shape=target_lat.shape,
                source_crs="EPSG:4326",
            )

    def test_rejects_wrong_input_shape(self) -> None:
        """Runtime tensors must match the grid used to build the mapping."""
        source_lat, source_lon = _source_coordinates()
        target_lat, target_lon = _target_coordinates()
        interpolation = GeographicInterpolation(
            source_latitudes=source_lat.ravel().tolist(),
            source_longitudes=source_lon.ravel().tolist(),
            source_shape=source_lat.shape,
            target_latitudes=target_lat.ravel().tolist(),
            target_longitudes=target_lon.ravel().tolist(),
            target_shape=target_lat.shape,
            source_crs="EPSG:4326",
        )

        with pytest.raises(ValueError, match="source shape"):
            interpolation(torch.zeros((1, 1, 4, 4)))
