import numpy as np
import pytest

from icenet_mp.geotools.reproject import nearest_neighbour_indices


class TestNearestNeighbourIndices:
    """Unit tests for the nearest_neighbour_indices function."""

    def test_raises_value_error_for_wrong_input_ndim(self) -> None:
        """Reject an input lat/lon array that is not 3-dimensional."""
        input_latlons = np.zeros((2, 2), dtype=np.float32)
        output_latlons = np.zeros((1, 1, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="Input lat/lons must have shape"):
            nearest_neighbour_indices(input_latlons, output_latlons)

    def test_raises_value_error_for_wrong_input_last_dimension(self) -> None:
        """Reject an input lat/lon array whose last dimension is not size 2."""
        input_latlons = np.zeros((2, 2, 3), dtype=np.float32)
        output_latlons = np.zeros((1, 1, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="Input lat/lons must have shape"):
            nearest_neighbour_indices(input_latlons, output_latlons)

    def test_raises_value_error_for_wrong_output_ndim(self) -> None:
        """Reject an output lat/lon array that is not 3-dimensional."""
        input_latlons = np.zeros((2, 2, 2), dtype=np.float32)
        output_latlons = np.zeros((1, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="Output lat/lons must have shape"):
            nearest_neighbour_indices(input_latlons, output_latlons)

    def test_raises_value_error_for_wrong_output_last_dimension(self) -> None:
        """Reject an output lat/lon array whose last dimension is not size 2."""
        input_latlons = np.zeros((2, 2, 2), dtype=np.float32)
        output_latlons = np.zeros((1, 1, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Output lat/lons must have shape"):
            nearest_neighbour_indices(input_latlons, output_latlons)

    def test_finds_correct_nearest_input_cell_for_each_output_point(self) -> None:
        """Each output point maps to the (h, w) index of its closest input cell."""
        input_latlons = np.array(
            [
                [[0.0, 0.0], [0.0, 10.0]],
                [[10.0, 0.0], [10.0, 10.0]],
            ],
            dtype=np.float32,
        )
        output_latlons = np.array(
            [[[0.5, 0.5], [9.5, 9.6]]],
            dtype=np.float32,
        )

        nn_indices_h, nn_indices_w = nearest_neighbour_indices(
            input_latlons, output_latlons
        )

        assert nn_indices_h.shape == (1, 2)
        assert nn_indices_w.shape == (1, 2)
        np.testing.assert_array_equal(nn_indices_h, [[0, 1]])
        np.testing.assert_array_equal(nn_indices_w, [[0, 1]])
