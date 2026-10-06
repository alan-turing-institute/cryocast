from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from earthkit.data import SimpleFieldList

from cryocast.ingestion.filters.reproject_filter import ReprojectFilter

from .conftest import make_array_field


class MockField:
    """Minimal stand-in for earthkit.data.Field."""

    shape = (2, 2)

    def grid_points(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-85.0, -80.0, -75.0, -70.0]),
            np.array([0.0, 10.0, 20.0, 30.0]),
        )


@pytest.fixture
def mock_reproject_filter(monkeypatch: pytest.MonkeyPatch) -> ReprojectFilter:
    """Patch grid_factory and return a ReprojectFilter instance."""
    mock_geo = MagicMock()
    mock_geo.latitudes.return_value = np.array([[0.0, 5.0], [10.0, 15.0]])
    mock_geo.longitudes.return_value = np.array([[0.0, 5.0], [10.0, 15.0]])
    monkeypatch.setattr(
        "cryocast.ingestion.filters.reproject_filter.grid_factory",
        MagicMock(create=MagicMock(return_value=mock_geo)),
    )
    return ReprojectFilter(crs="epsg:6931", resolution="25km", shape=(2, 2))


@pytest.fixture
def mock_nearest_neighbour_indices(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch Field with MockField and nearest_neighbour_indices with a dummy 2x2 stub."""
    monkeypatch.setattr(
        "cryocast.ingestion.filters.reproject_filter.Field",
        MockField,
    )
    mock_nn = MagicMock(
        return_value=(np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int))
    )
    monkeypatch.setattr(
        "cryocast.ingestion.filters.reproject_filter.nearest_neighbour_indices",
        mock_nn,
    )
    return mock_nn


class TestReprojectFilter:
    def setup_method(self) -> None:
        ReprojectFilter.nn_indices_cached.clear()

    def teardown_method(self) -> None:
        ReprojectFilter.nn_indices_cached.clear()

    def test_forward_raises_type_error_for_non_fieldlist(
        self, mock_reproject_filter: ReprojectFilter
    ) -> None:
        """forward() raises TypeError when passed a non-FieldList."""
        with pytest.raises(TypeError, match="FieldList"):
            mock_reproject_filter.forward(pd.DataFrame())

    def test_nearest_neighbours_raises_value_error_when_no_field_found(
        self, mock_reproject_filter: ReprojectFilter
    ) -> None:
        """nearest_neighbours() raises ValueError when no earthkit Field is present."""
        with pytest.raises(ValueError, match="No latitudes"):
            mock_reproject_filter.nearest_neighbours([])

    @pytest.mark.usefixtures("mock_nearest_neighbour_indices")
    def test_nn_indices_cached_after_first_call(
        self, mock_reproject_filter: ReprojectFilter
    ) -> None:
        """Class-level nn_indices_cached is populated after nearest_neighbours() runs."""
        mock_reproject_filter.nearest_neighbours([MockField()])
        assert len(ReprojectFilter.nn_indices_cached) == 1

    def test_nn_indices_cache_hit_avoids_recomputation(
        self,
        mock_reproject_filter: ReprojectFilter,
        mock_nearest_neighbour_indices: MagicMock,
    ) -> None:
        """nearest_neighbour_indices() is only called once for the same input shape."""
        mock_reproject_filter.nearest_neighbours([MockField()])
        mock_reproject_filter.nearest_neighbours([MockField()])
        assert mock_nearest_neighbour_indices.call_count == 1

    def test_forward_reprojects_real_fieldlist(self) -> None:
        """Test a real ReprojectFilter against a real SimpleFieldList."""
        field = make_array_field(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            lats=[80.0, 80.0, 85.0, 85.0],
            lons=[0.0, 90.0, 0.0, 90.0],
        )
        data = SimpleFieldList([field])

        filter_instance = ReprojectFilter(
            crs="EPSG:6931", resolution="500km", shape=(2, 2)
        )
        result = filter_instance.forward(data)

        assert len(result) == 1
        output = next(iter(result))
        assert output.to_numpy().shape == (2, 2)
        assert output.metadata("param") == "siconc"
