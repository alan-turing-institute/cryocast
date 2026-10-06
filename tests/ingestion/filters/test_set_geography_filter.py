from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from earthkit.data import SimpleFieldList

from cryocast.ingestion.filters import SetGeographyFilter

from .conftest import make_array_field


@pytest.fixture
def mock_geo_filter(monkeypatch: pytest.MonkeyPatch) -> SetGeographyFilter:
    """Patch grid_factory and return a SetGeographyFilter instance."""
    factory = MagicMock()
    factory.create.side_effect = lambda *_args, **_kwargs: MagicMock()
    monkeypatch.setattr(
        "cryocast.ingestion.filters.set_geography_filter.grid_factory",
        factory,
    )
    return SetGeographyFilter(crs="epsg:6931", resolution="25km")


class TestSetGeographyFilter:
    def test_forward_raises_type_error_for_non_fieldlist(
        self, mock_geo_filter: SetGeographyFilter
    ) -> None:
        """forward() raises TypeError when passed a non-FieldList."""
        with pytest.raises(TypeError, match="FieldList"):
            mock_geo_filter.forward(pd.DataFrame())

    def test_cached_geography_same_shape_returns_same_object(
        self, mock_geo_filter: SetGeographyFilter
    ) -> None:
        """The same shape returns the cached geography object."""
        geo1 = mock_geo_filter.cached_geography((432, 432))
        geo2 = mock_geo_filter.cached_geography((432, 432))
        assert geo1 is geo2

    def test_cached_geography_different_shapes_computed_separately(
        self, mock_geo_filter: SetGeographyFilter
    ) -> None:
        """Different shapes yield distinct geography objects."""
        geo_a = mock_geo_filter.cached_geography((432, 432))
        geo_b = mock_geo_filter.cached_geography((100, 100))
        assert geo_a is not geo_b

    @pytest.mark.usefixtures("mock_geo_filter")
    def test_geography_cache_is_per_instance(self) -> None:
        """Two SetGeographyFilter instances have independent geography caches."""
        f1 = SetGeographyFilter(crs="epsg:6931", resolution="25km")
        f2 = SetGeographyFilter(crs="epsg:6931", resolution="25km")
        geo1 = f1.cached_geography((432, 432))
        geo2 = f2.cached_geography((432, 432))
        assert geo1 is not geo2

    def test_forward_wraps_real_fieldlist_with_geography(self) -> None:
        """Test a real SetGeographyFilter against a real SimpleFieldList."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        field = make_array_field(
            values, lats=[80.0, 80.0, 85.0, 85.0], lons=[0.0, 90.0, 0.0, 90.0]
        )
        data = SimpleFieldList([field])

        filter_instance = SetGeographyFilter(crs="EPSG:6931", resolution="500km")
        result = filter_instance.forward(data)

        assert len(result) == 1
        output = next(iter(result))
        assert output.shape == (2, 2)
        assert output.metadata("param") == "siconc"
        np.testing.assert_array_equal(output.to_numpy(), values)
