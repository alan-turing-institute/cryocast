from unittest.mock import MagicMock

import numpy as np
import pytest

from cryocast.ingestion.filters.nan_to_num_filter import NanToNumFilter

from .conftest import make_array_field


@pytest.fixture
def filter_instance() -> NanToNumFilter:
    """NanToNumFilter with two variables and replace_with=-1.0."""
    return NanToNumFilter(variables=["siconc", "sithick"], replace_with=-1.0)


class TestNanToNumFilter:
    """Test suite for NanToNumFilter class."""

    def test_forward_select_returns_param_list(
        self, filter_instance: NanToNumFilter
    ) -> None:
        """Ensure forward_select returns a param list matching the configured variables."""
        result = filter_instance.forward_select()
        assert result == {"param": ["siconc", "sithick"]}

    def test_forward_select_single_variable(self) -> None:
        """Ensure forward_select works correctly with a single variable."""
        f = NanToNumFilter(variables=["sithick"], replace_with=0.0)
        assert f.forward_select() == {"param": ["sithick"]}

    def test_forward_transform_replaces_nan(
        self, filter_instance: NanToNumFilter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """forward_transform replaces NaN values and delegates to new_field_from_numpy."""
        data = np.array([[1.0, np.nan], [np.nan, 2.0]])

        mock_field = MagicMock()
        mock_field.to_numpy.return_value = data
        mock_field.shape = data.shape

        mock_output_field = MagicMock()
        mock_nffn = MagicMock(return_value=mock_output_field)
        monkeypatch.setattr(filter_instance, "new_field_from_numpy", mock_nffn)

        result = filter_instance.forward_transform(mock_field)

        assert result is mock_output_field
        called_array = mock_nffn.call_args[0][0]
        assert not np.any(np.isnan(called_array)), "NaN values were not replaced"
        np.testing.assert_array_equal(
            called_array, np.array([[1.0, -1.0], [-1.0, 2.0]])
        )
        assert mock_nffn.call_args[1]["template"] is mock_field

    def test_forward_transform_no_nan_passthrough(
        self, filter_instance: NanToNumFilter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """forward_transform leaves non-NaN arrays unchanged."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_field = MagicMock()
        mock_field.to_numpy.return_value = data
        mock_field.shape = data.shape

        mock_nffn = MagicMock()
        monkeypatch.setattr(filter_instance, "new_field_from_numpy", mock_nffn)
        filter_instance.forward_transform(mock_field)

        called_array = mock_nffn.call_args[0][0]
        np.testing.assert_array_equal(called_array, data)

    def test_forward_transform_replaces_nan_with_real_field(
        self, filter_instance: NanToNumFilter
    ) -> None:
        """Test a real NanToNumFilter against a real ArrayField."""
        field = make_array_field(
            np.array([1.0, np.nan]), lats=[80.0, 85.0], lons=[0.0, 90.0]
        )

        result = filter_instance.forward_transform(field)

        np.testing.assert_array_equal(result.to_numpy(), [1.0, -1.0])
        assert result.metadata("param") == "siconc"
