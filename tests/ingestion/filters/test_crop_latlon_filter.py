"""Tests for latitude/longitude region cropping."""

from typing import Any
from unittest.mock import MagicMock

import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr
from anemoi.datasets.create.sources.xarray import load_one

from cryocast.ingestion.filters.crop_latlon_filter import CropLatLonFilter


def _field_list(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> ekd.FieldList:
    """Build one tiny gridded field with explicit latitude/longitude coordinates."""
    height, width = latitudes.shape
    data = np.arange(height * width, dtype=float).reshape(1, height, width)
    dataset = xr.Dataset(
        data_vars={"ci": (("time", "y", "x"), data)},
        coords={
            "time": (
                "time",
                np.array([np.datetime64("2024-01-15T12:00:00")]),
                {"standard_name": "time"},
            ),
            "lat": (
                ("y", "x"),
                latitudes,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ("y", "x"),
                longitudes,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )
    return load_one(
        "test",
        MagicMock(),
        ["2024-01-15T12:00:00"],
        dataset,
    )


class TestCropLatLonFilter:
    """Test structured-grid cropping to a geographic region."""

    def test_crops_to_smallest_native_grid_box_covering_roi(self) -> None:
        """Only rows/columns needed to cover the ROI are retained."""
        latitudes = np.array(
            [[82.0] * 4, [80.0] * 4, [78.0] * 4, [76.0] * 4, [74.0] * 4]
        )
        longitudes = np.array([[10.0, 20.0, 30.0, 40.0]] * 5)
        result = CropLatLonFilter(
            north=81.0,
            west=15.0,
            south=76.0,
            east=35.0,
            resolution="2p5km",
        ).forward(_field_list(latitudes, longitudes))

        field = result[0]
        assert field.shape == (3, 2)
        assert field.metadata().geography.resolution() == "2p5km"
        np.testing.assert_array_equal(
            field.to_numpy(), np.array([[5.0, 6.0], [9.0, 10.0], [13.0, 14.0]])
        )
        cropped_latitudes, cropped_longitudes = field.grid_points()
        np.testing.assert_array_equal(
            cropped_latitudes.reshape(field.shape), latitudes[1:4, 1:3]
        )
        np.testing.assert_array_equal(
            cropped_longitudes.reshape(field.shape), longitudes[1:4, 1:3]
        )

    def test_wraps_zero_to_360_longitudes_for_roi_selection(self) -> None:
        """Western-hemisphere boxes work with grids encoded as 0..360 degrees."""
        latitudes = np.array([[80.0, 80.0, 80.0], [78.0, 78.0, 78.0]])
        longitudes = np.array([[330.0, 340.0, 350.0], [330.0, 340.0, 350.0]])
        result = CropLatLonFilter(
            north=81.0, west=-25.0, south=77.0, east=-5.0
        ).forward(_field_list(latitudes, longitudes))

        field = result[0]
        assert field.shape == (2, 2)
        _, cropped_longitudes = field.grid_points()
        np.testing.assert_array_equal(
            cropped_longitudes.reshape(field.shape),
            np.array([[-20.0, -10.0], [-20.0, -10.0]]),
        )

    def test_supports_roi_crossing_antimeridian(self) -> None:
        """West greater than east denotes an antimeridian-crossing region."""
        latitudes = np.array([[80.0] * 4, [78.0] * 4])
        longitudes = np.array([[160.0, 175.0, 185.0, 200.0]] * 2)
        result = CropLatLonFilter(
            north=81.0, west=170.0, south=77.0, east=-170.0
        ).forward(_field_list(latitudes, longitudes))

        field = result[0]
        assert field.shape == (2, 2)
        _, cropped_longitudes = field.grid_points()
        np.testing.assert_array_equal(
            cropped_longitudes.reshape(field.shape),
            np.array([[175.0, -175.0], [175.0, -175.0]]),
        )

    def test_rejects_roi_with_no_grid_points(self) -> None:
        """An ROI outside the source grid fails clearly."""
        fields = _field_list(
            np.array([[80.0, 80.0], [79.0, 79.0]]),
            np.array([[10.0, 20.0], [10.0, 20.0]]),
        )
        with pytest.raises(ValueError, match="contains no grid points"):
            CropLatLonFilter(north=60.0, west=10.0, south=50.0, east=20.0).forward(
                fields
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"north": 75.0, "west": 0.0, "south": 80.0, "east": 10.0}, "latitude"),
            ({"north": 80.0, "west": -181.0, "south": 75.0, "east": 10.0}, "Longitude"),
            ({"north": 80.0, "west": 0.0, "south": 75.0, "east": 181.0}, "Longitude"),
        ],
    )
    def test_rejects_invalid_bounds(self, kwargs: dict[str, Any], message: str) -> None:
        """Invalid geographic bounds fail at construction."""
        with pytest.raises(ValueError, match=message):
            CropLatLonFilter(**kwargs)

    def test_rejects_non_field_list(self) -> None:
        """The filter does not silently accept unsupported tabular input."""
        with pytest.raises(TypeError, match="Expected data to be a FieldList"):
            CropLatLonFilter(north=81.0, west=15.0, south=76.0, east=35.0).forward(
                MagicMock()
            )
