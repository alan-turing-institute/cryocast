import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer

from icenet_mp.geotools import regrid_forecast_run

# Small EASE-Grid 2.0 North grid centred on the Greenland Sea
X_POINTS = np.arange(-1_000_000.0, -199_999.0, 25_000.0)
Y_POINTS = np.arange(-1_700_000.0, -2_600_001.0, -25_000.0)

# Small NSIDC Sea Ice Polar Stereographic North (EPSG:3413) grid covering the same area
X_POINTS_3413 = np.arange(700_000.0, 1_500_001.0, 25_000.0)
Y_POINTS_3413 = np.arange(-1_500_000.0, -2_300_001.0, -25_000.0)


def linear_in_x(x: np.ndarray, x_points: np.ndarray, lead: int, run: int) -> np.ndarray:
    """Field that varies linearly in projected x, so bilinear interpolation is exact.

    Values stay within [0.1, 0.9] across the grid, so are not clipped to the valid
    sea ice concentration range.
    """
    fraction = (x - x_points[0]) / (x_points[-1] - x_points[0])
    return 0.1 * (1 + lead + run) + 0.5 * fraction


def make_predictions(
    crs: str, x_points: np.ndarray, y_points: np.ndarray, attrs: dict[str, str]
) -> xr.Dataset:
    """Predictions for three daily runs with two lead times on a regular grid in crs."""
    x2d, y2d = np.meshgrid(x_points, y_points)
    to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon2d, lat2d = to_latlon.transform(x2d, y2d)
    reference_times = pd.date_range("2024-01-12T12:00", periods=3, freq="D")
    lead_times = pd.to_timedelta([1, 2], unit="D")
    ice_conc = np.stack(
        [
            np.stack(
                [
                    linear_in_x(x2d, x_points, lead, run)
                    for lead in range(len(lead_times))
                ]
            )
            for run in range(len(reference_times))
        ]
    ).astype(np.float32)
    return xr.Dataset(
        {
            "ice_conc": (
                ("forecast_reference_time", "lead_time", "y", "x"),
                ice_conc,
                # As written by the prediction writer
                {
                    "standard_name": "sea_ice_area_fraction",
                    "long_name": "predicted sea ice concentration",
                    "units": "1",
                    "ancillary_variables": "land_mask",
                },
            )
        },
        coords={
            "forecast_reference_time": reference_times,
            "lead_time": lead_times,
            "valid_time": (
                ("forecast_reference_time", "lead_time"),
                reference_times.to_numpy()[:, None] + lead_times.to_numpy()[None, :],
            ),
            "latitude": (("y", "x"), lat2d.astype(np.float32)),
            "longitude": (("y", "x"), lon2d.astype(np.float32)),
        },
        attrs=attrs,
    )


@pytest.fixture
def predictions() -> xr.Dataset:
    """Predictions on an EASE2 North grid."""
    return make_predictions("EPSG:6931", X_POINTS, Y_POINTS, {"hemisphere": "north"})


@pytest.fixture
def predictions_3413() -> xr.Dataset:
    """Predictions on a polar stereographic grid, with no hemisphere attribute."""
    return make_predictions("EPSG:3413", X_POINTS_3413, Y_POINTS_3413, {})


@pytest.fixture
def target_grid() -> tuple[xr.DataArray, xr.DataArray]:
    """A 1 degree lat/lon grid lying inside the prediction grid."""
    latitudes = xr.DataArray(
        np.arange(68.0, 72.1, 1.0, dtype=np.float32), dims="latitude"
    )
    longitudes = xr.DataArray(
        np.arange(-20.0, -9.9, 1.0, dtype=np.float32), dims="longitude"
    )
    return latitudes, longitudes


class TestRegridForecastRun:
    """Unit tests for the regrid_forecast_run function."""

    def test_output_has_cmems_layout(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Output has (time, latitude, longitude) dims on the target coordinates."""
        latitudes, longitudes = target_grid
        output = regrid_forecast_run(predictions, latitudes, longitudes, "2024-01-14")

        assert output["siconc"].dims == ("time", "latitude", "longitude")
        assert output["siconc"].dtype == np.float32
        np.testing.assert_array_equal(output["latitude"], latitudes)
        np.testing.assert_array_equal(output["longitude"], longitudes)

    def test_output_uses_cmems_dims_for_other_target_dim_names(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Target coordinates on dims named lat/lon give a (time, latitude, longitude) output."""
        latitudes, longitudes = target_grid
        latitudes = latitudes.rename(latitude="lat").assign_attrs(units="degrees_north")
        longitudes = longitudes.rename(longitude="lon")
        output = regrid_forecast_run(predictions, latitudes, longitudes, "2024-01-14")

        assert set(output.sizes) == {"time", "latitude", "longitude"}
        assert output["siconc"].dims == ("time", "latitude", "longitude")
        np.testing.assert_array_equal(output["latitude"], latitudes)
        np.testing.assert_array_equal(output["longitude"], longitudes)
        assert output["latitude"].attrs["units"] == "degrees_north"

    def test_raises_value_error_for_2d_target_coordinates(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Reject target latitudes or longitudes that are not 1-D."""
        latitudes, longitudes = target_grid
        lat2d, _ = xr.broadcast(latitudes, longitudes)
        with pytest.raises(ValueError, match="must be 1-D"):
            regrid_forecast_run(predictions, lat2d, longitudes, "2024-01-14")

    def test_selects_run_by_first_valid_day(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """The run issued at 2024-01-13T12 is selected, with times floored to 00:00."""
        output = regrid_forecast_run(predictions, *target_grid, "2024-01-14")

        np.testing.assert_array_equal(
            output["time"].to_numpy(),
            np.array(["2024-01-14", "2024-01-15"], dtype="datetime64[ns]"),
        )
        assert output.attrs["forecast_reference_time"].startswith("2024-01-13T12")

    def test_interpolated_values_match_linear_field(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Bilinear interpolation reproduces a field that is linear in EASE x."""
        latitudes, longitudes = target_grid
        output = regrid_forecast_run(predictions, latitudes, longitudes, "2024-01-14")

        lon2d, lat2d = np.meshgrid(longitudes.to_numpy(), latitudes.to_numpy())
        to_ease = Transformer.from_crs("EPSG:4326", "EPSG:6931", always_xy=True)
        x_tgt, _ = to_ease.transform(lon2d, lat2d)
        run = 1  # run issued at 2024-01-13T12
        for lead in range(2):
            np.testing.assert_allclose(
                output["siconc"].isel(time=lead),
                linear_in_x(x_tgt, X_POINTS, lead, run),
                atol=1e-4,
            )

    def test_interpolates_in_given_source_crs(
        self,
        predictions_3413: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """A grid in another CRS is regridded exactly when source_crs is given."""
        latitudes, longitudes = target_grid
        output = regrid_forecast_run(
            predictions_3413,
            latitudes,
            longitudes,
            "2024-01-14",
            source_crs="EPSG:3413",
        )

        lon2d, lat2d = np.meshgrid(longitudes.to_numpy(), latitudes.to_numpy())
        to_source = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
        x_tgt, _ = to_source.transform(lon2d, lat2d)
        run = 1  # run issued at 2024-01-13T12
        for lead in range(2):
            np.testing.assert_allclose(
                output["siconc"].isel(time=lead),
                linear_in_x(x_tgt, X_POINTS_3413, lead, run),
                atol=1e-4,
            )
        assert "EPSG:3413" in output.attrs["history"]

    def test_raises_value_error_for_irregular_grid(
        self,
        predictions_3413: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Reject a grid that is not regular in the default EASE2 CRS."""
        predictions_3413.attrs["hemisphere"] = "north"
        with pytest.raises(ValueError, match="not on a regular grid in EPSG:6931"):
            regrid_forecast_run(predictions_3413, *target_grid, "2024-01-14")

    def test_ice_conc_attributes(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Sea ice concentration keeps the writer's metadata plus its CF valid range."""
        output = regrid_forecast_run(predictions, *target_grid, "2024-01-14")

        attrs = output["siconc"].attrs
        assert attrs["standard_name"] == "sea_ice_area_fraction"
        assert attrs["long_name"] == "predicted sea ice concentration"
        assert attrs["units"] == "1"
        assert attrs["valid_min"] == np.float32(0.0)
        assert attrs["valid_max"] == np.float32(1.0)
        assert "ancillary_variables" not in attrs
        assert output.attrs["title"] == "IceNet-MP regridded forecast of ice_conc"

    def test_other_variable_keeps_its_own_attributes(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """A variable without CF attributes is not labelled or clipped as sea ice."""
        predictions["sst"] = (predictions["ice_conc"] + 270.0).assign_attrs(
            long_name="predicted sea surface temperature", units="K"
        )
        output = regrid_forecast_run(
            predictions,
            *target_grid,
            "2024-01-14",
            variable="sst",
            output_variable="tos",
        )

        assert output["tos"].attrs == {
            "long_name": "predicted sea surface temperature",
            "units": "K",
        }
        assert (output["tos"] > 270.0).all()

    def test_values_are_clipped_to_valid_range(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Values outside the valid sea ice concentration range are clipped to it."""
        predictions["ice_conc"] = predictions["ice_conc"] * 4 - 1.5
        output = regrid_forecast_run(predictions, *target_grid, "2024-01-14")

        values = output["siconc"].to_numpy()
        assert values.min() == 0.0
        assert values.max() == 1.0

    def test_points_outside_source_grid_are_nan(self, predictions: xr.Dataset) -> None:
        """Target points outside the prediction grid are filled with NaN."""
        latitudes = xr.DataArray(np.array([40.0], dtype=np.float32), dims="latitude")
        longitudes = xr.DataArray(np.array([0.0], dtype=np.float32), dims="longitude")
        output = regrid_forecast_run(predictions, latitudes, longitudes, "2024-01-14")

        assert np.isnan(output["siconc"].to_numpy()).all()

    def test_raises_value_error_for_missing_day(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Reject a day for which no run has its first lead time valid."""
        with pytest.raises(ValueError, match="No forecast run"):
            regrid_forecast_run(predictions, *target_grid, "2024-02-01")

    def test_raises_value_error_for_missing_hemisphere(
        self,
        predictions: xr.Dataset,
        target_grid: tuple[xr.DataArray, xr.DataArray],
    ) -> None:
        """Reject predictions without a valid hemisphere attribute."""
        predictions.attrs = {}
        with pytest.raises(ValueError, match="hemisphere"):
            regrid_forecast_run(predictions, *target_grid, "2024-01-14")
