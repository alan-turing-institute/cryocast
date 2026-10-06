import logging

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from icenet_mp.types import CF_VARIABLE_ATTRIBUTES

logger = logging.getLogger(__name__)

# Native CRS of the EASE-Grid 2.0 prediction grid for each hemisphere
EASE2_CRS = {"north": "EPSG:6931", "south": "EPSG:6932"}

# Name of the land mask (1 = ocean, 0 = land) written by the prediction writer, and
# of the regridded land mask in the output
LAND_MASK = "land_mask"

# Smallest interpolated ocean fraction for a target point to count as ocean
OCEAN_FRACTION_THRESHOLD = 0.5

# Largest allowed distance between a projected source point and the 1-D axes
# recovered from them, as a fraction of the grid spacing
REGULAR_GRID_TOLERANCE = 0.01

# Source variable attributes that refer to the source grid or file, so are not
# carried over to the regridded variable
SOURCE_ONLY_ATTRIBUTES = ("_FillValue", "ancillary_variables", "coordinates")


def regular_grid_axes(
    x: np.ndarray, y: np.ndarray, crs: str
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the 1-D axes of a grid that is regular in a projected CRS.

    Args:
        x: 2-D (y, x) array of projected x coordinates of each grid point.
        y: 2-D (y, x) array of projected y coordinates of each grid point.
        crs: CRS of the projected coordinates, used in the error message.

    Returns:
        Tuple of (x_axis, y_axis), the 1-D coordinates along each grid axis.

    Raises:
        ValueError: If the points do not lie on a regular grid in `crs`.

    """
    x_axis = x.mean(axis=0)
    y_axis = y.mean(axis=1)
    spacing = min(np.abs(np.diff(x_axis)).min(), np.abs(np.diff(y_axis)).min())
    deviation = max(
        np.abs(x - x_axis[None, :]).max(), np.abs(y - y_axis[:, None]).max()
    )
    # Written so that NaN coordinates also fail the check
    if not deviation <= REGULAR_GRID_TOLERANCE * spacing:
        msg = (
            f"Predictions are not on a regular grid in {crs}: grid points deviate by "
            f"up to {deviation:.3g} from the recovered axes, compared with a grid "
            f"spacing of {spacing:.3g}. Pass the CRS in which the grid is regular as "
            "source_crs."
        )
        raise ValueError(msg)
    return x_axis, y_axis


def regrid_forecast_run(  # noqa: PLR0913
    predictions: xr.Dataset,
    target_latitudes: xr.DataArray,
    target_longitudes: xr.DataArray,
    first_valid_day: str | np.datetime64,
    *,
    variable: str = "ice_conc",
    output_variable: str = "siconc",
    source_crs: str | None = None,
) -> xr.Dataset:
    """Regrid a single forecast run onto a regular lat/lon grid.

    Selects the forecast run whose first lead time is valid on `first_valid_day` and
    bilinearly interpolates it from its native grid onto the target lat/lon grid,
    cropping to the target extent. The interpolation is done in the CRS in which the
    native grid is regular, which is EASE-Grid 2.0 unless `source_crs` is given. The
    output follows the CMEMS layout, with dimensions (time, latitude, longitude) and
    a daily (00:00) time axis.

    If the predictions have a land mask, land cells are left out of the
    interpolation, so that they do not pull coastal values towards zero. Target
    points that are mostly land are set to NaN, and the land mask is regridded into
    the output.

    Args:
        predictions: Predictions with dimensions (forecast_reference_time, lead_time,
            y, x), 2-D latitude/longitude coordinates and a valid_time coordinate, as
            written by the prediction writer. Unless `source_crs` is given, it must
            also have a "hemisphere" attribute of "north" or "south". It may have a
            (y, x) "land_mask" variable, with 1 for ocean and 0 for land.
        target_latitudes: 1-D latitudes of the target grid.
        target_longitudes: 1-D longitudes of the target grid.
        first_valid_day: Day on which the first lead time of the selected run is
            valid, e.g. "2024-01-14".
        variable: Name of the variable to regrid in `predictions`.
        output_variable: Name of the regridded variable in the output.
        source_crs: CRS in which the prediction grid is regular, e.g. "EPSG:3413".
            Defaults to the EASE-Grid 2.0 CRS for the predictions' hemisphere.

    Returns:
        Dataset containing `output_variable` with dimensions (time, latitude,
        longitude), where time holds the valid day of each lead time. The variable
        keeps the attributes of `variable` in `predictions`, plus the CF attributes
        in `CF_VARIABLE_ATTRIBUTES` for known variables, and is clipped to its valid
        range where one is defined. If the predictions have a land mask, the
        output also contains a (latitude, longitude) "land_mask" (1 for ocean, 0
        for land, missing outside the prediction grid), linked from
        `output_variable` as an ancillary variable.

    Raises:
        ValueError: If the target latitudes or longitudes are not 1-D, if no run has
            its first lead time valid on `first_valid_day`, if `source_crs` is not
            given and the hemisphere attribute is missing or invalid, or if the
            prediction grid is not regular in the source CRS.

    """
    if target_latitudes.ndim != 1 or target_longitudes.ndim != 1:
        msg = (
            "target_latitudes and target_longitudes must be 1-D, but got shapes "
            f"{target_latitudes.shape} and {target_longitudes.shape}."
        )
        raise ValueError(msg)

    # Select the run whose first lead time is valid on first_valid_day
    first_valid = predictions["valid_time"].isel(lead_time=0).dt.floor("D")
    matches = np.flatnonzero(first_valid == pd.Timestamp(first_valid_day).normalize())
    if matches.size == 0:
        msg = f"No forecast run has its first lead time valid on {first_valid_day}."
        raise ValueError(msg)
    run = predictions.isel(forecast_reference_time=int(matches[0]))

    # Default to the EASE-Grid 2.0 CRS for the predictions' hemisphere
    if source_crs is None:
        hemisphere = predictions.attrs.get("hemisphere")
        if hemisphere not in EASE2_CRS:
            msg = f"Predictions must have a 'hemisphere' attribute of 'north' or 'south' unless source_crs is given, but got {hemisphere!r}."
            raise ValueError(msg)
        source_crs = EASE2_CRS[hemisphere]

    # Source is a regular grid in source_crs: recover its 1-D x/y coordinates
    to_source = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
    x_src, y_src = to_source.transform(
        run["longitude"].to_numpy(), run["latitude"].to_numpy()
    )
    x_axis, y_axis = regular_grid_axes(x_src, y_src, source_crs)

    # Project the target grid into the source CRS, for bilinear interpolation
    lon2d, lat2d = np.meshgrid(
        target_longitudes.to_numpy(), target_latitudes.to_numpy()
    )
    x_tgt, y_tgt = to_source.transform(lon2d, lat2d)

    def to_target(data: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate a (..., y, x) source array onto the target grid."""
        dims = (*(f"dim_{i}" for i in range(data.ndim - 2)), "y", "x")
        source = xr.DataArray(data, dims=dims, coords={"y": y_axis, "x": x_axis})
        return source.interp(
            x=xr.DataArray(x_tgt, dims=("latitude", "longitude")),
            y=xr.DataArray(y_tgt, dims=("latitude", "longitude")),
            method="linear",
        ).to_numpy()

    # Keep the source variable's metadata (e.g. its "predicted ..." long_name),
    # filling any gaps from the CF attributes for known variables, such as the valid
    # range missing from older prediction files
    attributes = CF_VARIABLE_ATTRIBUTES.get(variable, {}) | {
        key: value
        for key, value in run[variable].attrs.items()
        if key not in SOURCE_ONLY_ATTRIBUTES
    }

    values = run[variable].to_numpy()
    data_vars = {}
    if LAND_MASK in run:
        # Land cells hold no real value (masked models write zero there), so leave
        # them out of the interpolation by normalising by the interpolated ocean
        # fraction, rather than letting them pull coastal values towards zero
        ocean = run[LAND_MASK].to_numpy() == 1
        ocean_fraction = to_target(ocean.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            regridded = to_target(np.where(ocean, values, 0.0)) / ocean_fraction
        # Target points that are mostly land are land, and have no value. The mask
        # is NaN outside the source grid, where land and ocean are unknown
        is_ocean = ocean_fraction >= OCEAN_FRACTION_THRESHOLD
        regridded = np.where(is_ocean, regridded, np.nan)
        target_mask = np.where(np.isnan(ocean_fraction), np.nan, is_ocean)
        data_vars[LAND_MASK] = xr.Variable(
            ("latitude", "longitude"),
            target_mask.astype(np.float32),
            {
                key: value
                for key, value in run[LAND_MASK].attrs.items()
                if key not in SOURCE_ONLY_ATTRIBUTES
            },
            encoding={"dtype": "int8", "_FillValue": np.int8(-1)},
        )
        attributes["ancillary_variables"] = LAND_MASK
    else:
        logger.warning(
            "Predictions have no '%s' variable, so land cells cannot be excluded "
            "from the regridding and coastal values of '%s' may be biased.",
            LAND_MASK,
            variable,
        )
        regridded = to_target(values)

    # Bilinear interpolation does not guarantee values stay within the valid range,
    # so clip to it wherever one is defined (NaNs on land and outside the source grid
    # are kept)
    values = regridded.astype(np.float32)
    valid_min = attributes.get("valid_min")
    valid_max = attributes.get("valid_max")
    if valid_min is not None or valid_max is not None:
        values = np.clip(values, valid_min, valid_max)
    for key in ("valid_min", "valid_max"):
        if key in attributes:
            attributes[key] = np.float32(attributes[key])

    # Build a dataset in the CMEMS layout, with a daily-mean style (00:00) time axis
    output = xr.Dataset(
        {
            output_variable: (("time", "latitude", "longitude"), values, attributes),
            **data_vars,
        },
        coords={
            "time": (
                "time",
                pd.DatetimeIndex(run["valid_time"].to_numpy()).floor("D"),
                {"axis": "T", "standard_name": "time", "long_name": "Time"},
            ),
            # Use the target values on the CMEMS dimension names, whatever the
            # dimensions of the target coordinates are called
            "latitude": (
                "latitude",
                target_latitudes.to_numpy(),
                target_latitudes.attrs,
            ),
            "longitude": (
                "longitude",
                target_longitudes.to_numpy(),
                target_longitudes.attrs,
            ),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"IceNet-MP regridded forecast of {variable}",
            "source": "IceNet-MP",
            "forecast_reference_time": str(run["forecast_reference_time"].to_numpy()),
            "history": (
                f"Bilinearly regridded from a regular {source_crs} grid onto a "
                "regular lat/lon grid"
                + (", excluding land" if LAND_MASK in data_vars else "")
            ),
        },
    )
    output["time"].encoding.update(units="hours since 1950-01-01", calendar="standard")
    return output
