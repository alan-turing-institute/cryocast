import logging
import time
from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import xarray as xr
from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data import FieldList
from haversine import Unit, haversine_vector
from pandas import DataFrame
from typing_extensions import override

from cryocast.geotools import grid_factory
from cryocast.ingestion.sources import lazy_argopy as argopy

logger = logging.getLogger(__name__)


@source_registry.register("argo")
class ArgoSource(Source):
    missing_dates: ClassVar[set[datetime]] = set()

    def __init__(  # noqa: PLR0913
        self,
        context: Context,
        *,
        area: str,
        crs: str,
        param: list[str],
        resolution: str,
        shape: tuple[int, int],
        distance_scale_km: float = 2000,
        ignore_missing_dates: bool = False,
        min_weight: float = 1e-10,
        time_half_window_hrs: int = 2,
    ) -> None:
        """Set parameters for fetching and interpolating Argo float data."""
        self.context = context
        self.param = param

        # Set the output geography
        self.output_geography = grid_factory.create(
            crs, resolution=resolution, shape=shape
        )

        # Set weighting parameters for interpolation
        self.distance_scale_km = distance_scale_km
        self.ignore_missing_dates = ignore_missing_dates
        self.min_weight = min_weight

        # Set and verify the window parameters
        self.time_half_window_hrs = time_half_window_hrs
        self.north, self.west, self.south, self.east = map(float, area.split("/"))
        if (self.north < self.south) or (self.east < self.west):
            msg = f"Invalid area: {area}. Expected format: 'N/W/S/E'"
            raise ValueError(msg)

    @override
    def execute(self, argument: list[datetime] | GroupOfDates) -> FieldList:
        """Download Argo float data within given date range."""
        # Construct the grid that we want to project onto
        grid_points = np.stack(
            (self.output_geography.latitudes(), self.output_geography.longitudes()),
            axis=-1,
        )  # shape: (n_lat, n_lon, 2)
        grid_shape = grid_points.shape[0:2]
        grid_points_flat = grid_points.reshape(-1, 2)  # shape: (n_lat*n_lon, 2)
        logger.info("Created grid with %d latitudes and %d longitudes.", *grid_shape)

        # Create a dummy data structure of the correct shape
        requested_dates: list[datetime] = sorted(date for date in argument)

        weighted_data: dict[str, np.ndarray] = {
            variable: np.full((len(requested_dates), *grid_shape), np.nan, dtype=float)
            for variable in self.param
        }
        logger.info(
            "Fetching Argo data inside [N: %s, W: %s, S: %s, E: %s] between %s and %s.",
            self.north,
            self.west,
            self.south,
            self.east,
            min(requested_dates),
            max(requested_dates),
        )

        # Iterate over dates, requesting data for each one separately
        for t_idx, date in enumerate(requested_dates):
            logger.info("Requesting data for %s", date.date())
            start_time = date - timedelta(hours=self.time_half_window_hrs)
            end_time = date + timedelta(hours=self.time_half_window_hrs)

            # Extract data for the mixed layer (top 50m), retry if 503 error from erddap
            try:
                region = [self.west, self.east, self.south, self.north, 0, 50]
                time_window = [start_time, end_time]
                df = _fetch_argo_dataframe_with_retry(region, time_window)
            except LookupError:
                if self.ignore_missing_dates:
                    ArgoSource.missing_dates.add(date)
                    continue
                raise

            # Get positions of observations
            obs_lat = df["LATITUDE"].to_numpy(dtype=float)
            obs_lon = df["LONGITUDE"].to_numpy(dtype=float)
            obs_points = np.column_stack((obs_lat, obs_lon))  # shape: (n_obs, 2)

            # Pairwise distances in km
            distance_km: np.ndarray = haversine_vector(
                obs_points,
                grid_points_flat,
                unit=Unit.KILOMETERS,
                comb=True,
                check=False,  # faster if your lat/lon are already valid
            )  # shape: (n_lat*n_lon, n_obs)

            # Construct exponential weights, with a minimum for numerical stability
            weights = np.exp(-0.5 * (distance_km / self.distance_scale_km) ** 2).clip(
                min=self.min_weight
            )  # shape: (n_lat*n_lon, n_obs)
            sum_weights = np.sum(weights, axis=1)  # shape: (n_lat*n_lon,)

            # Apply weights to the data
            for variable, weighted_array in weighted_data.items():
                unweighted_data = df[variable].to_numpy(dtype=float)
                weighted_array[t_idx] = (
                    np.matmul(weights, unweighted_data) / sum_weights
                ).reshape(*grid_shape)  # shape: (n_lat, n_lon)

        if self.ignore_missing_dates:
            logger.info(
                "Identified %d missing dates:",
                len(ArgoSource.missing_dates),
            )
            for missing_date in sorted(ArgoSource.missing_dates):
                logger.warning(missing_date.isoformat())

        # Construct an xarray dataset
        ds_out = xr.Dataset(
            data_vars={
                variable: (
                    ("time", "x_pos", "y_pos"),
                    data_array,
                )
                for variable, data_array in weighted_data.items()
            },
            coords={
                "time": (
                    "time",
                    np.asarray(requested_dates, dtype="datetime64[ns]"),
                    {
                        "standard_name": "time",
                        "units": "seconds since 1970-01-01 00:00:00",
                        "calendar": "standard",
                    },
                ),
                "lat": (
                    ("x_pos", "y_pos"),
                    self.output_geography.latitudes(),
                    {
                        "standard_name": "latitude",
                        "units": "degrees_north",
                    },
                ),
                "lon": (
                    ("x_pos", "y_pos"),
                    self.output_geography.longitudes(),
                    {
                        "standard_name": "longitude",
                        "units": "degrees_east",
                    },
                ),
            },
        )

        return load_one(
            "📂", self.context, [date.isoformat() for date in requested_dates], ds_out
        )


def _fetch_argo_dataframe_with_retry(
    region: list[float],
    time_window: list[datetime],
    max_attempts: int = 10,
    initial_backoff_s: float = 0.5,
) -> DataFrame:
    """Fetch Argo data with exponential backoff.

    Args:
        region: List of [west, east, south, north, depth_min, depth_max]
        time_window: List of [start_time, end_time]
        max_attempts: Maximum number of attempts to make when ERDDAP download fails.
        initial_backoff_s: Initial backoff delay in seconds before retrying.

    Returns:
        DataFrame from Argo data

    Raises:
        LookupError if data cannot be retrieved.

    """
    data_target = (
        f"Argo float data for {region} between {time_window[0].isoformat()} and "
        f"{time_window[1].isoformat()}"
    )

    # Download from ERDDAP with exponential backoff
    for attempt in range(1, max_attempts + 1):
        try:
            return argopy.DataFetcher().region(region + time_window).to_dataframe()
        except (FileNotFoundError, TimeoutError) as exc:
            # Annoyingly, both 50x errors and 404 errors raise a FileNotFoundError and
            # the error message does not contain the HTTP status code so we need to
            # retry both cases.
            logger.warning(
                "Downloading %s attempt %d/%d failed.",
                data_target,
                attempt,
                max_attempts,
            )
            if attempt >= max_attempts:
                msg = f"{data_target} is unavailable."
                raise LookupError(msg) from exc
            backoff = initial_backoff_s * (2 ** (attempt - 1))
            logger.info("Retrying download in %.1fs.", backoff)
            time.sleep(backoff)
        except argopy.NoData as exc:
            msg = f"{data_target} is unavailable."
            logger.warning(msg)
            raise LookupError(msg) from exc

    # Raise an error if all retries have failed
    msg = f"{data_target} could not be downloaded after {max_attempts} attempts."
    raise LookupError(msg)
