import re
from collections.abc import Callable
from typing import Any

import numpy as np

from .geographic_grid import GeographicGrid


class GridFactory:
    def __init__(self) -> None:
        """Factory for creating GeographicGrid instances from CRS and resolution parameters."""
        self.builders: dict[str, Callable[..., GeographicGrid]] = {}  # type: ignore[annotation-unchecked]

    def create(self, crs: str, **kwargs: Any) -> GeographicGrid:
        if not (builder := self.builders.get(crs)):
            msg = f"No builder registered for CRS: {crs}"
            raise ValueError(msg)
        return builder(**kwargs)

    def register_crs(self, crs: str, builder_fn: Callable[..., GeographicGrid]) -> None:
        self.builders[crs] = builder_fn


def epsg_4326n_builder(resolution: str, shape: tuple[int, int]) -> GeographicGrid:
    """Builder function to create a GeographicGrid for the northern hemisphere in WGS (EPSG:4326)."""
    res_deg = float(resolution.replace("p", "."))
    lat_points = np.arange(90 + 0.5 * res_deg - shape[0], 90, res_deg)
    lon_points = np.arange(0.5 * res_deg - shape[1] / 2, shape[1] / 2, res_deg)
    return GeographicGrid("EPSG:4326", str(res_deg), lon_points, lat_points)


def epsg_4326s_builder(resolution: str, shape: tuple[int, int]) -> GeographicGrid:
    """Builder function to create a GeographicGrid for the southern hemisphere in WGS (EPSG:4326)."""
    res_deg = float(resolution.replace("p", "."))
    lat_points = np.arange(0.5 * res_deg - shape[0], 0, res_deg)
    lon_points = np.arange(0.5 * res_deg - shape[1] / 2, shape[1] / 2, res_deg)
    return GeographicGrid("EPSG:4326", str(res_deg), lon_points, lat_points)


def epsg_6931_builder(resolution: str, shape: tuple[int, int]) -> GeographicGrid:
    """Builder function to create a GeographicGrid for the EASE2 northern polar grid (EPSG:6931)."""
    normalised_resolution, h_points, w_points = ease2_grid_helper(resolution, *shape)
    return GeographicGrid("EPSG:6931", normalised_resolution, h_points, w_points)


def epsg_6932_builder(resolution: str, shape: tuple[int, int]) -> GeographicGrid:
    """Builder function to create a GeographicGrid for the EASE2 southern polar grid (EPSG:6932)."""
    normalised_resolution, h_points, w_points = ease2_grid_helper(resolution, *shape)
    return GeographicGrid("EPSG:6932", normalised_resolution, h_points, w_points)


def ease2_grid_helper(
    resolution: str, h_size: int, w_size: int
) -> tuple[str, np.ndarray[tuple[int]], np.ndarray[tuple[int]]]:
    """Helper function to generate the grid points for an EASE2 grid given the resolution and shape."""
    # Normalise the resolution
    if (match := re.match(r"^([0-9p]+)([^0-9]+)$", resolution)) is None:
        msg = f"Invalid resolution format: {resolution}"
        raise ValueError(msg)
    scale = float(match.group(1).replace("p", "."))
    unit = match.group(2)
    scale_m = scale * (1000 if unit in ("k", "km") else 1)
    normalised_resolution = str(scale_m / 1000).replace(".", "p") + "km"
    # Get grid positions in EPSG space
    h_lim = scale_m * ((h_size - 1) / 2 if h_size % 2 == 0 else h_size // 2)
    w_lim = scale_m * ((w_size - 1) / 2 if w_size % 2 == 0 else w_size // 2)
    h_points = np.linspace(-h_lim, h_lim, h_size)
    w_points = np.linspace(w_lim, -w_lim, w_size)
    return (normalised_resolution, h_points, w_points)
