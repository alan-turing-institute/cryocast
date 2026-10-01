import warnings
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from icenet_mp.types import ArrayHW, ArrayTHW, Hemisphere, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask

# Suppress Matplotlib animation warning during tests; we intentionally do not keep
# long-lived references to animation objects beyond saving to buffer.
warnings.filterwarnings(
    "ignore",
    message="Animation was deleted without rendering anything",
    category=UserWarning,
)

TEST_DATE = date(2020, 1, 15)
TEST_HEIGHT = 16
TEST_WIDTH = 16
N_TIMESTEPS = 2


@pytest.fixture(autouse=True)
def close_all_figures() -> Iterator[None]:
    """Automatically close all matplotlib figures after each test to prevent warnings."""
    yield
    plt.close("all")


@pytest.fixture
def sic_pair_2d(
    sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
) -> tuple[ArrayHW, ArrayHW, date]:
    """Extract a single frame from the 3D stream for static plots.

    Returns the first timestep from the stream as 2D arrays.
    """
    ground_truth_stream, prediction_stream, dates = sic_pair_3d_stream
    return ground_truth_stream[0], prediction_stream[0], dates[0]


def make_varying_sic_stream(
    *,
    dist_grid: np.ndarray,
    timesteps: int,
    base_radius: float,
    rng: np.random.Generator,
    ring_width: float = 6.0,
    noise_std: float = 0.03,
    radius_oscillation_amplitude: float = 0.5,
    radius_oscillation_frequency: float = 0.7,
) -> np.ndarray:
    """Vectorized [T, H, W] sea-ice concentration with oscillating coastline radius."""
    height, width = dist_grid.shape
    radius_t = base_radius + radius_oscillation_amplitude * np.sin(
        radius_oscillation_frequency * np.arange(timesteps, dtype=float)
    )
    dist_b = dist_grid[None, :, :]
    radius_b = radius_t[:, None, None]
    outside = np.maximum(0.0, dist_b - radius_b)
    sic = np.clip(
        np.exp(-(outside / max(1.0, float(ring_width))))
        + rng.normal(0.0, noise_std, size=(timesteps, height, width)),
        0.0,
        1.0,
    )
    sic[dist_b < radius_b] = 0.0
    return sic.astype(np.float32)


def _make_circular_arctic(
    height: int,
    width: int,
    *,
    rng: np.random.Generator,
    ring_width: float = 6.0,
    noise: float = 0.05,
) -> np.ndarray:
    """Return a synthetic [H, W] sea-ice concentration map with a circular ice ring."""
    # Create a grid of distances from the centre
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Choose a radius so the land takes most of the centre
    radius = min(height, width) * 0.25

    # Ice strength: 1 at the ring, then smoothly falls to 0 with distance
    falloff = np.exp(-(np.maximum(0.0, dist - radius) / max(1.0, ring_width)))

    # Add some noise so the ring looks less perfect
    sic = np.clip(falloff + rng.normal(0.0, noise, size=(height, width)), 0.0, 1.0)

    # Land mask: set everything inside the circle to zero
    sic[dist < radius] = 0.0
    return sic.astype(np.float32)


def make_central_distance_grid(height: int, width: int) -> ArrayHW:
    """Return per-pixel Euclidean distance from the image centre.

    The centre is defined as ((H-1)/2, (W-1)/2) so that distances are symmetric
    for even-sized grids.
    """
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


@pytest.fixture
def sic_pair_3d_stream() -> tuple[ArrayTHW, ArrayTHW, list[date]]:
    """Short 3D streams (time, height, width) and dates for animations.

    Shape (4, 48, 48), values in [0, 1]. Frames drift slightly over time
    with a bit of noise to mimic day-to-day change.
    """
    rng = np.random.default_rng(100)
    timesteps, height, width = N_TIMESTEPS, 48, 48

    dist = make_central_distance_grid(height, width)

    coastline_base_radius = min(height, width) * 0.25

    # Generate ground truth stream using oscillating sea-ice radius
    ground_truth_stream = make_varying_sic_stream(
        dist_grid=dist,
        timesteps=timesteps,
        base_radius=coastline_base_radius,
        rng=rng,
        ring_width=6.0,
        noise_std=0.03,
        radius_oscillation_amplitude=0.5,
        radius_oscillation_frequency=0.7,
    )

    # Prediction: same circle shape as ground truth, but randomised ice distribution noise
    prediction_stream = np.stack(
        [
            _make_circular_arctic(height, width, rng=rng, noise=0.08)
            for _ in range(timesteps)
        ],
        axis=0,
    )
    dates = [TEST_DATE + timedelta(days=int(d)) for d in range(timesteps)]
    return ground_truth_stream, prediction_stream, dates


@pytest.fixture
def base_plot_spec() -> PlotSpec:
    """Base plotting specification for raw inputs."""
    return PlotSpec(
        colourmap="viridis",
        hemisphere=Hemisphere.SOUTH,
    )


@pytest.fixture
def test_dates_short() -> list[date]:
    """Generate a short sequence of test dates for animations (4 days)."""
    return [TEST_DATE + timedelta(days=i) for i in range(N_TIMESTEPS)]


@pytest.fixture
def no_land_mask() -> LandMask:
    """A LandMask with no mask registered; apply_to is a pass-through no-op."""
    return LandMask(None)


@pytest.fixture
def era5_temperature_2d() -> ArrayHW:
    """Generate synthetic ERA5 2m temperature data (K) [H, W]."""
    rng = np.random.default_rng(100)
    # Temperature centreed around 273.15K (0°C) with realistic variation
    base_temp = 273.15 + rng.normal(0, 10, size=(TEST_HEIGHT, TEST_WIDTH))
    return base_temp.astype(np.float32)


@pytest.fixture
def osisaf_ice_conc_2d() -> ArrayHW:
    """Generate synthetic OSISAF sea ice concentration data (fraction 0-1) [H, W]."""
    rng = np.random.default_rng(100)
    # Ice concentration between 0 and 1
    ice_conc = rng.uniform(0.0, 1.0, size=(TEST_HEIGHT, TEST_WIDTH))
    return ice_conc.astype(np.float32)


@pytest.fixture
def era5_temperature_thw(test_dates_short: list[date]) -> ArrayTHW:
    """Generate synthetic 3D temperature stream [T, H, W]."""
    rng = np.random.default_rng(100)
    n_timesteps = len(test_dates_short)
    # Temperature evolving over time
    data = np.zeros((n_timesteps, TEST_HEIGHT, TEST_WIDTH), dtype=np.float32)
    for t in range(n_timesteps):
        data[t] = 273.15 + rng.normal(0, 10, size=(TEST_HEIGHT, TEST_WIDTH))
    return data


@pytest.fixture
def variable_styles() -> dict[str, dict[str, Any]]:
    """Sample variable styling configuration for raw inputs."""
    return {
        "era5:2t": {"cmap": "RdBu_r", "units": "K"},
        "era5:10u": {"cmap": "RdBu_r", "units": "m/s"},
        "era5:10v": {"cmap": "RdBu_r", "units": "m/s"},
        "era5:q_10": {"cmap": "viridis", "units": "kg/kg"},
        "osisaf-south:ice_conc": {"cmap": "Blues_r"},
    }
