"""Deterministic synthetic spatiotemporal sequences for fast model-pipeline checks.

Two families of trivially-learnable spatiotemporal forecasting task, each with a known
ground truth, so they can validate that a model+data pipeline is wired up correctly
(shapes, history/forecast windowing, rollout) and that a model is actually learning,
without needing real sea-ice data:

- ``MovingCircleConfig`` -- a rigid circle translating and bouncing off the grid edges.
  Exercises whether the model learns advection (drift).
- ``GrowShrinkCircleConfig`` -- a stationary blob that grows and shrinks in place via
  morphological dilation/erosion (an open/close cycle), mimicking sea ice growing and
  melting seasonally. Exercises whether the model learns concentration change in place,
  which the translating circle never shows.
"""

import math
from dataclasses import dataclass

import numpy as np

from cryocast.types import ArrayTHW


@dataclass(frozen=True)
class MovingCircleConfig:
    """Configuration for a single circle bouncing around inside an HxW grid."""

    height: int = 32
    width: int = 32
    n_timesteps: int = 60
    radius: float = 5.0
    velocity: tuple[float, float] = (2.0, 1.0)
    start_position: tuple[float, float] | None = None
    foreground_value: float = 1.0
    background_value: float = 0.0


def generate_moving_circle_frames(config: MovingCircleConfig) -> ArrayTHW:
    """Generate a [T, H, W] sequence of a circle bouncing around inside a grid.

    The circle reflects off the grid edges rather than wrapping around them, so it
    always stays fully inside the frame and is never split across edges/corners.
    """
    height, width = config.height, config.width
    row, col = config.start_position or (height / 2, width / 2)
    v_row, v_col = config.velocity
    row_margin, col_margin = config.radius, config.radius
    row_max, col_max = height - 1 - row_margin, width - 1 - col_margin

    row_grid, col_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    frames = np.empty((config.n_timesteps, height, width), dtype=np.float32)
    for t in range(config.n_timesteps):
        inside = (row_grid - row) ** 2 + (col_grid - col) ** 2 <= config.radius**2
        frames[t] = np.where(inside, config.foreground_value, config.background_value)

        row, col = row + v_row, col + v_col
        if row < row_margin or row > row_max:
            v_row = -v_row
            row = np.clip(row, row_margin, row_max)
        if col < col_margin or col > col_max:
            v_col = -v_col
            col = np.clip(col, col_margin, col_max)

    return frames


@dataclass(frozen=True)
class GrowShrinkCircleConfig:
    """Configuration for a stationary blob that grows and shrinks in place.

    A seed circle is repeatedly dilated (grow) or eroded (shrink) with a small
    structuring element, following a sinusoidal schedule -- a morphological open/close
    cycle. This mimics sea ice advancing and retreating seasonally: the shape's extent
    changes in place rather than translating.
    """

    height: int = 32
    width: int = 32
    n_timesteps: int = 20
    base_radius: float = 6.0
    growth: int = 4  # max dilation iterations at the peak of the cycle
    shrinkage: int = 4  # max erosion iterations at the trough of the cycle
    period: float = 20.0  # timesteps for one full grow-shrink cycle
    phase: float = 0.0  # fraction of a cycle to offset the start by, in [0, 1)
    center: tuple[float, float] | None = None
    foreground_value: float = 1.0
    background_value: float = 0.0


# Two structuring elements: 4-connectivity (a plus, growing a diamond) and
# 8-connectivity (a square). Alternating them each iteration approximates an octagon,
# i.e. near-circular growth, avoiding the pure-diamond artefact of a fixed plus.
_CROSS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _morph_once(mask: np.ndarray, *, erode: bool, diagonal: bool) -> np.ndarray:
    """Apply one morphological dilation or erosion step to a boolean mask.

    Out-of-bounds neighbours count as background (False), so the shape grows freely
    into the grid and erodes against its edges, matching ``scipy.ndimage``'s default
    ``binary_dilation``/``binary_erosion`` border behaviour.
    """
    height, width = mask.shape
    offsets = _CROSS_OFFSETS + (_DIAGONAL_OFFSETS if diagonal else ())
    padded = np.pad(mask, 1, constant_values=False)
    result: np.ndarray | None = None
    for d_row, d_col in offsets:
        neighbour = padded[
            1 + d_row : 1 + d_row + height, 1 + d_col : 1 + d_col + width
        ]
        if result is None:
            result = neighbour.copy()
        elif erode:
            result &= neighbour
        else:
            result |= neighbour
    assert result is not None  # noqa: S101 -- offsets is never empty
    return result


def _apply_morphology(seed: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate (``iterations`` > 0) or erode (< 0) ``seed`` that many times."""
    mask = seed
    erode = iterations < 0
    for step in range(abs(iterations)):
        mask = _morph_once(mask, erode=erode, diagonal=step % 2 == 1)
    return mask


def generate_grow_shrink_frames(config: GrowShrinkCircleConfig) -> ArrayTHW:
    """Generate a [T, H, W] sequence of a blob growing and shrinking in place.

    Each frame is the seed circle dilated (grown) or eroded (shrunk) by an amount that
    follows a sine wave over time: the extent expands to ``+growth`` iterations at the
    peak and contracts to ``-shrinkage`` iterations at the trough. The blob never moves.
    """
    height, width = config.height, config.width
    row, col = config.center or (height / 2, width / 2)

    row_grid, col_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    seed = (row_grid - row) ** 2 + (col_grid - col) ** 2 <= config.base_radius**2

    frames = np.empty((config.n_timesteps, height, width), dtype=np.float32)
    for t in range(config.n_timesteps):
        cycle = math.sin(2 * math.pi * (t / config.period + config.phase))
        amplitude = config.growth if cycle >= 0 else config.shrinkage
        iterations = round(amplitude * cycle)
        mask = _apply_morphology(seed, iterations)
        frames[t] = np.where(mask, config.foreground_value, config.background_value)

    return frames
