"""Geographically aligned interpolation between structured grids."""

from collections.abc import Sequence

import numpy as np
import torch
from pyproj import Transformer
from torch import nn
from torch.nn import functional as F

from cryocast.types import TensorNCHW


class GeographicInterpolation(nn.Module):
    """Bilinearly sample a regular source grid at target-grid coordinates."""

    sampling_grid: torch.Tensor

    def __init__(  # noqa: PLR0913
        self,
        *,
        source_latitudes: Sequence[float],
        source_longitudes: Sequence[float],
        source_shape: tuple[int, int],
        target_latitudes: Sequence[float],
        target_longitudes: Sequence[float],
        target_shape: tuple[int, int],
        source_crs: str,
    ) -> None:
        """Precompute a differentiable source-to-target sampling grid."""
        super().__init__()
        source_lat = np.asarray(source_latitudes, dtype=np.float64).reshape(
            source_shape
        )
        source_lon = np.asarray(source_longitudes, dtype=np.float64).reshape(
            source_shape
        )
        target_lat = np.asarray(target_latitudes, dtype=np.float64).reshape(
            target_shape
        )
        target_lon = np.asarray(target_longitudes, dtype=np.float64).reshape(
            target_shape
        )

        transformer = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
        source_x, source_y = transformer.transform(source_lon, source_lat)
        target_x, target_y = transformer.transform(target_lon, target_lat)

        source_x = np.asarray(source_x, dtype=np.float64)
        source_y = np.asarray(source_y, dtype=np.float64)
        target_x = np.asarray(target_x, dtype=np.float64)
        target_y = np.asarray(target_y, dtype=np.float64)

        x_axis = np.mean(source_x, axis=0)
        y_axis = np.mean(source_y, axis=1)
        self._validate_regular_axis(x_axis, "x")
        self._validate_regular_axis(y_axis, "y")
        self._validate_separable(source_x, x_axis[None, :], x_axis, "x")
        self._validate_separable(source_y, y_axis[:, None], y_axis, "y")

        grid_x = self._normalise_axis(target_x, x_axis)
        grid_y = self._normalise_axis(target_y, y_axis)
        tolerance = 1e-3
        if (
            np.nanmin(grid_x) < -1.0 - tolerance
            or np.nanmax(grid_x) > 1.0 + tolerance
            or np.nanmin(grid_y) < -1.0 - tolerance
            or np.nanmax(grid_y) > 1.0 + tolerance
        ):
            msg = "Target grid extends outside the source grid."
            raise ValueError(msg)

        grid = np.stack(
            (np.clip(grid_x, -1.0, 1.0), np.clip(grid_y, -1.0, 1.0)), axis=-1
        )
        self.register_buffer(
            "sampling_grid",
            torch.from_numpy(grid).float().unsqueeze(0),
            persistent=True,
        )
        self.source_shape = source_shape
        self.target_shape = target_shape

    @staticmethod
    def _validate_regular_axis(axis: np.ndarray, name: str) -> None:
        """Require a finite, monotonic, regularly spaced source axis."""
        if not np.all(np.isfinite(axis)):
            msg = f"Source {name}-axis contains non-finite coordinates."
            raise ValueError(msg)
        steps = np.diff(axis)
        if (
            steps.size == 0
            or np.any(steps == 0)
            or not (np.all(steps > 0) or np.all(steps < 0))
        ):
            msg = f"Source {name}-axis must be strictly monotonic."
            raise ValueError(msg)
        if not np.allclose(steps, np.mean(steps), rtol=1e-3, atol=1e-3):
            msg = f"Source {name}-axis must be regularly spaced."
            raise ValueError(msg)

    @staticmethod
    def _validate_separable(
        coordinates: np.ndarray,
        expected: np.ndarray,
        axis: np.ndarray,
        name: str,
    ) -> None:
        """Require projected source coordinates to be separable into two axes."""
        step = float(np.mean(np.abs(np.diff(axis))))
        tolerance = max(step * 1e-3, 1e-3)
        if float(np.max(np.abs(coordinates - expected))) > tolerance:
            msg = (
                f"Projected source {name}-coordinates are not a separable regular grid."
            )
            raise ValueError(msg)

    @staticmethod
    def _normalise_axis(target: np.ndarray, source_axis: np.ndarray) -> np.ndarray:
        """Map projected coordinates onto grid_sample's [-1, 1] coordinates."""
        span = source_axis[-1] - source_axis[0]
        return 2.0 * (target - source_axis[0]) / span - 1.0

    def forward(self, inputs: TensorNCHW) -> TensorNCHW:
        """Interpolate NCHW inputs onto the precomputed target grid."""
        if tuple(inputs.shape[-2:]) != self.source_shape:
            msg = (
                f"Expected source shape {self.source_shape}, got "
                f"{tuple(inputs.shape[-2:])}."
            )
            raise ValueError(msg)
        grid = self.sampling_grid.to(device=inputs.device, dtype=inputs.dtype)
        return F.grid_sample(
            inputs,
            grid.expand(inputs.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
