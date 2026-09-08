"""SetConv encoder for sparse or irregular observations on fixed coordinates."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from haversine import Unit
from torch import nn

from icenet_mp.geotools import pairwise_haversine_distances
from icenet_mp.types import DataSpace, TensorNCHW

from .base_encoder import BaseEncoder


class SetConvEncoder(BaseEncoder):
    """Map sparse observations at fixed coordinates onto a regular latent grid.

    The encoder applies a Gaussian set convolution on the sphere. For each output
    grid cell it computes a distance-weighted mean of all finite observations and
    a corresponding observation-density feature, then mixes those features with a
    learnable 1x1 convolution.

    Input locations are taken from ``latitudes_fn``/``longitudes_fn`` under the
    input dataset name. Output locations are taken from ``project_to``. This first
    implementation intentionally assumes those sensor coordinates are fixed across
    timesteps; moving-sensor support can be added once coordinates are carried with
    each sample by the data pipeline.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        data_space_in: DataSpace,
        latent_space: tuple[int, int],
        project_to: str,
        output_channels: int | None = None,
        length_scale_degrees: float = 5.0,
        learnable_length_scale: bool = True,
        latitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        longitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a fixed-coordinate SetConv encoder."""
        if length_scale_degrees <= 0:
            msg = "length_scale_degrees must be positive."
            raise ValueError(msg)

        super().__init__(
            data_space_in=data_space_in,
            latent_space=latent_space,
            output_channels=output_channels or data_space_in.channels,
            latitudes_fn=latitudes_fn,
            longitudes_fn=longitudes_fn,
            **kwargs,
        )

        self.project_from = data_space_in.name
        self.project_to = project_to
        self._validate_coordinates()

        input_latlons = np.column_stack(
            (
                self.latitudes[self.project_from],
                self.longitudes[self.project_from],
            )
        )
        output_latlons = np.column_stack(
            (self.latitudes[self.project_to], self.longitudes[self.project_to])
        )
        angular_distances = torch.tensor(
            pairwise_haversine_distances(
                input_latlons, output_latlons, unit=Unit.RADIANS
            ),
            dtype=torch.float32,
        )
        self.register_buffer("_angular_distances", angular_distances, persistent=False)

        self.log_length_scale_degrees = nn.Parameter(
            torch.tensor(length_scale_degrees).log(),
            requires_grad=learnable_length_scale,
        )
        self.feature_projection = nn.Conv2d(
            2 * data_space_in.channels,
            self.data_space_out.channels,
            kernel_size=1,
        )

    def _validate_coordinates(self) -> None:
        """Validate that source and target coordinate counts match their data spaces."""
        for name, expected in (
            (self.project_from, self.data_space_in.area),
            (self.project_to, self.data_space_out.area),
        ):
            if name not in self.latitudes or name not in self.longitudes:
                msg = f"Missing coordinates for dataset '{name}'."
                raise ValueError(msg)
            if (
                len(self.latitudes[name]) != expected
                or len(self.longitudes[name]) != expected
            ):
                msg = (
                    f"Dataset '{name}' has an incompatible coordinate count; "
                    f"expected {expected} latitude/longitude pairs."
                )
                raise ValueError(msg)

    def _kernel(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return Gaussian weights from every output point to every input point."""
        angular_distances = self.get_buffer("_angular_distances").to(
            device=device, dtype=dtype
        )
        length_scale = torch.deg2rad(self.log_length_scale_degrees.exp()).to(
            device=device, dtype=dtype
        )
        length_scale = length_scale.clamp_min(torch.finfo(dtype).eps)
        return torch.exp(-0.5 * angular_distances.square() / length_scale.square())

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Interpolate finite observations onto the latent grid and project channels."""
        batch_size, channels, _, _ = x.shape
        values = x.reshape(batch_size, channels, self.data_space_in.area)
        valid = torch.isfinite(values)
        values = torch.nan_to_num(values)

        kernel = self._kernel(device=x.device, dtype=x.dtype)
        weights = kernel[None, None, :, :] * valid[:, :, None, :].to(x.dtype)
        density = weights.sum(dim=-1)
        weighted_values = (weights * values[:, :, None, :]).sum(dim=-1)
        smooth = weighted_values / density.clamp_min(torch.finfo(x.dtype).eps)

        features = torch.cat((smooth, torch.log1p(density)), dim=1).reshape(
            batch_size,
            2 * channels,
            *self.data_space_out.shape,
        )
        return self.feature_projection(features)
