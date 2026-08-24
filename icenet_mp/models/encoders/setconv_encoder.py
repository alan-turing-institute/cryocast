"""SetConv encoder for sparse or irregular observations on fixed coordinates."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

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

        self.register_buffer(
            "input_xyz",
            self._latlon_to_xyz(
                self.latitudes[self.project_from], self.longitudes[self.project_from]
            ),
            persistent=False,
        )
        self.register_buffer(
            "output_xyz",
            self._latlon_to_xyz(
                self.latitudes[self.project_to], self.longitudes[self.project_to]
            ),
            persistent=False,
        )

        self.log_length_scale_degrees = nn.Parameter(
            torch.tensor(length_scale_degrees).log(),
            requires_grad=learnable_length_scale,
        )
        self.feature_projection = nn.Conv2d(
            2 * data_space_in.channels,
            self.data_space_out.channels,
            kernel_size=1,
        )

    @staticmethod
    def _latlon_to_xyz(latitudes: list[float], longitudes: list[float]) -> torch.Tensor:
        """Convert latitude/longitude degrees to unit-sphere Cartesian coordinates."""
        lat = torch.deg2rad(torch.tensor(latitudes, dtype=torch.float32))
        lon = torch.deg2rad(torch.tensor(longitudes, dtype=torch.float32))
        cos_lat = torch.cos(lat)
        return torch.stack(
            (cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)), dim=-1
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
            if len(self.latitudes[name]) != expected or len(self.longitudes[name]) != expected:
                msg = (
                    f"Dataset '{name}' has an incompatible coordinate count; "
                    f"expected {expected} latitude/longitude pairs."
                )
                raise ValueError(msg)

    def _kernel(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return Gaussian weights from every output point to every input point."""
        input_xyz = self.input_xyz.to(device=device, dtype=dtype)
        output_xyz = self.output_xyz.to(device=device, dtype=dtype)
        squared_distance = torch.sum(
            (output_xyz[:, None, :] - input_xyz[None, :, :]) ** 2, dim=-1
        )

        # Convert the learnable angular scale into a unit-sphere chord distance.
        angle = torch.deg2rad(self.log_length_scale_degrees.exp()).to(dtype=dtype)
        chord_scale = (2 * torch.sin(angle / 2)).clamp_min(torch.finfo(dtype).eps)
        return torch.exp(-0.5 * squared_distance / chord_scale.square())

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
