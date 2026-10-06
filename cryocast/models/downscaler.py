"""Trainable geospatial downscaling of low-resolution sea-ice predictions."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from typing_extensions import override

from cryocast.types import TensorNTCHW

from .base_model import BaseModel
from .common import GeographicInterpolation


class _ResidualUnit(nn.Module):
    """Small residual block used by the downscaling refiner."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class _ResidualRefiner(nn.Module):
    """Predict a high-resolution correction around an interpolation baseline."""

    def __init__(
        self,
        *,
        channels: int,
        hidden_channels: int,
        n_residual_blocks: int,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            msg = "hidden_channels must be greater than 0."
            raise ValueError(msg)
        if n_residual_blocks < 0:
            msg = "n_residual_blocks must be greater than or equal to 0."
            raise ValueError(msg)

        self.head = nn.Sequential(
            nn.Conv2d(channels + 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            *[_ResidualUnit(hidden_channels) for _ in range(n_residual_blocks)]
        )
        self.tail = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

        # Start exactly at the interpolation baseline. Training therefore has to
        # demonstrate a useful residual rather than starting from an arbitrary field.
        nn.init.zeros_(self.tail.weight)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    def forward(
        self, baseline: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((baseline, coordinates), dim=1)
        return self.tail(self.body(self.head(features)))


class Downscaler(BaseModel):
    """Learn residual high-resolution structure around a geographic interpolation."""

    ignored_hparams: ClassVar[frozenset[str]] = BaseModel.ignored_hparams | {
        "target_variable_indices",
        "variable_names",
    }

    coordinate_features: torch.Tensor
    validity_mask: torch.Tensor | None

    def __init__(  # noqa: PLR0913
        self,
        *,
        source_group_name: str,
        source_variable: str,
        source_crs: str,
        variable_names: Mapping[str, list[str]] | DictConfig,
        target_variable_indices: Sequence[int] | None = None,
        hidden_channels: int = 32,
        n_residual_blocks: int = 4,
        residual_scale: float = 0.25,
        mask_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a downscaler from one low-resolution source variable."""
        _ = target_variable_indices  # Target selection is reflected in output_space.
        super().__init__(mask_dir=mask_dir, **kwargs)

        if residual_scale <= 0:
            msg = "residual_scale must be greater than 0."
            raise ValueError(msg)
        self.residual_scale = float(residual_scale)
        self.source_group_name = source_group_name
        self.source_variable = source_variable

        source_spaces = [
            space for space in self.input_spaces if space.name == source_group_name
        ]
        if len(source_spaces) != 1:
            msg = (
                f"Expected exactly one input space named {source_group_name!r}, found "
                f"{len(source_spaces)}."
            )
            raise ValueError(msg)
        self.source_space = source_spaces[0]

        names = list(variable_names[source_group_name])
        try:
            self.source_variable_index = names.index(source_variable)
        except ValueError as exc:
            msg = (
                f"Source variable {source_variable!r} was not found in "
                f"{source_group_name!r}: {names}."
            )
            raise ValueError(msg) from exc

        if self.output_space.channels != 1:
            msg = (
                "Downscaler currently requires a single target channel, got "
                f"{self.output_space.channels}."
            )
            raise ValueError(msg)

        self.interpolator = GeographicInterpolation(
            source_latitudes=self.latitudes[source_group_name],
            source_longitudes=self.longitudes[source_group_name],
            source_shape=self.source_space.shape,
            target_latitudes=self.latitudes[self.output_space.name],
            target_longitudes=self.longitudes[self.output_space.name],
            target_shape=self.output_space.shape,
            source_crs=source_crs,
        )
        self.refiner = _ResidualRefiner(
            channels=self.output_space.channels,
            hidden_channels=hidden_channels,
            n_residual_blocks=n_residual_blocks,
        )

        target_latitudes = torch.as_tensor(
            self.latitudes[self.output_space.name], dtype=torch.float32
        ).reshape(self.output_space.shape)
        target_longitudes = torch.as_tensor(
            self.longitudes[self.output_space.name], dtype=torch.float32
        ).reshape(self.output_space.shape)
        target_longitudes = torch.remainder(target_longitudes + 180.0, 360.0) - 180.0
        coordinate_features = torch.stack(
            (target_latitudes / 90.0, target_longitudes / 180.0), dim=0
        ).unsqueeze(0)
        self.register_buffer(
            "coordinate_features", coordinate_features, persistent=True
        )

        validity_mask = None
        if mask_dir is not None:
            mask_path = Path(mask_dir) / "land_mask.npy"
            if mask_path.exists():
                mask = np.load(mask_path)
                if tuple(mask.shape) != self.output_space.shape:
                    msg = (
                        f"Target mask shape {tuple(mask.shape)} does not match output "
                        f"shape {self.output_space.shape}."
                    )
                    raise ValueError(msg)
                validity_mask = torch.from_numpy(mask.astype(bool))
        self.register_buffer(
            "validity_mask",
            validity_mask,
            persistent=False,
        )

    def interpolation_baseline(self, inputs: TensorNTCHW) -> TensorNTCHW:
        """Geographically interpolate low-resolution fields to the target grid."""
        if inputs.ndim != 5:  # noqa: PLR2004
            msg = f"Expected NTCHW input, got shape {tuple(inputs.shape)}."
            raise ValueError(msg)
        if inputs.shape[2] != self.output_space.channels:
            msg = (
                f"Expected {self.output_space.channels} channel(s), got "
                f"{inputs.shape[2]}."
            )
            raise ValueError(msg)
        batch, time, channels, height, width = inputs.shape
        flattened = inputs.reshape(batch * time, channels, height, width)
        interpolated = self.interpolator(flattened)
        return interpolated.reshape(batch, time, channels, *self.output_space.shape)

    def downscale(self, low_resolution_prediction: TensorNTCHW) -> TensorNTCHW:
        """Downscale every forecast step independently."""
        baseline = self.interpolation_baseline(low_resolution_prediction)
        batch, time, channels, height, width = baseline.shape
        flat_baseline = baseline.reshape(batch * time, channels, height, width)
        coordinates = self.coordinate_features.to(
            device=flat_baseline.device, dtype=flat_baseline.dtype
        ).expand(batch * time, -1, -1, -1)
        residual = self.refiner(flat_baseline, coordinates)
        refined = torch.clamp(
            flat_baseline + self.residual_scale * residual,
            min=0.0,
            max=1.0,
        )
        return refined.reshape(batch, time, channels, height, width)

    @override
    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Downscale contemporaneous low-resolution observations during training."""
        source = inputs[self.source_group_name]
        if source.shape[1] < self.n_forecast_steps:
            msg = (
                f"Need at least {self.n_forecast_steps} source timestep(s), got "
                f"{source.shape[1]}."
            )
            raise ValueError(msg)
        source = source[
            :,
            -self.n_forecast_steps :,
            self.source_variable_index : self.source_variable_index + 1,
            :,
            :,
        ]
        return self.downscale(source)

    @override
    def loss(self, prediction: TensorNTCHW, target: TensorNTCHW) -> torch.Tensor:
        """Compute loss only where the high-resolution target is valid."""
        if prediction.shape != target.shape:
            msg = (
                f"Prediction shape {tuple(prediction.shape)} does not match target "
                f"shape {tuple(target.shape)}."
            )
            raise ValueError(msg)

        valid = torch.isfinite(target)
        if self.validity_mask is not None:
            mask = self.validity_mask.to(device=target.device)
            valid &= mask.view(1, 1, 1, *self.output_space.shape)

        if not torch.any(valid):
            msg = "Downscaling target contains no valid cells."
            raise ValueError(msg)
        return self.loss_fn(prediction[valid], target[valid])


class DownscalingPipeline(nn.Module):
    """Run a forecast model and immediately downscale its low-resolution output."""

    def __init__(self, forecast_model: nn.Module, downscaler: Downscaler) -> None:
        """Initialise the forecast-to-downscaler composition."""
        super().__init__()
        self.forecast_model = forecast_model
        self.downscaler = downscaler

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Produce a low-resolution forecast and downscale each forecast step."""
        low_resolution_prediction = self.forecast_model(inputs)
        if not isinstance(low_resolution_prediction, torch.Tensor):
            msg = (
                "Forecast model must return a tensor before downscaling, got "
                f"{type(low_resolution_prediction).__name__}."
            )
            raise TypeError(msg)
        return self.downscaler.downscale(low_resolution_prediction)
