"""Experimental SetConv-style encoding for sparse, moving Argo observations."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .argo_torch import TorchSparseSequenceBatch, latlon_to_unit_xyz

_EARTH_RADIUS_KM = 6371.0088
_LATENT_GRID_NDIM = 2


@dataclass(frozen=True, slots=True)
class SparseSetConvConfig:
    """Configuration for the experimental sparse SetConv encoder."""

    variable_names: tuple[str, ...] = ("TEMP", "PSAL")
    latent_channels: int = 16
    hidden_channels: int = 16
    length_scale_km: float = 2000.0
    query_chunk_size: int = 1024
    auxiliary_channels: int = 0
    use_pressure: bool = False
    use_time_offsets: bool = False
    epsilon: float = 1e-8


def _great_circle_distance_km(
    observation_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
) -> torch.Tensor:
    """Return great-circle distances for observations and query coordinates."""
    chord = torch.linalg.vector_norm(
        query_xyz[None, :, None, :] - observation_xyz[:, None, :, :],
        dim=-1,
    )
    half_chord = torch.clamp(chord * 0.5, min=0.0, max=1.0)
    return 2.0 * _EARTH_RADIUS_KM * torch.asin(half_chord)


class SparseSetConvEncoder(nn.Module):
    """Encode moving sparse observations directly onto a latent spatial grid.

    Coordinates are supplied on every forward call. Unit-sphere coordinates avoid a
    longitude discontinuity at the dateline. Gaussian kernels aggregate measurements
    and an observation-density channel directly at query locations. A pointwise MLP
    maps this SetConv representation to the latent width. Using the same projection
    for grid and arbitrary query locations enables a symmetric reconstruction probe.
    """

    latent_query_xyz: torch.Tensor

    def __init__(
        self,
        config: SparseSetConvConfig,
        latent_latitudes: np.ndarray,
        latent_longitudes: np.ndarray,
    ) -> None:
        """Initialise the encoder and fixed latent-grid query coordinates."""
        super().__init__()
        self._validate_config(config)
        latitudes = torch.as_tensor(latent_latitudes, dtype=torch.float32)
        longitudes = torch.as_tensor(latent_longitudes, dtype=torch.float32)
        if latitudes.ndim != _LATENT_GRID_NDIM or longitudes.shape != latitudes.shape:
            msg = "Latent latitude and longitude grids must be aligned 2D arrays."
            raise ValueError(msg)

        self.config = config
        self.latent_shape = (latitudes.shape[0], latitudes.shape[1])
        self.register_buffer(
            "latent_query_xyz",
            latlon_to_unit_xyz(latitudes, longitudes).reshape(-1, 3),
        )
        feature_channels = len(config.variable_names) + config.auxiliary_channels
        feature_channels += int(config.use_pressure) + int(config.use_time_offsets)
        self.input_feature_channels = feature_channels
        self.query_projection = nn.Sequential(
            nn.Linear(feature_channels + 1, config.hidden_channels, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_channels, config.latent_channels, bias=False),
        )

    @staticmethod
    def _validate_config(config: SparseSetConvConfig) -> None:
        if not config.variable_names:
            msg = "At least one input variable is required."
            raise ValueError(msg)
        if (
            min(
                config.latent_channels,
                config.hidden_channels,
                config.query_chunk_size,
            )
            <= 0
            or config.auxiliary_channels < 0
        ):
            msg = "Channel counts and query chunk size must be valid positive values."
            raise ValueError(msg)
        if config.length_scale_km <= 0 or config.epsilon <= 0:
            msg = "Kernel length scale and epsilon must be positive."
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return a stable encoder name."""
        return "sparse-setconv"

    @property
    def latent_channels(self) -> int:
        """Return the output latent width used by the common reconstruction head."""
        return self.config.latent_channels

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return the ordered observed variables expected by the encoder."""
        return self.config.variable_names

    def forward(self, batch: TorchSparseSequenceBatch) -> torch.Tensor:
        """Return direct latent encoding with shape ``[B,T,C_latent,H,W]``."""
        self._validate_batch(batch)
        latent = self._encode_query_xyz(batch, self.latent_query_xyz)
        batch_size, n_times = batch.latitudes.shape[:2]
        height, width = self.latent_shape
        return latent.reshape(
            batch_size,
            n_times,
            height,
            width,
            self.config.latent_channels,
        ).permute(0, 1, 4, 2, 3)

    def encode_queries(
        self,
        batch: TorchSparseSequenceBatch,
        query_latitudes: torch.Tensor,
        query_longitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode arbitrary geographic queries to ``[B,T,Q,C_latent]``."""
        self._validate_batch(batch)
        if query_latitudes.ndim != 1 or query_longitudes.shape != query_latitudes.shape:
            msg = "Query latitude and longitude must be aligned 1D tensors."
            raise ValueError(msg)
        query_xyz = latlon_to_unit_xyz(
            query_latitudes.to(
                device=batch.latitudes.device,
                dtype=batch.latitudes.dtype,
            ),
            query_longitudes.to(
                device=batch.longitudes.device,
                dtype=batch.longitudes.dtype,
            ),
        )
        return self._encode_query_xyz(batch, query_xyz)

    def _encode_query_xyz(
        self,
        batch: TorchSparseSequenceBatch,
        query_xyz: torch.Tensor,
    ) -> torch.Tensor:
        features, density = self._aggregate_to_queries(batch, query_xyz)
        combined = torch.cat((features, torch.log1p(density)), dim=-1)
        return self.query_projection(combined)

    def _validate_batch(self, batch: TorchSparseSequenceBatch) -> None:
        if batch.variable_names != self.config.variable_names:
            msg = "Sparse batch variables must match the encoder configuration."
            raise ValueError(msg)
        actual_auxiliary = (
            0
            if batch.auxiliary_measurements is None
            else batch.auxiliary_measurements.shape[-1]
        )
        if actual_auxiliary != self.config.auxiliary_channels:
            msg = "Sparse batch auxiliary channels do not match the encoder config."
            raise ValueError(msg)
        self._validate_optional_features(batch)

    def _validate_optional_features(self, batch: TorchSparseSequenceBatch) -> None:
        valid = batch.mask
        if self.config.use_pressure:
            if batch.pressure is None:
                msg = "Encoder is configured to use pressure but none was provided."
                raise ValueError(msg)
            if bool(valid.any()) and not bool(
                torch.isfinite(batch.pressure[valid]).all()
            ):
                msg = "Configured pressure values must be finite at valid observations."
                raise ValueError(msg)
        if self.config.use_time_offsets:
            if batch.time_offsets_hours is None:
                msg = (
                    "Encoder is configured to use time offsets but none were provided."
                )
                raise ValueError(msg)
            if bool(valid.any()) and not bool(
                torch.isfinite(batch.time_offsets_hours[valid]).all()
            ):
                msg = "Configured time offsets must be finite at valid observations."
                raise ValueError(msg)
        if (
            batch.auxiliary_measurements is not None
            and bool(valid.any())
            and not bool(torch.isfinite(batch.auxiliary_measurements[valid]).all())
        ):
            msg = "Configured auxiliary values must be finite at valid observations."
            raise ValueError(msg)

    def _input_features(self, batch: TorchSparseSequenceBatch) -> torch.Tensor:
        feature_parts = [batch.measurements]
        if self.config.use_pressure and batch.pressure is not None:
            feature_parts.append(batch.pressure.unsqueeze(-1))
        if self.config.use_time_offsets and batch.time_offsets_hours is not None:
            feature_parts.append(batch.time_offsets_hours.unsqueeze(-1))
        if batch.auxiliary_measurements is not None:
            feature_parts.append(batch.auxiliary_measurements)
        features = torch.cat(feature_parts, dim=-1)
        return torch.where(
            batch.mask.unsqueeze(-1),
            torch.nan_to_num(features),
            torch.zeros_like(features),
        )

    def _aggregate_to_queries(
        self,
        batch: TorchSparseSequenceBatch,
        query_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_times, n_points = batch.latitudes.shape
        n_samples = batch_size * n_times
        observation_xyz = latlon_to_unit_xyz(
            batch.latitudes,
            batch.longitudes,
        ).reshape(n_samples, n_points, 3)
        measurements = self._input_features(batch).reshape(
            n_samples,
            n_points,
            self.input_feature_channels,
        )
        mask = batch.mask.reshape(n_samples, n_points)
        query_xyz = query_xyz.to(
            device=measurements.device,
            dtype=measurements.dtype,
        )
        feature_chunks: list[torch.Tensor] = []
        density_chunks: list[torch.Tensor] = []
        for start in range(0, query_xyz.shape[0], self.config.query_chunk_size):
            query_chunk = query_xyz[start : start + self.config.query_chunk_size]
            distance_km = _great_circle_distance_km(observation_xyz, query_chunk)
            weights = torch.exp(-0.5 * (distance_km / self.config.length_scale_km) ** 2)
            weights = weights * mask[:, None, :].to(dtype=weights.dtype)
            density = weights.sum(dim=-1, keepdim=True)
            weighted_measurements = torch.einsum("sqn,snf->sqf", weights, measurements)
            normalised = torch.where(
                density > self.config.epsilon,
                weighted_measurements / density.clamp_min(self.config.epsilon),
                torch.zeros_like(weighted_measurements),
            )
            feature_chunks.append(normalised)
            density_chunks.append(density)

        features = torch.cat(feature_chunks, dim=1)
        density = torch.cat(density_chunks, dim=1)
        return (
            features.reshape(batch_size, n_times, query_xyz.shape[0], -1),
            density.reshape(batch_size, n_times, query_xyz.shape[0], 1),
        )
