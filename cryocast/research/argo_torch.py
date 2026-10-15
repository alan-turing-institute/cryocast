"""Shared torch helpers for experimental sparse Argo encoders."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .argo_sparse import SparseObservations

_SEQUENCE_COORDINATE_NDIM = 3


@dataclass(frozen=True, slots=True)
class TorchSparseSequenceBatch:
    """Padded sparse observations for a batch of moving-point time sequences."""

    latitudes: torch.Tensor
    longitudes: torch.Tensor
    measurements: torch.Tensor
    mask: torch.Tensor
    variable_names: tuple[str, ...]
    pressure: torch.Tensor | None = None
    time_offsets_hours: torch.Tensor | None = None
    auxiliary_measurements: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate batch, time, observation, and feature dimensions."""
        if (
            self.latitudes.ndim != _SEQUENCE_COORDINATE_NDIM
            or self.longitudes.shape != self.latitudes.shape
        ):
            msg = "Latitude and longitude must have shape [batch, time, observations]."
            raise ValueError(msg)
        if self.mask.shape != self.latitudes.shape or self.mask.dtype != torch.bool:
            msg = "mask must be boolean and match the coordinate shape."
            raise ValueError(msg)
        expected_measurements = (*self.latitudes.shape, len(self.variable_names))
        if self.measurements.shape != expected_measurements:
            msg = "measurements must have shape [batch, time, observations, variables]."
            raise ValueError(msg)
        for optional in (self.pressure, self.time_offsets_hours):
            if optional is not None and optional.shape != self.latitudes.shape:
                msg = "Optional scalar features must match the coordinate shape."
                raise ValueError(msg)
        if (
            self.auxiliary_measurements is not None
            and self.auxiliary_measurements.shape[:_SEQUENCE_COORDINATE_NDIM]
            != self.latitudes.shape
        ):
            msg = "Auxiliary measurements must align with sparse observations."
            raise ValueError(msg)
        valid = self.mask
        if bool(valid.any()) and (
            not bool(torch.isfinite(self.latitudes[valid]).all())
            or not bool(torch.isfinite(self.longitudes[valid]).all())
            or not bool(torch.isfinite(self.measurements[valid]).all())
        ):
            msg = (
                "Valid sparse observations must contain finite coordinates and values."
            )
            raise ValueError(msg)

    @property
    def lengths(self) -> torch.Tensor:
        """Return valid observation counts with shape [batch, time]."""
        return self.mask.sum(dim=-1)


def torch_sparse_sequence_from_observations(
    sequences: Sequence[Sequence[SparseObservations]],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchSparseSequenceBatch:
    """Pad nested ``[batch][time]`` sparse samples into torch tensors."""
    if not sequences or not sequences[0]:
        msg = "At least one non-empty observation sequence is required."
        raise ValueError(msg)
    n_times = len(sequences[0])
    if any(len(sequence) != n_times for sequence in sequences):
        msg = "All batch elements must contain the same number of timesteps."
        raise ValueError(msg)
    variable_names = sequences[0][0].variable_names
    if any(
        sample.variable_names != variable_names
        for sequence in sequences
        for sample in sequence
    ):
        msg = "All sparse samples must use the same variable ordering."
        raise ValueError(msg)

    batch_size = len(sequences)
    max_points = max(sample.count for sequence in sequences for sample in sequence)
    shape = (batch_size, n_times, max_points)
    latitudes = torch.zeros(shape, dtype=dtype, device=device)
    longitudes = torch.zeros(shape, dtype=dtype, device=device)
    measurements = torch.zeros(
        (*shape, len(variable_names)), dtype=dtype, device=device
    )
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    has_pressure = any(
        sample.pressure is not None for sequence in sequences for sample in sequence
    )
    has_time_offsets = any(
        sample.time_offsets_hours is not None
        for sequence in sequences
        for sample in sequence
    )
    pressure = (
        torch.full(shape, torch.nan, dtype=dtype, device=device)
        if has_pressure
        else None
    )
    time_offsets = (
        torch.full(shape, torch.nan, dtype=dtype, device=device)
        if has_time_offsets
        else None
    )

    for batch_index, sequence in enumerate(sequences):
        for time_index, sample in enumerate(sequence):
            count = sample.count
            if count == 0:
                continue
            point_slice = slice(0, count)
            latitudes[batch_index, time_index, point_slice] = torch.as_tensor(
                sample.latitudes, dtype=dtype, device=device
            )
            longitudes[batch_index, time_index, point_slice] = torch.as_tensor(
                sample.longitudes, dtype=dtype, device=device
            )
            measurements[batch_index, time_index, point_slice] = torch.as_tensor(
                sample.measurements, dtype=dtype, device=device
            )
            mask[batch_index, time_index, point_slice] = True
            if pressure is not None and sample.pressure is not None:
                pressure[batch_index, time_index, point_slice] = torch.as_tensor(
                    sample.pressure, dtype=dtype, device=device
                )
            if time_offsets is not None and sample.time_offsets_hours is not None:
                time_offsets[batch_index, time_index, point_slice] = torch.as_tensor(
                    sample.time_offsets_hours, dtype=dtype, device=device
                )

    return TorchSparseSequenceBatch(
        latitudes=latitudes,
        longitudes=longitudes,
        measurements=measurements,
        mask=mask,
        variable_names=variable_names,
        pressure=pressure,
        time_offsets_hours=time_offsets,
    )


def latlon_to_unit_xyz(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
) -> torch.Tensor:
    """Map latitude/longitude to unit-sphere coordinates continuously at the dateline."""
    latitude_radians = torch.deg2rad(latitudes)
    longitude_radians = torch.deg2rad(longitudes)
    cos_latitude = torch.cos(latitude_radians)
    return torch.stack(
        (
            cos_latitude * torch.cos(longitude_radians),
            cos_latitude * torch.sin(longitude_radians),
            torch.sin(latitude_radians),
        ),
        dim=-1,
    )


def spherical_fourier_features(
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    n_frequencies: int,
) -> torch.Tensor:
    """Return unit-sphere coordinates plus Fourier features of those coordinates."""
    xyz = latlon_to_unit_xyz(latitudes, longitudes)
    features = [xyz]
    for exponent in range(n_frequencies):
        frequency = float(2**exponent) * torch.pi
        features.extend((torch.sin(frequency * xyz), torch.cos(frequency * xyz)))
    return torch.cat(features, dim=-1)
