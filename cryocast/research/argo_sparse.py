"""Shared research scaffold for sparse Argo encoder experiments.

This module is intentionally isolated from the production ingestion and model paths.
It provides data adaptation, the current interpolation baseline, reproducible held-out
splits, metrics, retention experiments, and lightweight performance accounting.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from haversine import Unit, haversine_vector
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_LATITUDE_MIN = -90.0
_LATITUDE_MAX = 90.0
_LONGITUDE_MIN = -180.0
_LONGITUDE_MAX = 180.0
_COORDINATE_VECTOR_NDIM = 1
_METRIC_MATRIX_NDIM = 2
_MIN_SPLIT_OBSERVATIONS = 2


@dataclass(frozen=True, slots=True)
class ArgoFrameColumns:
    """Column names used when adapting a raw Argo DataFrame."""

    latitude: str = "LATITUDE"
    longitude: str = "LONGITUDE"
    pressure: str | None = "PRES"
    time: str | None = "TIME"
    metadata: tuple[str, ...] = ()


_DEFAULT_FRAME_COLUMNS = ArgoFrameColumns()


@dataclass(frozen=True, slots=True)
class SparseObservations:
    """Variable-length sparse observations for one model timestep."""

    latitudes: FloatArray
    longitudes: FloatArray
    measurements: FloatArray
    variable_names: tuple[str, ...]
    pressure: FloatArray | None = None
    time_offsets_hours: FloatArray | None = None
    metadata: Mapping[str, tuple[object, ...]] | None = None

    def __post_init__(self) -> None:
        """Validate aligned sparse-observation dimensions and coordinates."""
        n_observations = self.latitudes.shape[0]
        if self.latitudes.ndim != _COORDINATE_VECTOR_NDIM or self.longitudes.shape != (
            n_observations,
        ):
            msg = "Latitude and longitude must be one-dimensional aligned arrays."
            raise ValueError(msg)
        if self.measurements.shape != (n_observations, len(self.variable_names)):
            msg = (
                "Measurements must have shape (n_observations, n_variables) "
                "matching variable_names."
            )
            raise ValueError(msg)
        if (
            not np.isfinite(self.latitudes).all()
            or not np.isfinite(self.longitudes).all()
        ):
            msg = "Sparse observation coordinates must be finite."
            raise ValueError(msg)
        if np.any((self.latitudes < _LATITUDE_MIN) | (self.latitudes > _LATITUDE_MAX)):
            msg = "Latitudes must be within [-90, 90] degrees."
            raise ValueError(msg)
        if np.any(
            (self.longitudes < _LONGITUDE_MIN) | (self.longitudes > _LONGITUDE_MAX)
        ):
            msg = "Longitudes must be within [-180, 180] degrees."
            raise ValueError(msg)
        for optional_values in (self.pressure, self.time_offsets_hours):
            if optional_values is not None and optional_values.shape != (
                n_observations,
            ):
                msg = "Optional sparse observation arrays must align with coordinates."
                raise ValueError(msg)
        if self.metadata is not None and any(
            len(values) != n_observations for values in self.metadata.values()
        ):
            msg = "Metadata fields must align with the observation dimension."
            raise ValueError(msg)

    @property
    def count(self) -> int:
        """Return the number of sparse observations."""
        return self.latitudes.shape[0]

    @property
    def n_variables(self) -> int:
        """Return the number of observed variables."""
        return len(self.variable_names)

    def take(self, indices: IntArray) -> "SparseObservations":
        """Return a subset while preserving aligned optional fields."""
        metadata = (
            None
            if self.metadata is None
            else {
                key: tuple(values[int(index)] for index in indices)
                for key, values in self.metadata.items()
            }
        )
        return SparseObservations(
            latitudes=self.latitudes[indices],
            longitudes=self.longitudes[indices],
            measurements=self.measurements[indices],
            variable_names=self.variable_names,
            pressure=None if self.pressure is None else self.pressure[indices],
            time_offsets_hours=(
                None
                if self.time_offsets_hours is None
                else self.time_offsets_hours[indices]
            ),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class PaddedSparseBatch:
    """NumPy padded sparse observations with an explicit validity mask."""

    latitudes: FloatArray
    longitudes: FloatArray
    measurements: FloatArray
    mask: BoolArray
    variable_names: tuple[str, ...]
    pressure: FloatArray | None = None
    time_offsets_hours: FloatArray | None = None

    @property
    def lengths(self) -> IntArray:
        """Return the number of valid observations in each batch element."""
        return self.mask.sum(axis=1, dtype=np.int64)


def pad_sparse_observations(
    observations: Sequence[SparseObservations],
    *,
    fill_value: float = 0.0,
) -> PaddedSparseBatch:
    """Pad variable-length sparse samples and return a validity mask."""
    if not observations:
        msg = "At least one sparse observation sample is required."
        raise ValueError(msg)

    variable_names = observations[0].variable_names
    if any(sample.variable_names != variable_names for sample in observations):
        msg = "All sparse samples in a batch must use the same variables."
        raise ValueError(msg)

    batch_size = len(observations)
    max_observations = max(sample.count for sample in observations)
    n_variables = len(variable_names)
    shape = (batch_size, max_observations)
    latitudes = np.full(shape, fill_value, dtype=np.float64)
    longitudes = np.full(shape, fill_value, dtype=np.float64)
    measurements = np.full(
        (*shape, n_variables),
        fill_value,
        dtype=np.float64,
    )
    mask = np.zeros(shape, dtype=np.bool_)

    has_pressure = any(sample.pressure is not None for sample in observations)
    pressure = np.full(shape, np.nan, dtype=np.float64) if has_pressure else None
    has_time_offsets = any(
        sample.time_offsets_hours is not None for sample in observations
    )
    time_offsets = (
        np.full(shape, np.nan, dtype=np.float64) if has_time_offsets else None
    )

    for batch_index, sample in enumerate(observations):
        length = sample.count
        latitudes[batch_index, :length] = sample.latitudes
        longitudes[batch_index, :length] = sample.longitudes
        measurements[batch_index, :length] = sample.measurements
        mask[batch_index, :length] = True
        if pressure is not None and sample.pressure is not None:
            pressure[batch_index, :length] = sample.pressure
        if time_offsets is not None and sample.time_offsets_hours is not None:
            time_offsets[batch_index, :length] = sample.time_offsets_hours

    return PaddedSparseBatch(
        latitudes=latitudes,
        longitudes=longitudes,
        measurements=measurements,
        mask=mask,
        variable_names=variable_names,
        pressure=pressure,
        time_offsets_hours=time_offsets,
    )


def sparse_observations_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    variable_names: Sequence[str] = ("TEMP", "PSAL"),
    columns: ArgoFrameColumns = _DEFAULT_FRAME_COLUMNS,
    reference_time: pd.Timestamp | None = None,
) -> SparseObservations:
    """Convert a raw Argo DataFrame into aligned sparse observations.

    Rows with invalid coordinates or any missing requested measurement are removed.
    Pressure and selected metadata are retained when present. Time offsets are only
    derived when a reference time is supplied, avoiding assumptions about temporal
    encoding before the raw-data schema is agreed.
    """
    names = tuple(variable_names)
    required_columns = (columns.latitude, columns.longitude, *names)
    missing_columns = [name for name in required_columns if name not in dataframe]
    if missing_columns:
        msg = f"Argo DataFrame is missing required columns: {missing_columns}."
        raise ValueError(msg)

    latitudes = pd.to_numeric(dataframe[columns.latitude], errors="coerce").to_numpy(
        dtype=np.float64
    )
    longitudes = pd.to_numeric(dataframe[columns.longitude], errors="coerce").to_numpy(
        dtype=np.float64
    )
    measurements = np.column_stack(
        [
            pd.to_numeric(dataframe[name], errors="coerce").to_numpy(dtype=np.float64)
            for name in names
        ]
    )
    valid = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & np.isfinite(measurements).all(axis=1)
    )
    filtered = dataframe.loc[valid]
    latitudes = latitudes[valid]
    longitudes = longitudes[valid]
    measurements = measurements[valid]

    pressure = None
    if columns.pressure is not None and columns.pressure in filtered:
        pressure = pd.to_numeric(filtered[columns.pressure], errors="coerce").to_numpy(
            dtype=np.float64
        )

    time_offsets_hours = None
    if (
        reference_time is not None
        and columns.time is not None
        and columns.time in filtered
    ):
        timestamps = pd.to_datetime(filtered[columns.time], errors="coerce", utc=True)
        reference = pd.Timestamp(reference_time)
        reference = (
            reference.tz_localize("UTC")
            if reference.tzinfo is None
            else reference.tz_convert("UTC")
        )
        time_offsets_hours = (
            (timestamps - reference) / pd.Timedelta(hours=1)
        ).to_numpy(dtype=np.float64)

    metadata = {
        name: tuple(filtered[name].tolist())
        for name in columns.metadata
        if name in filtered
    }
    return SparseObservations(
        latitudes=latitudes,
        longitudes=longitudes,
        measurements=measurements,
        variable_names=names,
        pressure=pressure,
        time_offsets_hours=time_offsets_hours,
        metadata=metadata or None,
    )


@dataclass(frozen=True, slots=True)
class ObservationSplit:
    """Observed and held-out points for one benchmark repeat."""

    observed: SparseObservations
    held_out: SparseObservations


def split_observations(
    observations: SparseObservations,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> ObservationSplit:
    """Split one timestep into observed and held-out points reproducibly."""
    if observations.count < _MIN_SPLIT_OBSERVATIONS:
        msg = "At least two observations are required for a held-out benchmark."
        raise ValueError(msg)
    if not 0.0 < holdout_fraction < 1.0:
        msg = "holdout_fraction must be between 0 and 1."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(observations.count)
    held_out_count = round(observations.count * holdout_fraction)
    held_out_count = max(1, min(observations.count - 1, held_out_count))
    held_out_indices = np.sort(indices[:held_out_count]).astype(np.int64)
    observed_indices = np.sort(indices[held_out_count:]).astype(np.int64)
    return ObservationSplit(
        observed=observations.take(observed_indices),
        held_out=observations.take(held_out_indices),
    )


def retain_observations(
    observations: SparseObservations,
    *,
    fraction: float,
    seed: int,
) -> SparseObservations:
    """Retain a deterministic random fraction of an observed sparse set."""
    if not 0.0 < fraction <= 1.0:
        msg = "Retention fraction must be within (0, 1]."
        raise ValueError(msg)
    if fraction >= 1.0 or observations.count <= 1:
        return observations
    keep_count = max(1, round(observations.count * fraction))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(observations.count, size=keep_count, replace=False))
    return observations.take(indices.astype(np.int64))


class SparseEncoder(Protocol):
    """Minimal experimental interface for sparse-observation reconstruction."""

    @property
    def name(self) -> str:
        """Return a stable benchmark name."""
        ...

    def predict(
        self,
        observations: SparseObservations,
        query_latitudes: FloatArray,
        query_longitudes: FloatArray,
    ) -> FloatArray:
        """Predict observed variables at arbitrary query coordinates."""
        ...


@dataclass(frozen=True, slots=True)
class GaussianInterpolationBaseline:
    """Current production Argo Gaussian distance rule at arbitrary query points."""

    distance_scale_km: float = 2000.0
    min_weight: float = 1e-10

    @property
    def name(self) -> str:
        """Return the benchmark name."""
        return "current-gaussian-interpolation"

    def predict(
        self,
        observations: SparseObservations,
        query_latitudes: FloatArray,
        query_longitudes: FloatArray,
    ) -> FloatArray:
        """Interpolate sparse measurements using the production weighting rule."""
        if self.distance_scale_km <= 0:
            msg = "distance_scale_km must be greater than zero."
            raise ValueError(msg)
        if self.min_weight < 0:
            msg = "min_weight must be non-negative."
            raise ValueError(msg)

        query_latitudes = np.asarray(query_latitudes, dtype=np.float64)
        query_longitudes = np.asarray(query_longitudes, dtype=np.float64)
        if (
            query_latitudes.ndim != _COORDINATE_VECTOR_NDIM
            or query_longitudes.shape != query_latitudes.shape
        ):
            msg = "Query latitude and longitude must be aligned one-dimensional arrays."
            raise ValueError(msg)

        predictions = np.full(
            (query_latitudes.shape[0], observations.n_variables),
            np.nan,
            dtype=np.float64,
        )
        if observations.count == 0 or query_latitudes.size == 0:
            return predictions

        observation_points = np.column_stack(
            (observations.latitudes, observations.longitudes)
        )
        query_points = np.column_stack((query_latitudes, query_longitudes))
        distance_km = np.asarray(
            haversine_vector(
                observation_points,
                query_points,
                unit=Unit.KILOMETERS,
                comb=True,
                check=False,
            ),
            dtype=np.float64,
        )
        weights = np.exp(-0.5 * (distance_km / self.distance_scale_km) ** 2)
        weights = np.clip(weights, a_min=self.min_weight, a_max=None)
        sum_weights = weights.sum(axis=1)
        for variable_index in range(observations.n_variables):
            variable_measurements = observations.measurements[:, variable_index]
            predictions[:, variable_index] = np.divide(
                weights @ variable_measurements,
                sum_weights,
                out=np.full(query_latitudes.shape[0], np.nan, dtype=np.float64),
                where=sum_weights > 0,
            )
        return predictions


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Scalar reconstruction metrics for one observed variable."""

    mae: float
    rmse: float
    count: int


def compute_regression_metrics(
    predicted: FloatArray,
    target: FloatArray,
    variable_names: Sequence[str],
) -> dict[str, RegressionMetrics]:
    """Compute finite-value MAE and RMSE independently for each variable."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    names = tuple(variable_names)
    if predicted.shape != target.shape:
        msg = "Predicted and target arrays must have matching shapes."
        raise ValueError(msg)
    if predicted.ndim != _METRIC_MATRIX_NDIM or predicted.shape[1] != len(names):
        msg = "Metric arrays must have shape (n_observations, n_variables)."
        raise ValueError(msg)

    metrics: dict[str, RegressionMetrics] = {}
    for variable_index, variable_name in enumerate(names):
        prediction_values = predicted[:, variable_index]
        target_values = target[:, variable_index]
        valid = np.isfinite(prediction_values) & np.isfinite(target_values)
        count = int(valid.sum())
        if count == 0:
            metrics[variable_name] = RegressionMetrics(
                mae=float("nan"), rmse=float("nan"), count=0
            )
            continue
        differences = prediction_values[valid] - target_values[valid]
        metrics[variable_name] = RegressionMetrics(
            mae=float(np.mean(np.abs(differences))),
            rmse=float(np.sqrt(np.mean(np.square(differences)))),
            count=count,
        )
    return metrics


class BenchmarkMemoryHook(Protocol):
    """Hook for measuring accelerator memory around one encoder invocation."""

    def start(self) -> None:
        """Reset memory accounting before inference."""
        ...

    def stop(self) -> int | None:
        """Return peak allocated bytes, or None when unavailable."""
        ...


@dataclass(slots=True)
class TorchCudaMemoryHook:
    """CUDA peak-memory hook for torch-based sparse encoders."""

    device: torch.device | str | None = None

    def start(self) -> None:
        """Reset CUDA peak-memory statistics when CUDA is available."""
        device = self._device()
        if device is None:
            return
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def stop(self) -> int | None:
        """Return CUDA peak allocated memory in bytes when available."""
        device = self._device()
        if device is None:
            return None
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))

    def _device(self) -> torch.device | None:
        device = torch.device("cuda" if self.device is None else self.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        return device


@dataclass(frozen=True, slots=True)
class BenchmarkRepeat:
    """Metrics and performance measurements for one held-out split."""

    repeat: int
    n_observed: int
    n_held_out: int
    elapsed_seconds: float
    peak_memory_bytes: int | None
    metrics: Mapping[str, RegressionMetrics]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Repeated held-out reconstruction results for one encoder."""

    encoder_name: str
    repeats: tuple[BenchmarkRepeat, ...]


def benchmark_encoder(
    encoder: SparseEncoder,
    observations: SparseObservations,
    *,
    holdout_fraction: float = 0.2,
    repeats: int = 5,
    seed: int = 0,
    memory_hook: BenchmarkMemoryHook | None = None,
) -> BenchmarkResult:
    """Benchmark an encoder using repeated reproducible held-out point splits.

    Held-out reconstruction is an information-preservation proxy only. It does not
    establish downstream sea-ice forecast skill.
    """
    if repeats <= 0:
        msg = "repeats must be greater than zero."
        raise ValueError(msg)

    results: list[BenchmarkRepeat] = []
    for repeat in range(repeats):
        split = split_observations(
            observations,
            holdout_fraction=holdout_fraction,
            seed=seed + repeat,
        )
        if memory_hook is not None:
            memory_hook.start()
        start_time = perf_counter()
        predicted = encoder.predict(
            split.observed,
            split.held_out.latitudes,
            split.held_out.longitudes,
        )
        elapsed_seconds = perf_counter() - start_time
        peak_memory_bytes = None if memory_hook is None else memory_hook.stop()
        results.append(
            BenchmarkRepeat(
                repeat=repeat,
                n_observed=split.observed.count,
                n_held_out=split.held_out.count,
                elapsed_seconds=elapsed_seconds,
                peak_memory_bytes=peak_memory_bytes,
                metrics=compute_regression_metrics(
                    predicted,
                    split.held_out.measurements,
                    split.held_out.variable_names,
                ),
            )
        )
    return BenchmarkResult(encoder_name=encoder.name, repeats=tuple(results))


def benchmark_retention_levels(  # noqa: PLR0913
    encoder: SparseEncoder,
    observations: SparseObservations,
    *,
    retention_fractions: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    holdout_fraction: float = 0.2,
    repeats: int = 5,
    seed: int = 0,
    memory_hook: BenchmarkMemoryHook | None = None,
) -> dict[float, BenchmarkResult]:
    """Benchmark fixed held-out splits while dropping observed points.

    Each retention level reuses the same outer held-out split for a repeat so the
    fractions are directly comparable.
    """
    fractions = tuple(float(fraction) for fraction in retention_fractions)
    if not fractions or any(
        fraction <= 0.0 or fraction > 1.0 for fraction in fractions
    ):
        msg = "Retention fractions must be within (0, 1]."
        raise ValueError(msg)
    if repeats <= 0:
        msg = "repeats must be greater than zero."
        raise ValueError(msg)

    results: dict[float, BenchmarkResult] = {}
    for fraction in fractions:
        repeat_results: list[BenchmarkRepeat] = []
        for repeat in range(repeats):
            split = split_observations(
                observations,
                holdout_fraction=holdout_fraction,
                seed=seed + repeat,
            )
            observed = retain_observations(
                split.observed,
                fraction=fraction,
                seed=seed + 10_000 + repeat,
            )
            if memory_hook is not None:
                memory_hook.start()
            start_time = perf_counter()
            predicted = encoder.predict(
                observed,
                split.held_out.latitudes,
                split.held_out.longitudes,
            )
            elapsed_seconds = perf_counter() - start_time
            peak_memory_bytes = None if memory_hook is None else memory_hook.stop()
            repeat_results.append(
                BenchmarkRepeat(
                    repeat=repeat,
                    n_observed=observed.count,
                    n_held_out=split.held_out.count,
                    elapsed_seconds=elapsed_seconds,
                    peak_memory_bytes=peak_memory_bytes,
                    metrics=compute_regression_metrics(
                        predicted,
                        split.held_out.measurements,
                        split.held_out.variable_names,
                    ),
                )
            )
        results[fraction] = BenchmarkResult(
            encoder_name=f"{encoder.name}@{fraction:.0%}",
            repeats=tuple(repeat_results),
        )
    return results


def count_trainable_parameters(module: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a torch module."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
