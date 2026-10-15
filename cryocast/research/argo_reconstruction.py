"""Symmetric reconstruction benchmark for experimental sparse Argo encoders."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .argo_cross_attention import SparseCrossAttentionEncoder
from .argo_profile_split import split_observations
from .argo_setconv import SparseSetConvEncoder
from .argo_sparse import (
    FloatArray,
    GaussianInterpolationBaseline,
    RegressionMetrics,
    SparseObservations,
    TorchCudaMemoryHook,
    compute_regression_metrics,
    count_trainable_parameters,
    retain_observations,
)
from .argo_torch import (
    TorchSparseSequenceBatch,
    torch_sparse_sequence_from_observations,
)

_MIN_SCALE = 1e-6
_MIN_TRAINING_POINTS = 2
_RETENTION_SEED_OFFSET = 1_000_000
_TRAINING_EPOCH_SEED_STRIDE = 10_000

SparseQueryEncoder = SparseSetConvEncoder | SparseCrossAttentionEncoder


@dataclass(frozen=True, slots=True)
class MeasurementNormaliser:
    """Training-only affine normalisation for sparse measurement channels."""

    variable_names: tuple[str, ...]
    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, observations: Sequence[SparseObservations]) -> "MeasurementNormaliser":
        """Fit channel statistics using training observations only."""
        if not observations:
            msg = "At least one training observation sample is required."
            raise ValueError(msg)
        variable_names = observations[0].variable_names
        if any(sample.variable_names != variable_names for sample in observations):
            msg = "All training samples must use the same variable ordering."
            raise ValueError(msg)
        nonempty_measurements = [
            sample.measurements for sample in observations if sample.count > 0
        ]
        if not nonempty_measurements:
            msg = "Training samples must contain at least one valid observation."
            raise ValueError(msg)
        measurements = np.concatenate(nonempty_measurements, axis=0)
        mean = np.mean(measurements, axis=0, dtype=np.float64)
        scale = np.std(measurements, axis=0, dtype=np.float64)
        scale = np.where(scale >= _MIN_SCALE, scale, 1.0).astype(np.float64)
        return cls(
            variable_names=variable_names,
            mean=np.asarray(mean, dtype=np.float64),
            scale=scale,
        )

    def normalise(self, observations: SparseObservations) -> SparseObservations:
        """Return a sample with measurement channels normalised."""
        self._validate_variables(observations)
        return replace(
            observations,
            measurements=(observations.measurements - self.mean) / self.scale,
        )

    def denormalise(self, measurements: FloatArray) -> FloatArray:
        """Convert normalised model predictions back to physical units."""
        values = np.asarray(measurements, dtype=np.float64)
        if values.shape[-1] != len(self.variable_names):
            msg = "Prediction channels must match the fitted normaliser."
            raise ValueError(msg)
        return values * self.scale + self.mean

    def _validate_variables(self, observations: SparseObservations) -> None:
        if observations.variable_names != self.variable_names:
            msg = "Sparse observation variables must match the fitted normaliser."
            raise ValueError(msg)


class SparseReconstructionModel(nn.Module):
    """Attach the same pointwise reconstruction head to either sparse encoder."""

    def __init__(self, encoder: SparseQueryEncoder, *, readout_seed: int) -> None:
        """Initialise a common linear readout from candidate query latents."""
        super().__init__()
        self.encoder = encoder
        self.readout = nn.Linear(
            encoder.latent_channels,
            len(encoder.variable_names),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(readout_seed)
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    @property
    def name(self) -> str:
        """Return the underlying sparse encoder name."""
        return self.encoder.name

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return ordered measurement channels predicted by the common head."""
        return self.encoder.variable_names

    def forward(
        self,
        batch: TorchSparseSequenceBatch,
        query_latitudes: torch.Tensor,
        query_longitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Predict normalised measurements at arbitrary query coordinates."""
        latent = self.encoder.encode_queries(
            batch,
            query_latitudes,
            query_longitudes,
        )
        return self.readout(latent)


@dataclass(frozen=True, slots=True)
class ReconstructionTrainingConfig:
    """Shared optimisation settings for both learned reconstruction probes."""

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    holdout_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate common optimisation settings."""
        if self.epochs <= 0:
            msg = "epochs must be greater than zero."
            raise ValueError(msg)
        if self.learning_rate <= 0 or self.weight_decay < 0:
            msg = "Optimiser settings must be non-negative and use a positive rate."
            raise ValueError(msg)
        if not 0.0 < self.holdout_fraction < 1.0:
            msg = "holdout_fraction must be between zero and one."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ReconstructionFitSummary:
    """Training cost and final normalised reconstruction loss."""

    seed: int
    epochs: int
    examples: int
    training_seconds: float
    final_loss: float


def fit_reconstruction_model(
    model: SparseReconstructionModel,
    training_observations: Sequence[SparseObservations],
    *,
    config: ReconstructionTrainingConfig,
    seed: int,
    device: torch.device | str = "cpu",
) -> ReconstructionFitSummary:
    """Train one candidate using the common masked-reconstruction schedule.

    Calling this function for both candidates with the same training samples, config,
    and seed produces identical held-out point splits at every epoch.
    """
    eligible = [
        sample
        for sample in training_observations
        if sample.count >= _MIN_TRAINING_POINTS
    ]
    if not eligible:
        msg = "Training samples must contain at least two valid observations."
        raise ValueError(msg)
    if any(sample.variable_names != model.variable_names for sample in eligible):
        msg = "Training sample variables must match the reconstruction model."
        raise ValueError(msg)

    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_time = perf_counter()
    final_loss = float("nan")
    for epoch in range(config.epochs):
        losses: list[torch.Tensor] = []
        for sample_index, observations in enumerate(eligible):
            split = split_observations(
                observations,
                holdout_fraction=config.holdout_fraction,
                seed=seed + epoch * _TRAINING_EPOCH_SEED_STRIDE + sample_index,
            )
            batch = torch_sparse_sequence_from_observations(
                [[split.observed]],
                device=device,
            )
            query_latitudes = torch.as_tensor(
                split.held_out.latitudes,
                dtype=torch.float32,
                device=device,
            )
            query_longitudes = torch.as_tensor(
                split.held_out.longitudes,
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(
                split.held_out.measurements,
                dtype=torch.float32,
                device=device,
            )
            prediction = model(batch, query_latitudes, query_longitudes)[0, 0]
            losses.append(torch.mean(torch.square(prediction - target)))
        loss = torch.stack(losses).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    return ReconstructionFitSummary(
        seed=seed,
        epochs=config.epochs,
        examples=len(eligible),
        training_seconds=perf_counter() - start_time,
        final_loss=final_loss,
    )


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One fixed held-out split and observation-retention condition."""

    sample_index: int
    repeat: int
    retention: float
    observed: SparseObservations
    held_out: SparseObservations


def build_evaluation_cases(
    observations: Sequence[SparseObservations],
    *,
    retention_fractions: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    holdout_fraction: float = 0.2,
    repeats: int = 3,
    seed: int = 0,
) -> tuple[EvaluationCase, ...]:
    """Build fixed evaluation cases reused by every candidate and training seed."""
    if repeats <= 0:
        msg = "repeats must be greater than zero."
        raise ValueError(msg)
    fractions = tuple(float(fraction) for fraction in retention_fractions)
    if not fractions or any(
        fraction <= 0.0 or fraction > 1.0 for fraction in fractions
    ):
        msg = "Retention fractions must be within (0, 1]."
        raise ValueError(msg)

    cases: list[EvaluationCase] = []
    for sample_index, sample in enumerate(observations):
        for repeat in range(repeats):
            split_seed = seed + sample_index * _TRAINING_EPOCH_SEED_STRIDE + repeat
            split = split_observations(
                sample,
                holdout_fraction=holdout_fraction,
                seed=split_seed,
            )
            for retention_index, retention in enumerate(fractions):
                observed = retain_observations(
                    split.observed,
                    fraction=retention,
                    seed=split_seed + _RETENTION_SEED_OFFSET + retention_index,
                )
                cases.append(
                    EvaluationCase(
                        sample_index=sample_index,
                        repeat=repeat,
                        retention=retention,
                        observed=observed,
                        held_out=split.held_out,
                    )
                )
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Physical-unit reconstruction metrics for one fixed evaluation case."""

    candidate: str
    training_seed: int | None
    sample_index: int
    repeat: int
    retention: float
    n_observed: int
    n_held_out: int
    elapsed_seconds: float
    peak_memory_bytes: int | None
    metrics: Mapping[str, RegressionMetrics]


def evaluate_interpolation(
    cases: Sequence[EvaluationCase],
    *,
    baseline: GaussianInterpolationBaseline | None = None,
) -> tuple[EvaluationRecord, ...]:
    """Evaluate the current non-learned Gaussian interpolation control."""
    encoder = GaussianInterpolationBaseline() if baseline is None else baseline
    records: list[EvaluationRecord] = []
    for case in cases:
        start_time = perf_counter()
        prediction = encoder.predict(
            case.observed,
            case.held_out.latitudes,
            case.held_out.longitudes,
        )
        elapsed_seconds = perf_counter() - start_time
        records.append(
            EvaluationRecord(
                candidate="interpolation",
                training_seed=None,
                sample_index=case.sample_index,
                repeat=case.repeat,
                retention=case.retention,
                n_observed=case.observed.count,
                n_held_out=case.held_out.count,
                elapsed_seconds=elapsed_seconds,
                peak_memory_bytes=None,
                metrics=compute_regression_metrics(
                    prediction,
                    case.held_out.measurements,
                    case.held_out.variable_names,
                ),
            )
        )
    return tuple(records)


def evaluate_reconstruction_model(
    model: SparseReconstructionModel,
    cases: Sequence[EvaluationCase],
    *,
    normaliser: MeasurementNormaliser,
    training_seed: int,
    device: torch.device | str = "cpu",
) -> tuple[EvaluationRecord, ...]:
    """Evaluate one learned encoder using exactly the supplied fixed cases."""
    model.to(device)
    model.eval()
    memory_hook = TorchCudaMemoryHook(device)
    records: list[EvaluationRecord] = []
    for case in cases:
        observed = normaliser.normalise(case.observed)
        batch = torch_sparse_sequence_from_observations([[observed]], device=device)
        query_latitudes = torch.as_tensor(
            case.held_out.latitudes,
            dtype=torch.float32,
            device=device,
        )
        query_longitudes = torch.as_tensor(
            case.held_out.longitudes,
            dtype=torch.float32,
            device=device,
        )
        memory_hook.start()
        _synchronise_device(device)
        start_time = perf_counter()
        with torch.no_grad():
            prediction = model(batch, query_latitudes, query_longitudes)[0, 0]
        _synchronise_device(device)
        elapsed_seconds = perf_counter() - start_time
        peak_memory_bytes = memory_hook.stop()
        prediction_physical = normaliser.denormalise(
            prediction.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        records.append(
            EvaluationRecord(
                candidate=model.name,
                training_seed=training_seed,
                sample_index=case.sample_index,
                repeat=case.repeat,
                retention=case.retention,
                n_observed=case.observed.count,
                n_held_out=case.held_out.count,
                elapsed_seconds=elapsed_seconds,
                peak_memory_bytes=peak_memory_bytes,
                metrics=compute_regression_metrics(
                    prediction_physical,
                    case.held_out.measurements,
                    case.held_out.variable_names,
                ),
            )
        )
    return tuple(records)


def _synchronise_device(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


@dataclass(frozen=True, slots=True)
class ScalarSummary:
    """Mean and population standard deviation for one benchmark quantity."""

    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class RetentionSummary:
    """Aggregate metrics for one candidate at one observation-retention level."""

    retention: float
    temp_mae: ScalarSummary
    temp_rmse: ScalarSummary
    psal_mae: ScalarSummary
    psal_rmse: ScalarSummary
    runtime_seconds: ScalarSummary
    peak_memory_bytes: int | None
    evaluations: int


def summarise_records(
    records: Sequence[EvaluationRecord],
) -> tuple[RetentionSummary, ...]:
    """Aggregate evaluation cases by retention level without selecting a best seed."""
    grouped: dict[float, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.retention].append(record)

    summaries: list[RetentionSummary] = []
    for retention in sorted(grouped, reverse=True):
        retained_records = grouped[retention]
        peak_values = [
            record.peak_memory_bytes
            for record in retained_records
            if record.peak_memory_bytes is not None
        ]
        summaries.append(
            RetentionSummary(
                retention=retention,
                temp_mae=_metric_summary(retained_records, "TEMP", "mae"),
                temp_rmse=_metric_summary(retained_records, "TEMP", "rmse"),
                psal_mae=_metric_summary(retained_records, "PSAL", "mae"),
                psal_rmse=_metric_summary(retained_records, "PSAL", "rmse"),
                runtime_seconds=_scalar_summary(
                    [record.elapsed_seconds for record in retained_records]
                ),
                peak_memory_bytes=max(peak_values) if peak_values else None,
                evaluations=len(retained_records),
            )
        )
    return tuple(summaries)


def _metric_summary(
    records: Sequence[EvaluationRecord],
    variable_name: str,
    metric_name: str,
) -> ScalarSummary:
    values = [
        float(getattr(record.metrics[variable_name], metric_name)) for record in records
    ]
    return _scalar_summary(values)


def _scalar_summary(values: Sequence[float]) -> ScalarSummary:
    array = np.asarray(values, dtype=np.float64)
    return ScalarSummary(
        mean=float(np.mean(array)),
        std=float(np.std(array)),
    )


def parameter_counts(model: SparseReconstructionModel) -> dict[str, int]:
    """Return encoder-only and encoder-plus-common-readout parameter counts."""
    return {
        "encoder": count_trainable_parameters(model.encoder),
        "with_common_readout": count_trainable_parameters(model),
        "common_readout": count_trainable_parameters(model.readout),
    }
