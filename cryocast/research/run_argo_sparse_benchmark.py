"""Run the issue #477 symmetric sparse Argo reconstruction benchmark."""

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cryocast.geotools import grid_factory
from cryocast.ingestion.sources.argo import _fetch_argo_dataframe_with_retry
from cryocast.research.argo_cross_attention import (
    SparseCrossAttentionConfig,
    SparseCrossAttentionEncoder,
)
from cryocast.research.argo_profile_split import profile_group_count
from cryocast.research.argo_reconstruction import (
    EvaluationRecord,
    MeasurementNormaliser,
    ReconstructionFitSummary,
    ReconstructionTrainingConfig,
    SparseReconstructionModel,
    build_evaluation_cases,
    evaluate_interpolation,
    evaluate_reconstruction_model,
    fit_reconstruction_model,
    parameter_counts,
    summarise_records,
)
from cryocast.research.argo_setconv import SparseSetConvConfig, SparseSetConvEncoder
from cryocast.research.argo_sparse import (
    ArgoFrameColumns,
    SparseObservations,
    sparse_observations_from_dataframe,
)

_FULL_GRID_SHAPE = (432, 432)
_LATENT_STRIDE = 3
_DEFAULT_RETENTION = (1.0, 0.75, 0.5, 0.25)
_VARIABLES = ("TEMP", "PSAL")
_MIN_TIMESTEP_OBSERVATIONS = 2
_METADATA_NAMES = {
    "CYCLE_NUMBER",
    "DATA_MODE",
    "DIRECTION",
    "N_PROF",
    "PLATFORM_NUMBER",
    "POSITION_QC",
    "PRES",
    "TIME",
    "TIME_QC",
}


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO timestamp and return naive UTC for the Argo fetch helper."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


def _parse_area(value: str) -> tuple[float, float, float, float]:
    """Parse the repository's N/W/S/E area convention."""
    north, west, south, east = map(float, value.split("/"))
    if north < south or east < west:
        msg = "Area must use N/W/S/E ordering with north>=south and east>=west."
        raise ValueError(msg)
    return north, west, south, east


def _metadata_columns(columns: Iterable[object]) -> tuple[str, ...]:
    """Return potentially useful raw Argo metadata/QC fields for reporting."""
    names = {str(column) for column in columns}
    return tuple(
        sorted(
            name for name in names if name in _METADATA_NAMES or name.endswith("_QC")
        )
    )


def _fetch_sparse_timestep(
    date: datetime,
    *,
    area: tuple[float, float, float, float],
    depth_max_m: float,
    half_window_hours: float,
    max_observations: int,
    seed: int,
) -> tuple[SparseObservations, tuple[str, ...]]:
    """Fetch one raw Argo timestep and deterministically cap its point count."""
    north, west, south, east = area
    region = [west, east, south, north, 0.0, depth_max_m]
    half_window = timedelta(hours=half_window_hours)
    dataframe = _fetch_argo_dataframe_with_retry(
        region,
        [date - half_window, date + half_window],
        max_attempts=4,
        initial_backoff_s=0.5,
    )
    metadata = _metadata_columns(dataframe.columns)
    observations = sparse_observations_from_dataframe(
        dataframe,
        variable_names=_VARIABLES,
        columns=ArgoFrameColumns(metadata=metadata),
        reference_time=pd.Timestamp(date),
    )
    if observations.count < _MIN_TIMESTEP_OBSERVATIONS:
        msg = f"Only {observations.count} usable Argo observations for {date.isoformat()}."
        raise RuntimeError(msg)
    if observations.count <= max_observations:
        return observations, metadata
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(observations.count, size=max_observations, replace=False)
    ).astype(np.int64)
    return observations.take(indices), metadata


def _latent_grid() -> tuple[np.ndarray, np.ndarray]:
    """Build the 144x144 latent geography used by the CNN baseline configuration."""
    geography = grid_factory.create(
        "EPSG:6931",
        resolution="25p0km",
        shape=_FULL_GRID_SHAPE,
    )
    latitudes = np.asarray(geography.latitudes(), dtype=np.float64)
    longitudes = np.asarray(geography.longitudes(), dtype=np.float64)
    return (
        latitudes[::_LATENT_STRIDE, ::_LATENT_STRIDE],
        longitudes[::_LATENT_STRIDE, ::_LATENT_STRIDE],
    )


def _build_models(
    seed: int,
    latent_latitudes: np.ndarray,
    latent_longitudes: np.ndarray,
) -> tuple[SparseReconstructionModel, SparseReconstructionModel]:
    """Construct both candidates with the same latent width and readout seed."""
    torch.manual_seed(seed)
    setconv = SparseReconstructionModel(
        SparseSetConvEncoder(
            SparseSetConvConfig(),
            latent_latitudes,
            latent_longitudes,
        ),
        readout_seed=seed,
    )
    torch.manual_seed(seed)
    cross_attention = SparseReconstructionModel(
        SparseCrossAttentionEncoder(
            SparseCrossAttentionConfig(),
            latent_latitudes,
            latent_longitudes,
        ),
        readout_seed=seed,
    )
    return setconv, cross_attention


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark raw sparse Argo encoders for CryoCast issue #477."
    )
    parser.add_argument("--train-date", action="append", required=True)
    parser.add_argument("--eval-date", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=477)
    parser.add_argument("--area", default="90/-180/0/180")
    parser.add_argument("--depth-max-m", type=float, default=50.0)
    parser.add_argument("--half-window-hours", type=float, default=2.0)
    parser.add_argument("--max-observations", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _normalisation_payload(
    normaliser: MeasurementNormaliser,
) -> dict[str, dict[str, float]]:
    return {
        variable: {
            "mean": float(normaliser.mean[index]),
            "scale": float(normaliser.scale[index]),
        }
        for index, variable in enumerate(normaliser.variable_names)
    }


def main() -> None:
    """Fetch real Argo samples, fit both candidates fairly, and emit JSON results."""
    args = _build_parser().parse_args()
    area = _parse_area(args.area)
    train_dates = [_parse_datetime(value) for value in args.train_date]
    eval_dates = [_parse_datetime(value) for value in args.eval_date]
    seeds = tuple(dict.fromkeys(args.seed))
    if set(train_dates) & set(eval_dates):
        msg = "Training and evaluation dates must be disjoint."
        raise ValueError(msg)

    train_fetched = [
        _fetch_sparse_timestep(
            date,
            area=area,
            depth_max_m=args.depth_max_m,
            half_window_hours=args.half_window_hours,
            max_observations=args.max_observations,
            seed=args.evaluation_seed + index,
        )
        for index, date in enumerate(train_dates)
    ]
    eval_fetched = [
        _fetch_sparse_timestep(
            date,
            area=area,
            depth_max_m=args.depth_max_m,
            half_window_hours=args.half_window_hours,
            max_observations=args.max_observations,
            seed=args.evaluation_seed + 10_000 + index,
        )
        for index, date in enumerate(eval_dates)
    ]
    train_samples = [item[0] for item in train_fetched]
    eval_samples = [item[0] for item in eval_fetched]
    metadata_columns = sorted(
        {name for _, columns in (*train_fetched, *eval_fetched) for name in columns}
    )

    normaliser = MeasurementNormaliser.fit(train_samples)
    normalised_training = [normaliser.normalise(sample) for sample in train_samples]
    cases = build_evaluation_cases(
        eval_samples,
        retention_fractions=_DEFAULT_RETENTION,
        repeats=args.repeats,
        seed=args.evaluation_seed,
    )
    interpolation_records = evaluate_interpolation(cases)

    latent_latitudes, latent_longitudes = _latent_grid()
    training_config = ReconstructionTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learned_records: dict[str, list[EvaluationRecord]] = {
        "sparse-setconv": [],
        "sparse-cross-attention": [],
    }
    fit_summaries: dict[str, list[ReconstructionFitSummary]] = {
        "sparse-setconv": [],
        "sparse-cross-attention": [],
    }
    parameters: dict[str, dict[str, int]] = {}

    for seed in seeds:
        setconv, cross_attention = _build_models(
            seed,
            latent_latitudes,
            latent_longitudes,
        )
        for model in (setconv, cross_attention):
            fit_summary = fit_reconstruction_model(
                model,
                normalised_training,
                config=training_config,
                seed=seed,
                device=device,
            )
            records = evaluate_reconstruction_model(
                model,
                cases,
                normaliser=normaliser,
                training_seed=seed,
                device=device,
            )
            learned_records[model.name].extend(records)
            fit_summaries[model.name].append(fit_summary)
            parameters.setdefault(model.name, parameter_counts(model))

    payload = {
        "settings": {
            "train_dates": [date.isoformat() for date in train_dates],
            "eval_dates": [date.isoformat() for date in eval_dates],
            "training_seeds": list(seeds),
            "evaluation_seed": args.evaluation_seed,
            "area": args.area,
            "depth_max_m": args.depth_max_m,
            "half_window_hours": args.half_window_hours,
            "max_observations": args.max_observations,
            "epochs": args.epochs,
            "repeats": args.repeats,
            "learning_rate": args.learning_rate,
            "retention_fractions": list(_DEFAULT_RETENTION),
            "device": str(device),
        },
        "observation_counts": {
            "train": [sample.count for sample in train_samples],
            "eval": [sample.count for sample in eval_samples],
        },
        "profile_group_counts": {
            "train": [profile_group_count(sample) for sample in train_samples],
            "eval": [profile_group_count(sample) for sample in eval_samples],
        },
        "split_policy": (
            "Complete PLATFORM_NUMBER/CYCLE_NUMBER profiles are kept together when "
            "those identifiers are available; synthetic or incomplete samples fall "
            "back to deterministic row-wise splitting."
        ),
        "normalisation": _normalisation_payload(normaliser),
        "raw_metadata_columns_seen": metadata_columns,
        "metadata_policy": (
            "The architecture comparison uses coordinates plus TEMP/PSAL only. "
            "Pressure, time, profile identifiers, and QC fields are reported but not "
            "fed to either model until their semantics/filtering are agreed for #477."
        ),
        "parameters": parameters,
        "training": {
            candidate: [asdict(summary) for summary in summaries]
            for candidate, summaries in fit_summaries.items()
        },
        "results": {
            "interpolation": [
                asdict(summary) for summary in summarise_records(interpolation_records)
            ],
            **{
                candidate: [
                    asdict(summary) for summary in summarise_records(candidate_records)
                ]
                for candidate, candidate_records in learned_records.items()
            },
        },
        "interpretation": (
            "Held-out Argo reconstruction is an information-preservation proxy only. "
            "Any selected encoder still requires a downstream IceNet experiment "
            "comparing no Argo, existing interpolated Argo, and direct sparse Argo "
            "under identical forecast training and evaluation settings."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    sys.stdout.write(rendered + "\n")
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
