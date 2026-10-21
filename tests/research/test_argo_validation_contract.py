"""Leakage, failure-accounting and padded-observation regression cases."""

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from cryocast.research.argo_cross_attention import (
    SparseCrossAttentionConfig,
    SparseCrossAttentionEncoder,
)
from cryocast.research.argo_profile_split import (
    profiles_overlap,
    split_observations,
    validate_profile_partitions,
)
from cryocast.research.argo_setconv import SparseSetConvConfig, SparseSetConvEncoder
from cryocast.research.argo_sparse import (
    GaussianInterpolationBaseline,
    SparseObservations,
    compute_regression_metrics,
)
from cryocast.research.argo_torch import TorchSparseSequenceBatch
from cryocast.research.run_argo_sparse_benchmark import _validate_date_windows


def _observations() -> SparseObservations:
    """Return three float cycles, each with two pressure levels."""
    return SparseObservations(
        latitudes=np.array([70.0, 70.0, 72.0, 72.0, 74.0, 74.0]),
        longitudes=np.array([10.0, 10.0, 20.0, 20.0, 30.0, 30.0]),
        measurements=np.arange(12, dtype=np.float64).reshape(6, 2),
        variable_names=("TEMP", "PSAL"),
        metadata={
            "PLATFORM_NUMBER": ("A", "A", "B", "B", "C", "C"),
            "CYCLE_NUMBER": (1, 1, 2, 2, 3, 3),
        },
    )


@pytest.mark.parametrize("seed", [0, 477, 478])
def test_single_known_profile_never_falls_back_to_rows(seed: int) -> None:
    """A lone profile cannot provide an independent observed/held-out split."""
    sample = _observations().take(np.array([0, 1]))
    with pytest.raises(ValueError, match=r"two.*groups"):
        split_observations(sample, seed=seed)


@pytest.mark.parametrize("bad_id", [None, np.nan, np.inf, "", " ", pd.NA])
def test_incomplete_profile_identifiers_do_not_silently_allow_leakage(
    bad_id: object,
) -> None:
    """Partial profile metadata cannot justify a row-wise evaluation split."""
    sample = replace(
        _observations(),
        metadata={
            "PLATFORM_NUMBER": (bad_id, "A", "B", "B", "C", "C"),
            "CYCLE_NUMBER": (1, 1, 2, 2, 3, 3),
        },
    )
    with pytest.raises(ValueError, match="profile identifiers"):
        split_observations(sample)


@pytest.mark.parametrize("missing_field", ["PLATFORM_NUMBER", "CYCLE_NUMBER"])
def test_partially_available_profile_schema_is_rejected(missing_field: str) -> None:
    """Require both grouping columns when either profile column is supplied."""
    sample = _observations()
    assert sample.metadata is not None
    sample = replace(
        sample,
        metadata={k: v for k, v in sample.metadata.items() if k != missing_field},
    )
    with pytest.raises(ValueError, match="profile identifiers"):
        split_observations(sample)


@pytest.mark.parametrize("seed", [0, 477, 478])
def test_complete_cycle_groups_keep_both_directions_together(seed: int) -> None:
    """Retain the historical conservative grouping, including directional casts."""
    sample = _observations()
    assert sample.metadata is not None
    sample = replace(sample, metadata={**sample.metadata, "DIRECTION": ("A", "D") * 3})
    split = split_observations(sample, seed=seed)
    assert profiles_overlap(split.observed, split.held_out) == set()
    assert split.observed.count + split.held_out.count == 6


@pytest.mark.parametrize("bad_prediction", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_cannot_improve_score_by_exclusion(
    bad_prediction: float,
) -> None:
    """A failed prediction on a known target must not shrink its denominator."""
    predicted = np.array([[0.0, 1.0], [bad_prediction, 4.0]])
    target = np.array([[0.0, 1.0], [100.0, 5.0]])
    with pytest.raises(ValueError, match="Non-finite prediction"):
        compute_regression_metrics(predicted, target, ("TEMP", "PSAL"))


def test_missing_truth_remains_distinct_from_failed_prediction() -> None:
    """Exclude only unavailable truth, independently for each physical variable."""
    predicted = np.array([[1.0, 2.0], [np.nan, 5.0]])
    target = np.array([[0.0, 1.0], [np.nan, 1.0]])
    metrics = compute_regression_metrics(predicted, target, ("TEMP", "PSAL"))
    assert metrics["TEMP"].count == 1
    assert metrics["TEMP"].mae == 1.0
    assert metrics["PSAL"].count == 2
    assert metrics["PSAL"].mae == 2.5
    assert metrics["PSAL"].rmse == pytest.approx(np.sqrt(8.5))


def _encoder(kind: str) -> SparseSetConvEncoder | SparseCrossAttentionEncoder:
    """Use small learned encoders with non-coincident query coordinates."""
    latitudes = np.array([[71.0, 73.0]])
    longitudes = np.array([[11.0, 21.0]])
    if kind == "setconv":
        return SparseSetConvEncoder(
            SparseSetConvConfig(latent_channels=4, hidden_channels=8),
            latitudes,
            longitudes,
        )
    return SparseCrossAttentionEncoder(
        SparseCrossAttentionConfig(
            latent_channels=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
        ),
        latitudes,
        longitudes,
    )


def _batch(fill: float) -> TorchSparseSequenceBatch:
    """Only the first two observations are present; padding may be arbitrary."""
    return TorchSparseSequenceBatch(
        latitudes=torch.tensor([[[70.0, 72.0, fill]]], requires_grad=True),
        longitudes=torch.tensor([[[10.0, 20.0, fill]]], requires_grad=True),
        measurements=torch.tensor(
            [[[[1.0, 32.0], [2.0, 33.0], [fill, fill]]]], requires_grad=True
        ),
        mask=torch.tensor([[[True, True, False]]]),
        variable_names=("TEMP", "PSAL"),
    )


@pytest.mark.parametrize("kind", ["setconv", "attention"])
@pytest.mark.parametrize("fill", [float("nan"), float("inf"), -float("inf")])
def test_masked_nonfinite_padding_cannot_poison_values_or_gradients(
    kind: str,
    fill: float,
) -> None:
    """Mask coordinates before nonlinear geography, including the backward path."""
    torch.manual_seed(513)
    encoder = _encoder(kind).eval()
    padded = _batch(fill)
    reference = encoder(_batch(0.0))
    result = encoder(padded)
    torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-6)
    result.square().mean().backward()
    for tensor in (padded.latitudes, padded.longitudes, padded.measurements):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad[:, :, -1]) == 0
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("kind", ["setconv", "attention"])
def test_zero_queries_preserve_empty_output_shape(kind: str) -> None:
    """An empty query set should not fail in a chunk concatenation."""
    result = _encoder(kind).encode_queries(_batch(0.0), torch.empty(0), torch.empty(0))
    assert result.shape == (1, 1, 0, 4)


def test_gaussian_control_matches_independent_spherical_distance_formula() -> None:
    """Verify the control without reusing its haversine implementation."""
    sample = _observations()
    query_lat = np.array([71.0, 73.0, 80.0])
    query_lon = np.array([11.0, 21.0, -179.9])
    phi = np.deg2rad(sample.latitudes)
    lam = np.deg2rad(sample.longitudes)
    q_phi = np.deg2rad(query_lat)[:, None]
    q_lam = np.deg2rad(query_lon)[:, None]
    hav = (
        np.sin((q_phi - phi) / 2) ** 2
        + np.cos(q_phi) * np.cos(phi) * np.sin((q_lam - lam) / 2) ** 2
    )
    distances = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    weights = np.maximum(np.exp(-0.5 * (distances / 2000.0) ** 2), 1e-10)
    expected = weights @ sample.measurements / weights.sum(axis=1, keepdims=True)
    actual = GaussianInterpolationBaseline().predict(sample, query_lat, query_lon)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("offset_hours", [0, 1, 4])
def test_real_benchmark_rejects_overlapping_fetch_windows(offset_hours: int) -> None:
    """Different centre timestamps do not imply different raw observations."""
    origin = datetime(2024, 6, 17, 12)
    with pytest.raises(ValueError, match="observation windows must be disjoint"):
        _validate_date_windows([origin], [origin + timedelta(hours=offset_hours)], 2.0)


def test_real_benchmark_allows_separated_fetch_windows() -> None:
    """The historical weekly date schedule remains eligible."""
    origin = datetime(2024, 6, 17, 12)
    _validate_date_windows([origin], [origin + timedelta(days=7)], 2.0)


def test_real_benchmark_rejects_shared_training_evaluation_profiles() -> None:
    """The same float-cycle group cannot reappear on both date partitions."""
    with pytest.raises(ValueError, match="float-cycle groups must be disjoint"):
        validate_profile_partitions([_observations()], [_observations()])


def test_real_benchmark_requires_identifiable_profiles() -> None:
    """Only generic fixtures may use the metadata-free row-split fallback."""
    with pytest.raises(ValueError, match="complete profile identifiers"):
        validate_profile_partitions(
            [replace(_observations(), metadata=None)], [_observations()]
        )


def test_real_benchmark_accepts_disjoint_complete_cycle_groups() -> None:
    """Keep valid profile-disjoint benchmarks usable with unchanged splits."""
    evaluation = replace(
        _observations(),
        metadata={
            "PLATFORM_NUMBER": ("D", "D", "E", "E", "F", "F"),
            "CYCLE_NUMBER": (1, 1, 2, 2, 3, 3),
        },
    )
    validate_profile_partitions([_observations()], [evaluation])
