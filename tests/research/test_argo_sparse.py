import numpy as np
import pandas as pd
import pytest

from cryocast.research.argo_sparse import (
    ArgoFrameColumns,
    GaussianInterpolationBaseline,
    SparseObservations,
    TorchCudaMemoryHook,
    benchmark_encoder,
    benchmark_retention_levels,
    compute_regression_metrics,
    pad_sparse_observations,
    sparse_observations_from_dataframe,
    split_observations,
)


def _observations() -> SparseObservations:
    """Create deterministic sparse observations for benchmark tests."""
    return SparseObservations(
        latitudes=np.array([70.0, 71.0, 72.0, 73.0], dtype=np.float64),
        longitudes=np.array([-20.0, -19.0, -18.0, -17.0], dtype=np.float64),
        measurements=np.array(
            [
                [-1.0, 33.0],
                [0.0, 33.5],
                [1.0, 34.0],
                [2.0, 34.5],
            ],
            dtype=np.float64,
        ),
        variable_names=("TEMP", "PSAL"),
    )


def test_dataframe_adapter_preserves_aligned_optional_fields() -> None:
    """Convert raw Argo columns while filtering invalid requested observations."""
    dataframe = pd.DataFrame(
        {
            "LATITUDE": [70.0, 71.0, 72.0],
            "LONGITUDE": [-20.0, -19.0, -18.0],
            "TEMP": [-1.0, np.nan, 1.0],
            "PSAL": [33.0, 33.5, 34.0],
            "PRES": [5.0, 10.0, 15.0],
            "TIME": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T02:00:00Z",
            ],
            "PLATFORM_NUMBER": ["A", "B", "C"],
        }
    )
    observations = sparse_observations_from_dataframe(
        dataframe,
        columns=ArgoFrameColumns(metadata=("PLATFORM_NUMBER",)),
        reference_time=pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    assert observations.count == 2
    assert observations.variable_names == ("TEMP", "PSAL")
    np.testing.assert_allclose(observations.latitudes, [70.0, 72.0])
    np.testing.assert_allclose(
        observations.measurements,
        [[-1.0, 33.0], [1.0, 34.0]],
    )
    assert observations.pressure is not None
    np.testing.assert_allclose(observations.pressure, [5.0, 15.0])
    assert observations.time_offsets_hours is not None
    np.testing.assert_allclose(observations.time_offsets_hours, [0.0, 2.0])
    assert observations.metadata == {"PLATFORM_NUMBER": ("A", "C")}


def test_held_out_split_is_reproducible() -> None:
    """Use the same seed to produce the same observed and held-out points."""
    observations = _observations()
    first = split_observations(observations, holdout_fraction=0.5, seed=17)
    second = split_observations(observations, holdout_fraction=0.5, seed=17)
    np.testing.assert_array_equal(first.observed.latitudes, second.observed.latitudes)
    np.testing.assert_array_equal(first.held_out.latitudes, second.held_out.latitudes)
    assert first.observed.count == 2
    assert first.held_out.count == 2


def test_current_interpolation_baseline_returns_finite_query_values() -> None:
    """Evaluate the current Gaussian distance rule at arbitrary coordinates."""
    predicted = GaussianInterpolationBaseline().predict(
        _observations(),
        np.array([70.5, 72.5], dtype=np.float64),
        np.array([-19.5, -17.5], dtype=np.float64),
    )
    assert predicted.shape == (2, 2)
    assert np.isfinite(predicted).all()


def test_regression_metrics_are_computed_per_variable() -> None:
    """Calculate expected MAE and RMSE independently for TEMP and PSAL."""
    predicted = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    target = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    metrics = compute_regression_metrics(predicted, target, ("TEMP", "PSAL"))
    assert metrics["TEMP"].mae == pytest.approx(1.0)
    assert metrics["TEMP"].rmse == pytest.approx(np.sqrt(2.0))
    assert metrics["TEMP"].count == 2
    assert metrics["PSAL"].mae == pytest.approx(2.5)
    assert metrics["PSAL"].rmse == pytest.approx(np.sqrt(8.5))
    assert metrics["PSAL"].count == 2


def test_benchmark_runner_uses_repeated_fixed_seed_splits() -> None:
    """Run repeated held-out reconstruction without accelerator accounting."""
    result = benchmark_encoder(
        GaussianInterpolationBaseline(),
        _observations(),
        holdout_fraction=0.25,
        repeats=2,
        seed=4,
    )
    assert result.encoder_name == "current-gaussian-interpolation"
    assert len(result.repeats) == 2
    assert all(repeat.n_observed == 3 for repeat in result.repeats)
    assert all(repeat.n_held_out == 1 for repeat in result.repeats)
    assert all(repeat.peak_memory_bytes is None for repeat in result.repeats)


def test_retention_benchmark_reuses_held_out_split() -> None:
    """Reduce observed points without changing held-out counts."""
    results = benchmark_retention_levels(
        GaussianInterpolationBaseline(),
        _observations(),
        retention_fractions=(1.0, 0.5),
        holdout_fraction=0.25,
        repeats=1,
        seed=9,
    )
    assert results[1.0].repeats[0].n_observed == 3
    assert results[0.5].repeats[0].n_observed == 2
    assert results[1.0].repeats[0].n_held_out == 1
    assert results[0.5].repeats[0].n_held_out == 1


def test_padding_keeps_lengths_and_valid_mask() -> None:
    """Pad variable-length samples without losing observation lengths."""
    observations = _observations()
    shorter = observations.take(np.array([0, 2], dtype=np.int64))
    batch = pad_sparse_observations([shorter, observations])
    assert batch.measurements.shape == (2, 4, 2)
    np.testing.assert_array_equal(batch.lengths, [2, 4])
    np.testing.assert_array_equal(
        batch.mask,
        [[True, True, False, False], [True, True, True, True]],
    )


def test_cuda_memory_hook_is_disabled_for_cpu() -> None:
    """Return no GPU memory measurement when explicitly configured for CPU."""
    hook = TorchCudaMemoryHook(device="cpu")
    hook.start()
    assert hook.stop() is None
