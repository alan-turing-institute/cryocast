import numpy as np
import torch

from cryocast.research.argo_cross_attention import (
    SparseCrossAttentionConfig,
    SparseCrossAttentionEncoder,
)
from cryocast.research.argo_reconstruction import (
    MeasurementNormaliser,
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
from cryocast.research.argo_sparse import SparseObservations


def _observations(shift: float = 0.0) -> SparseObservations:
    """Create a deterministic smooth TEMP/PSAL point field."""
    latitudes = np.array([70.0, 71.0, 72.0, 73.0, 74.0, 75.0], dtype=np.float64)
    longitudes = np.array([-20.0, -19.0, -18.0, -17.0, -16.0, -15.0])
    measurements = np.column_stack(
        (
            0.2 * latitudes + shift,
            34.0 + 0.05 * longitudes - 0.1 * shift,
        )
    )
    return SparseObservations(
        latitudes=latitudes,
        longitudes=longitudes,
        measurements=measurements,
        variable_names=("TEMP", "PSAL"),
    )


def _latent_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return a compact latent geography for reconstruction tests."""
    return (
        np.array([[70.0, 70.0], [74.0, 74.0]], dtype=np.float64),
        np.array([[-20.0, -16.0], [-20.0, -16.0]], dtype=np.float64),
    )


def _models(
    seed: int = 11,
) -> tuple[SparseReconstructionModel, SparseReconstructionModel]:
    """Build small SetConv and attention probes with equal readout capacity."""
    latitudes, longitudes = _latent_grid()
    torch.manual_seed(seed)
    setconv = SparseReconstructionModel(
        SparseSetConvEncoder(
            SparseSetConvConfig(
                latent_channels=4,
                hidden_channels=8,
                length_scale_km=500.0,
                query_chunk_size=2,
            ),
            latitudes,
            longitudes,
        ),
        readout_seed=seed,
    )
    torch.manual_seed(seed)
    attention = SparseReconstructionModel(
        SparseCrossAttentionEncoder(
            SparseCrossAttentionConfig(
                latent_channels=4,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                feedforward_dim=24,
                fourier_frequencies=1,
                query_chunk_size=2,
            ),
            latitudes,
            longitudes,
        ),
        readout_seed=seed,
    )
    return setconv, attention


def test_measurement_normaliser_uses_training_values_and_round_trips() -> None:
    """Fit statistics on training data only and recover physical measurements."""
    training = [_observations(0.0), _observations(1.0)]
    normaliser = MeasurementNormaliser.fit(training)
    expected = np.mean(
        np.concatenate([sample.measurements for sample in training], axis=0),
        axis=0,
    )
    np.testing.assert_allclose(normaliser.mean, expected)

    evaluation = _observations(100.0)
    normalised = normaliser.normalise(evaluation)
    np.testing.assert_allclose(
        normaliser.denormalise(normalised.measurements),
        evaluation.measurements,
    )


def test_common_readout_has_identical_capacity_and_initialisation() -> None:
    """Give both learned encoders exactly the same reconstruction head."""
    setconv, attention = _models()
    assert (
        parameter_counts(setconv)["common_readout"]
        == parameter_counts(attention)["common_readout"]
    )
    torch.testing.assert_close(setconv.readout.weight, attention.readout.weight)
    torch.testing.assert_close(setconv.readout.bias, attention.readout.bias)


def test_evaluation_cases_reuse_held_out_points_across_retention_levels() -> None:
    """Change only retained inputs while holding each evaluation target fixed."""
    cases = build_evaluation_cases(
        [_observations()],
        retention_fractions=(1.0, 0.5, 0.25),
        repeats=1,
        seed=5,
    )
    assert [case.retention for case in cases] == [1.0, 0.5, 0.25]
    for case in cases[1:]:
        np.testing.assert_array_equal(
            cases[0].held_out.latitudes, case.held_out.latitudes
        )
        np.testing.assert_array_equal(
            cases[0].held_out.measurements,
            case.held_out.measurements,
        )
    assert cases[0].observed.count > cases[1].observed.count >= cases[2].observed.count


def test_both_learned_candidates_use_common_training_and_evaluation_protocol() -> None:
    """Train both probes with one schedule and evaluate the same physical-unit cases."""
    raw_training = [_observations(shift) for shift in (0.0, 0.5, 1.0)]
    normaliser = MeasurementNormaliser.fit(raw_training)
    training = [normaliser.normalise(sample) for sample in raw_training]
    cases = build_evaluation_cases(
        [_observations(1.5)],
        retention_fractions=(1.0, 0.5),
        repeats=1,
        seed=19,
    )
    config = ReconstructionTrainingConfig(
        epochs=2,
        learning_rate=1e-3,
        holdout_fraction=0.34,
    )
    setconv, attention = _models(seed=19)

    for model in (setconv, attention):
        fit = fit_reconstruction_model(
            model,
            training,
            config=config,
            seed=19,
        )
        records = evaluate_reconstruction_model(
            model,
            cases,
            normaliser=normaliser,
            training_seed=19,
        )
        assert np.isfinite(fit.final_loss)
        assert len(records) == len(cases)
        assert all(np.isfinite(record.metrics["TEMP"].mae) for record in records)
        assert all(np.isfinite(record.metrics["PSAL"].rmse) for record in records)
        assert [record.n_held_out for record in records] == [
            case.held_out.count for case in cases
        ]

    interpolation = evaluate_interpolation(cases)
    summaries = summarise_records(interpolation)
    assert [summary.retention for summary in summaries] == [1.0, 0.5]
    assert all(summary.evaluations == 1 for summary in summaries)
