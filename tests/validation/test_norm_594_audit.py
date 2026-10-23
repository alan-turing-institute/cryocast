"""Reference arithmetic for the independent verifier and published aggregation."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation.norm_594_longer.audit import EvidenceError, score_arrays
from validation.norm_594_longer.report import (
    cross_seed,
    load_cases,
    paired_comparisons,
    seed_summaries,
    training_budget_effects,
    write_csv,
)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_independent_extent_scoring_keeps_strict_representable_threshold(
    dtype: type,
) -> None:
    """Exactly fifteen percent is excluded; the next representable value is ice."""
    threshold: np.ndarray = np.array(0.15, dtype=dtype)
    above = np.nextafter(threshold, np.array(1.0, dtype=dtype))
    truth: np.ndarray = np.full((1, 2, 2), threshold, dtype=dtype)
    prediction = truth.copy()
    prediction[0, 0, 0] = above
    result = score_arrays(prediction, truth, np.ones_like(truth, dtype=bool))
    assert result["count"] == 4
    assert result["signed_sie_km2"] == 625.0
    assert result["absolute_sie_km2"] == 625.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_independent_verifier_rejects_nonfinite_prediction(bad: float) -> None:
    """The reference verifier cannot drop the model's difficult pixels."""
    truth = np.zeros((1, 2, 2), dtype=np.float64)
    prediction = truth.copy()
    prediction[0, 0, 0] = bad
    with pytest.raises(EvidenceError, match="Invalid raw"):
        score_arrays(prediction, truth, np.ones_like(truth, dtype=bool))


def test_independent_verifier_uses_double_before_subtracting() -> None:
    """A half-precision difference outside its range still has a finite score."""
    prediction = np.full((1, 2, 2), 60000, dtype=np.float16)
    truth = -prediction
    result = score_arrays(prediction, truth, np.ones_like(truth, dtype=bool))
    assert result["squared_sum"] == 4 * 120000.0**2
    assert result["absolute_sum"] == 480000.0


def test_report_pools_pixels_before_square_root_with_unequal_support() -> None:
    """A large high-error window must not be downweighted to a small window."""
    rows = []
    for month in ("2024-01", "2024-07"):
        for window in range(15):
            count, error = (10, 3) if window == 14 else (2, 1)
            for lead in range(1, 15):
                rows.append(  # noqa: PERF401 - explicit reference rows
                    {
                        "seed": 575,
                        "norm": "batchnorm",
                        "selection": "final",
                        "month": month,
                        "origin": f"{month}-{window + 3:02d}",
                        "lead": lead,
                        "count": count,
                        "absolute_sum": count * error,
                        "squared_sum": count * error**2,
                        "absolute_sie_km2": 625.0,
                        "signed_sie_km2": -625.0,
                    }
                )
    summaries, daily = seed_summaries(rows)
    assert len(summaries) == 3
    assert len(daily) == 42
    for row in summaries:
        assert row["mean_rmse"] == pytest.approx(np.sqrt(118 / 38))
        assert row["late_rmse"] == pytest.approx(np.sqrt(118 / 38))
        assert row["mean_mae"] == pytest.approx(58 / 38)
        assert row["mean_absolute_sie_km2"] == 625.0
    with pytest.raises(ValueError, match="temporal support"):
        seed_summaries(rows[:-1])


def _summaries() -> list[dict[str, Any]]:
    """Return transparent seed summaries for cross-seed arithmetic tests."""
    rows = []
    metrics = (
        "mean_rmse",
        "late_rmse",
        "mean_mae",
        "mean_absolute_sie_km2",
        "late_absolute_sie_km2",
    )
    for month in ("2024-01", "2024-07", "combined"):
        rows.append(
            {
                "seed": None,
                "month": month,
                "norm": "persistence",
                "selection": "reference",
                **dict.fromkeys(metrics, 0.05),
            }
        )
        for selection in ("initial", "selected", "final"):
            for index, seed in enumerate((575, 576, 577), 1):
                for norm, offset in (("batchnorm", 0.0), ("groupnorm", 0.03)):
                    rows.append(
                        {
                            "seed": seed,
                            "month": month,
                            "norm": norm,
                            "selection": selection,
                            **dict.fromkeys(metrics, index / 10 + offset),
                        }
                    )
    return rows


def test_cross_seed_spread_is_sample_sd_and_persistence_is_not_replicated() -> None:
    """Three training seeds imply ddof one; persistence has no seed error bar."""
    result = cross_seed(_summaries())
    assert len(result) == 27
    for row in result:
        if row["norm"] == "persistence":
            assert row["n_seeds"] == 1
            assert row["mean_rmse_sd"] is None
        else:
            assert row["n_seeds"] == 3
            assert row["mean_rmse_sd"] == pytest.approx(0.1)


def test_paired_effects_preserve_the_direction_of_every_seed() -> None:
    """A worse GroupNorm result must stay negative in the improvement column."""
    result = paired_comparisons(_summaries())
    assert len(result) == 27
    for row in result:
        assert row["groupnorm_minus_batchnorm_rmse"] == pytest.approx(0.03)
        assert row["groupnorm_rmse_reduction_percent"] < 0
        assert row["batchnorm_minus_persistence_rmse"] > 0
        assert row["groupnorm_minus_persistence_rmse"] > 0


def test_missing_seed_cannot_be_summarized_as_a_complete_comparison() -> None:
    """Do not disguise failed or absent runs by changing the seed denominator."""
    with pytest.raises(ValueError, match="Missing seed"):
        cross_seed(_summaries()[:-1])


def test_compact_csv_roundtrip_preserves_seed_types_and_sufficient_statistics(
    tmp_path: Path,
) -> None:
    """Public tables must regenerate exactly without retaining duplicate large JSON."""
    records: list[dict[str, Any]] = [
        {
            "seed": None,
            "norm": "persistence",
            "selection": "reference",
            "origin": "2024-01-03",
            "month": "2024-01",
            "lead": 1,
            "count": 100,
            "absolute_sum": 0.1234567890123456,
            "squared_sum": 0.0001234567890123456,
            "signed_sie_km2": -625.0,
            "absolute_sie_km2": 625.0,
        },
        {
            "seed": 575,
            "norm": "batchnorm",
            "selection": "final",
            "origin": "2024-07-03",
            "month": "2024-07",
            "lead": 14,
            "count": 101,
            "absolute_sum": 5.0,
            "squared_sum": 2.75,
            "signed_sie_km2": 1250.0,
            "absolute_sie_km2": 1250.0,
        },
    ]
    write_csv(tmp_path / "per-case.csv", records)
    assert load_cases(tmp_path) == records


def test_added_training_effect_uses_the_same_seed_and_test_population() -> None:
    """Keep a 96-to-288 update contrast separate from adding July test windows."""
    summaries = _summaries()
    for row in summaries:
        if (
            row["selection"] == "final"
            and row["norm"] == "batchnorm"
            and row["seed"] == 575
        ):
            row["mean_rmse"] -= 0.01
    effects = training_budget_effects(summaries)
    assert len(effects) == 18
    for row in effects:
        expected = -0.01 if row["norm"] == "batchnorm" and row["seed"] == 575 else 0.0
        assert row["final_minus_initial_rmse"] == pytest.approx(expected)
