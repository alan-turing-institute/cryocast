"""Reference checks for matched full-grid normalization continuation evidence."""

import numpy as np
import pytest

from validation.norm_594_longer.metrics import (
    EXTRA_STEPS,
    FINAL_LR,
    RESTART_LR,
    SEEDS,
    continuation_lr,
    per_lead_metrics,
    training_order,
)


def test_restart_schedule_is_equal_budget_monotone_and_exact_at_endpoints() -> None:
    """Freeze both arms to the same restart and final learning rates."""
    values = np.array([continuation_lr(i) for i in range(EXTRA_STEPS)])
    assert values[0] == RESTART_LR
    assert values[-1] == FINAL_LR
    assert np.all(np.diff(values) <= 0)
    assert np.isfinite(values).all()


@pytest.mark.parametrize("index", [-1, EXTRA_STEPS, EXTRA_STEPS + 1])
def test_learning_rate_rejects_invalid_update_indices(index: int) -> None:
    """Do not accidentally extend the fixed budget or use a negative update."""
    with pytest.raises(ValueError, match="index"):
        continuation_lr(index)


@pytest.mark.parametrize("seed", SEEDS)
def test_matched_training_order_is_reproducible_and_nonrepeating(seed: int) -> None:
    """The norm name cannot influence window selection or order."""
    first, second = training_order(seed, 715), training_order(seed, 715)
    np.testing.assert_array_equal(first, second)
    assert len(set(first)) == EXTRA_STEPS
    assert first.min() >= 0
    assert first.max() < 715


def test_different_seeds_change_training_order() -> None:
    """Preserve genuine differences between paired random seeds."""
    assert not np.array_equal(training_order(575, 715), training_order(576, 715))


@pytest.mark.parametrize("count", [0, 1, EXTRA_STEPS - 1])
def test_training_order_requires_enough_distinct_windows(count: int) -> None:
    """Never fill a budget by silently repeating a smaller sample."""
    with pytest.raises(ValueError, match="distinct"):
        training_order(575, count)


def test_metrics_match_hand_computation_and_fixed_pixel_support() -> None:
    """Use known MAE, squared errors and signed area rather than the tested formula."""
    truth = np.array([[[[0.0, 0.2], [0.8, 0.5]]]], dtype=np.float32)
    prediction = np.array([[[[0.2, 0.0], [0.6, 0.0]]]], dtype=np.float32)
    mask = np.array([[[[True, True], [True, False]]]])
    score = per_lead_metrics(prediction, truth, mask)[0]
    assert score["count"] == 3
    assert score["absolute_sum"] == pytest.approx(0.6)
    assert score["squared_sum"] == pytest.approx(0.12)
    assert score["signed_sie_km2"] == 0
    assert score["absolute_sie_km2"] == 0


def test_extent_errors_are_absolute_per_window_before_averaging() -> None:
    """Opposite area errors must not cancel into an artificially perfect score."""
    low = np.zeros((1, 1, 2, 2), dtype=np.float32)
    high = np.ones_like(low)
    mask = np.ones_like(low, dtype=bool)
    first = per_lead_metrics(low, high, mask)[0]
    second = per_lead_metrics(high, low, mask)[0]
    assert first["signed_sie_km2"] + second["signed_sie_km2"] == 0
    assert (first["absolute_sie_km2"] + second["absolute_sie_km2"]) / 2 == 2500


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_forecasts_at_valid_truth_fail(bad: float) -> None:
    """Failed predictions cannot shrink the scoring denominator."""
    truth = np.zeros((1, 1, 2, 2), dtype=np.float32)
    prediction = truth.copy()
    prediction[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="Nonfinite"):
        per_lead_metrics(prediction, truth, np.ones_like(truth, dtype=bool))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_errors_are_accumulated_in_float64_before_squaring(dtype: type) -> None:
    """Avoid half-precision overflow in a mathematically finite metric."""
    prediction: np.ndarray = np.full((1, 1, 2, 2), 1000.0, dtype=dtype)
    truth = np.zeros_like(prediction)
    score = per_lead_metrics(prediction, truth, np.ones_like(truth, dtype=bool))[0]
    assert score["squared_sum"] == 4_000_000.0


def test_masked_values_never_change_scores() -> None:
    """Masking is fixed by truth support, not either forecast method."""
    truth = np.zeros((1, 1, 2, 2), dtype=np.float32)
    prediction = np.ones_like(truth)
    mask = np.ones_like(truth, dtype=bool)
    mask[..., 0, 0] = False
    expected = per_lead_metrics(prediction, truth, mask)
    prediction[..., 0, 0] = np.nan
    truth[..., 0, 0] = np.inf
    assert per_lead_metrics(prediction, truth, mask) == expected


def test_empty_support_is_an_explicit_failure() -> None:
    """An unobserved forecast is not a zero-error forecast."""
    truth = np.zeros((1, 1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="valid cells"):
        per_lead_metrics(truth, truth, np.zeros_like(truth, dtype=bool))
