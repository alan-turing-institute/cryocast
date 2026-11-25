"""Analytical regression cases for per-lead-time spatial forecast metrics.

These tests use known per-pixel errors and multiple batch updates. They pin
the population (not batch-averaged) statistics and the handling of land cells.
"""

import math

import pytest
import torch

from cryocast.metrics import (
    MAEPerForecastDay,
    RMSEPerForecastDay,
    SpatialMeanGroundTruthPerForecastDay,
    SpatialMeanPredictionPerForecastDay,
    SSIMPerForecastDay,
)
from cryocast.metrics.base_daily_metric import BaseDailyMetric


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (MAEPerForecastDay, [1.75, 2.5]),
        (RMSEPerForecastDay, [math.sqrt(3.75), 3.0]),
    ],
)
def test_pointwise_error_aggregates_all_batches_and_leads(
    metric_type: type[BaseDailyMetric], expected: list[float]
) -> None:
    # Two samples, two forecast leads, one channel, two pixels:
    # lead 0 errors [1, 2, 3, 1]; lead 1 errors [2, 0, 4, 4].
    """Match hand-calculated global per-lead MAE/RMSE after split updates."""
    predictions = torch.tensor(
        [[[1.0, 2.0], [2.0, 0.0]], [[3.0, 1.0], [4.0, 4.0]]]
    ).reshape(2, 2, 1, 1, 2)
    target = torch.zeros_like(predictions)

    one_batch = metric_type()
    one_batch.update(predictions, target)

    split_batches = metric_type()
    split_batches.update(predictions[:1], target[:1])
    split_batches.update(predictions[1:], target[1:])

    torch.testing.assert_close(
        one_batch.compute(), torch.tensor(expected, dtype=torch.float32)
    )
    torch.testing.assert_close(split_batches.compute(), one_batch.compute())


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (MAEPerForecastDay, [2.0, 3.0]),
        (RMSEPerForecastDay, [math.sqrt(5.0), math.sqrt(10.0)]),
    ],
)
def test_land_mask_excludes_nan_land_values(
    metric_type: type[BaseDailyMetric], expected: list[float]
) -> None:
    # Two valid ocean pixels on the left, two invalid land pixels on the right.
    """Ignore invalid values at masked land pixels during averaging."""
    predictions = torch.tensor(
        [
            [
                [[[1.0, float("nan")], [3.0, float("nan")]]],
                [[[2.0, float("nan")], [4.0, float("nan")]]],
            ]
        ]
    )
    target = torch.zeros_like(predictions)
    target[..., 1] = float("nan")
    mask = torch.tensor([[True, False], [True, False]])

    metric = metric_type(land_mask=mask)
    metric.update(predictions, target)

    assert torch.isfinite(metric.compute()).all()
    torch.testing.assert_close(
        metric.compute(), torch.tensor(expected, dtype=torch.float32)
    )


def test_pointwise_error_counts_all_channels_not_only_first() -> None:
    """Average errors over every channel as well as spatial pixels."""
    predictions = torch.tensor(
        [[[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]]]
    ).reshape(1, 2, 2, 1, 2)
    metric = MAEPerForecastDay()
    metric.update(predictions, torch.zeros_like(predictions))

    torch.testing.assert_close(metric.compute(), torch.tensor([4.0, 5.0]))


@pytest.mark.parametrize("metric_type", [MAEPerForecastDay, RMSEPerForecastDay])
def test_incompatible_forecast_length_rejected(
    metric_type: type[BaseDailyMetric],
) -> None:
    """Reject accumulation when a later batch has a different lead count."""
    metric = metric_type()
    metric.update(torch.zeros(1, 2, 1, 2, 2), torch.zeros(1, 2, 1, 2, 2))

    with pytest.raises(ValueError, match="Time dimension mismatch"):
        metric.update(torch.ones(1, 3, 1, 2, 2), torch.zeros(1, 3, 1, 2, 2))


@pytest.mark.parametrize("metric_type", [MAEPerForecastDay, RMSEPerForecastDay])
def test_reset_discards_accumulated_forecasts(
    metric_type: type[BaseDailyMetric],
) -> None:
    """Start a new accumulation with no state from the previous forecasts."""
    metric = metric_type()
    metric.update(torch.ones(1, 2, 1, 2, 2), torch.zeros(1, 2, 1, 2, 2))
    torch.testing.assert_close(metric.compute(), torch.ones(2))

    metric.reset()
    assert metric.sum_errors.numel() == 0
    assert metric.count.numel() == 0

    metric.update(torch.zeros(1, 1, 1, 2, 2), torch.zeros(1, 1, 1, 2, 2))
    torch.testing.assert_close(metric.compute().reshape(-1), torch.zeros(1))


def test_spatial_mean_traces_use_respective_fields_and_ocean_mask() -> None:
    """Distinguish prediction and truth means and exclude land pixels."""
    predictions = torch.tensor(
        [[[[[1.0, 99.0], [3.0, 99.0]]], [[[2.0, 99.0], [6.0, 99.0]]]]]
    )
    truth = torch.tensor([[[[[2.0, 88.0], [4.0, 88.0]]], [[[4.0, 88.0], [8.0, 88.0]]]]])
    ocean = torch.tensor([[True, False], [True, False]])

    pred_metric = SpatialMeanPredictionPerForecastDay(land_mask=ocean)
    truth_metric = SpatialMeanGroundTruthPerForecastDay(land_mask=ocean)
    pred_metric.update(predictions, truth)
    truth_metric.update(predictions, truth)

    torch.testing.assert_close(pred_metric.compute(), torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(truth_metric.compute(), torch.tensor([3.0, 6.0]))


def test_ssim_is_unity_for_identical_nonconstant_multichannel_forecasts() -> None:
    """Report perfect similarity per lead on equal multichannel images."""
    generator = torch.Generator().manual_seed(26)
    observations = torch.rand((2, 3, 2, 8, 8), generator=generator)

    metric = SSIMPerForecastDay(filter_size=3, filter_sigma=1.0)
    metric.update(observations, observations.clone())

    torch.testing.assert_close(metric.compute(), torch.ones(3), atol=1e-5, rtol=0)


def test_single_pixel_ssim_matches_luminance_formula() -> None:
    # With a 1x1 kernel, all local variances/covariances vanish. The structural
    # component is 1 and SSIM reduces to (2ab + c1)/(a^2 + b^2 + c1).
    """Reduce SSIM to its analytic luminance term at filter size one."""
    pred = torch.full((1, 2, 1, 2, 3), 0.25)
    truth = torch.full_like(pred, 0.75)
    metric = SSIMPerForecastDay(filter_size=1)
    metric.update(pred, truth)

    c1 = 0.01**2
    expected = (2 * 0.25 * 0.75 + c1) / (0.25**2 + 0.75**2 + c1)
    torch.testing.assert_close(
        metric.compute(), torch.full((2,), expected), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("filter_size", [0, 2, -3])
def test_ssim_rejects_nonpositive_or_even_filter_sizes(filter_size: int) -> None:
    """Reject Gaussian windows that cannot have a central pixel."""
    with pytest.raises(ValueError, match="positive odd"):
        SSIMPerForecastDay(filter_size=filter_size)


def test_ssim_gaussian_kernel_is_normalized_and_checkpoint_free() -> None:
    """Normalize the kernel without serializing transient filter state."""
    metric = SSIMPerForecastDay(filter_size=5, filter_sigma=1.5)

    torch.testing.assert_close(metric.kernel.sum(), torch.tensor(1.0))
    assert "kernel" not in metric.state_dict()


@pytest.mark.parametrize("metric_type", [MAEPerForecastDay, RMSEPerForecastDay])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_full_grid_reduction_is_finite_and_exact(
    metric_type: type[BaseDailyMetric], dtype: torch.dtype
) -> None:
    """A unit error over a 432x432 grid must remain one, not overflow or round."""
    predictions = torch.ones((2, 2, 1, 432, 432), dtype=dtype)
    metric = metric_type()
    metric.update(predictions, torch.zeros_like(predictions))

    torch.testing.assert_close(metric.compute(), torch.ones(2), rtol=0, atol=0)


def test_half_precision_rmse_does_not_underflow_before_reduction() -> None:
    """Square a representable small error in float32 rather than rounding it to zero."""
    predictions = torch.full((1, 2, 1, 2, 2), 1e-4, dtype=torch.float16)
    expected = float(predictions[0, 0, 0, 0, 0])
    metric = RMSEPerForecastDay()
    metric.update(predictions, torch.zeros_like(predictions))

    torch.testing.assert_close(
        metric.compute(), torch.full((2,), expected), rtol=1e-6, atol=0
    )


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (MAEPerForecastDay, [0.5, 0.25]),
        (RMSEPerForecastDay, [math.sqrt(11 / 30), math.sqrt(11 / 120)]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unequal_batches_count_all_masked_channels_and_invalidate_cached_compute(
    metric_type: type[BaseDailyMetric], expected: list[float], dtype: torch.dtype
) -> None:
    """Accumulate six ocean errors per lead across batches of one and two examples."""
    # Lead 0 has [0, .2, .4, .6, .8, 1]; lead 1 is half that.
    predictions = torch.zeros((3, 2, 2, 1, 2), dtype=dtype)
    predictions[:, 0, :, 0, 0] = torch.arange(6, dtype=dtype).reshape(3, 2) / 5
    predictions[:, 1, :, 0, 0] = predictions[:, 0, :, 0, 0] / 2
    predictions[..., 1] = float("inf")
    target = torch.zeros_like(predictions)
    target[..., 1] = float("nan")
    mask = torch.tensor([[True, False]])
    metric = metric_type(land_mask=mask)
    metric.update(predictions[:1], target[:1])
    first = metric.compute().clone()
    metric.update(predictions[1:], target[1:])
    actual = metric.compute()

    assert actual.dtype == dtype
    assert not torch.equal(actual, first)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype))
    torch.testing.assert_close(metric.count, torch.tensor([6, 6]))


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (SpatialMeanPredictionPerForecastDay, [0.25, 0.5]),
        (SpatialMeanGroundTruthPerForecastDay, [0.5, 0.75]),
    ],
)
def test_spatial_traces_exclude_nonfinite_land_without_using_the_other_field(
    metric_type: type[BaseDailyMetric], expected: list[float]
) -> None:
    """Invalid land values must not contaminate either field's ocean-only trace."""
    predictions = torch.tensor(
        [
            [
                [[[0.25, float("nan"), float("inf")]]],
                [[[0.5, float("inf"), float("nan")]]],
            ]
        ]
    )
    truth = torch.tensor(
        [
            [
                [[[0.5, float("-inf"), float("nan")]]],
                [[[0.75, float("nan"), float("-inf")]]],
            ]
        ]
    )
    metric = metric_type(land_mask=torch.tensor([[True, False, False]]))
    metric.update(predictions, truth)

    torch.testing.assert_close(metric.compute(), torch.tensor(expected))


@pytest.mark.parametrize("metric_type", [MAEPerForecastDay, RMSEPerForecastDay])
def test_invalid_ocean_error_is_not_reported_as_a_valid_perfect_score(
    metric_type: type[BaseDailyMetric],
) -> None:
    """Land masking must not silently discard invalid errors on ocean cells."""
    predictions = torch.zeros(1, 2, 1, 1, 2)
    predictions[:, 0, :, :, 0] = float("nan")
    predictions[:, 1, :, :, 0] = float("inf")
    metric = metric_type(land_mask=torch.tensor([[True, False]]))
    metric.update(predictions, torch.zeros_like(predictions))

    actual = metric.compute()
    assert torch.isnan(actual[0])
    assert torch.isinf(actual[1])


def test_ssim_luminance_uses_population_weights_and_higher_is_better() -> None:
    """Average analytic similarities across unequal batches, not their batch means."""
    pred = (
        torch.tensor([0.25, 0.5, 0.75]).reshape(3, 1, 1, 1, 1).expand(-1, 2, -1, 1, 2)
    )
    truth = torch.full_like(pred, 0.75)
    mask = torch.tensor([[True, False]])
    metric = SSIMPerForecastDay(filter_size=1, land_mask=mask)
    metric.update(pred[:1], truth[:1])
    first = metric.compute().clone()
    metric.update(pred[1:], truth[1:])
    c1 = 0.01**2
    expected = (
        sum((2 * a * 0.75 + c1) / (a * a + 0.75**2 + c1) for a in (0.25, 0.5, 0.75)) / 3
    )

    torch.testing.assert_close(metric.compute(), torch.full((2,), expected))
    assert torch.all(metric.compute() > first)
    identity = SSIMPerForecastDay(filter_size=1, land_mask=mask)
    identity.update(truth, truth)
    torch.testing.assert_close(identity.compute(), torch.ones(2))
    assert torch.all(identity.compute() > metric.compute())
