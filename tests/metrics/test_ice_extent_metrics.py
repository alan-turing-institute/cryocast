"""Scientific reference cases for sea-ice extent, edge and accuracy metrics.

All expected values follow directly from hand-counted ice cells, the 15%
concentration threshold, and the configured km-per-pixel resolution.
"""

import pytest
import torch

from cryocast.metrics import (
    DistanceAveragedIceEdgeErrorPerForecastDay,
    FractionalSkillScorePerForecastDay,
    IceNetAccuracyPerForecastDay,
    IntegratedIceEdgeErrorPerForecastDay,
    SeaIceExtentErrorPerForecastDay,
)
from cryocast.metrics.base_ice_area_metric import MeanIceAreaMetric
from cryocast.types import SEA_ICE_THRESHOLD


def _two_lead_displacements() -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.zeros((1, 2, 1, 2, 2))
    truth = torch.zeros_like(predictions)
    # Lead 0: move one ice cell: net extent error 0, symmetric difference 2.
    predictions[0, 0, 0, 0, 1] = 1.0
    truth[0, 0, 0, 0, 0] = 1.0
    # Lead 1: two extra ice cells: net extent error +2, difference 2.
    predictions[0, 1, 0, 0, :] = 1.0
    return predictions, truth


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (SeaIceExtentErrorPerForecastDay, [0.0, 200.0]),
        (IntegratedIceEdgeErrorPerForecastDay, [200.0, 200.0]),
    ],
)
def test_signed_extent_and_symmetric_difference_are_distinct(
    metric_type: type[MeanIceAreaMetric], expected: list[float]
) -> None:
    """Keep net sea-ice area bias separate from mismatch area."""
    predictions, truth = _two_lead_displacements()
    metric = metric_type(pixel_size=10)
    metric.update(predictions, truth)

    torch.testing.assert_close(metric.compute(), torch.tensor(expected))
    # Two opposing errors cancel in signed SIE, not in IIEE.


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (SeaIceExtentErrorPerForecastDay, -12.5),
        (IntegratedIceEdgeErrorPerForecastDay, 62.5),
    ],
)
def test_signed_and_absolute_errors_average_by_sample(
    metric_type: type[MeanIceAreaMetric], expected: float
) -> None:
    """Average ice-area metrics over frames across update calls."""
    predictions = torch.zeros((2, 1, 1, 2, 2))
    truth = torch.zeros_like(predictions)
    # Sample 0: two false positives, area +2*25.
    predictions[0, 0, 0, 0, :] = 1.0
    # Sample 1: three false negatives, area -3*25.
    truth[1, 0, 0, 0, :] = 1.0
    truth[1, 0, 0, 1, 0] = 1.0
    # SIE average (+50 - 75)/2 = -12.5; IIEE (50 + 75)/2 = 62.5.
    combined = metric_type(pixel_size=5)
    combined.update(predictions, truth)
    separate = metric_type(pixel_size=5)
    separate.update(predictions[:1], truth[:1])
    separate.update(predictions[1:], truth[1:])

    torch.testing.assert_close(combined.compute().reshape(-1), torch.tensor([expected]))
    torch.testing.assert_close(combined.compute(), separate.compute())


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        (SeaIceExtentErrorPerForecastDay, -9.0),
        (IntegratedIceEdgeErrorPerForecastDay, 9.0),
    ],
)
def test_land_mask_removes_false_ice_on_land(
    metric_type: type[MeanIceAreaMetric], expected: float
) -> None:
    """Exclude land-only errors from the physical-area metrics."""
    predictions, truth = _two_lead_displacements()
    # The false positive in lead 0 occurs on land, so only the
    # remaining false negative contributes.
    ocean_mask = torch.tensor([[True, False], [True, True]])
    metric = metric_type(pixel_size=3, land_mask=ocean_mask)
    metric.update(predictions[:, :1], truth[:, :1])

    torch.testing.assert_close(metric.compute().reshape(-1), torch.tensor([expected]))


def test_exact_threshold_is_open_water_for_all_extent_metrics() -> None:
    """Treat SIC equal to the ice cutoff as ice-free."""
    predictions = torch.tensor([[[[[SEA_ICE_THRESHOLD, SEA_ICE_THRESHOLD + 0.01]]]]])
    truth = torch.zeros_like(predictions)
    for metric_type in (
        SeaIceExtentErrorPerForecastDay,
        IntegratedIceEdgeErrorPerForecastDay,
    ):
        metric = metric_type(pixel_size=4)
        metric.update(predictions, truth)
        torch.testing.assert_close(metric.compute().reshape(-1), torch.tensor([16.0]))


@pytest.mark.parametrize(
    "metric_type",
    [SeaIceExtentErrorPerForecastDay, IntegratedIceEdgeErrorPerForecastDay],
)
def test_ice_extent_metric_rejects_multiple_target_channels(
    metric_type: type[MeanIceAreaMetric],
) -> None:
    """Prevent ambiguous physical-area metrics for multichannel targets."""
    metric = metric_type()
    multi_channel = torch.zeros((1, 1, 2, 2, 2))
    with pytest.raises(ValueError, match="single sea-ice-concentration channel"):
        metric.update(multi_channel, multi_channel)


def test_distance_averaged_ice_edge_error_matches_isolated_cell_displacement() -> None:
    """Check area divided by edge length against an exact pixel example."""
    predictions = torch.zeros((1, 1, 1, 5, 5))
    truth = torch.zeros_like(predictions)
    truth[0, 0, 0, 1, 1] = 1
    predictions[0, 0, 0, 3, 3] = 1

    # Two disagreement cells = 2*p^2 of area, one edge cell in each field
    # = 2*p of edge length. DIIEE = 2*(2*p^2)/(2*p) = 2*p.
    metric = DistanceAveragedIceEdgeErrorPerForecastDay(pixel_size=7)
    metric.update(predictions, truth)
    torch.testing.assert_close(metric.compute().reshape(-1), torch.tensor([14.0]))


def test_distance_averaged_edge_error_is_undefined_without_edges() -> None:
    """Report undefined displacement where no genuine edges exist."""
    empty = torch.zeros((1, 2, 1, 3, 3))
    ice = torch.ones_like(empty)
    metric = DistanceAveragedIceEdgeErrorPerForecastDay(pixel_size=10)
    metric.update(empty, empty)
    metric.update(ice, ice)

    assert torch.isnan(metric.compute()).all()


def test_ice_accuracy_supports_land_mask_and_nonuniform_sample_weights() -> None:
    """Combine classification correctness with unequal ocean-only weights."""
    predictions = torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]]]])
    truth = torch.tensor([[[[[1.0, 1.0], [0.0, 0.0]]]]])
    sample_weights = torch.tensor([[[[[1.0, 3.0], [10.0, 10.0]]]]])

    unmasked = IceNetAccuracyPerForecastDay()
    unmasked.update(predictions, truth)
    torch.testing.assert_close(unmasked.compute().reshape(-1), torch.tensor([75.0]))

    ocean_mask = torch.tensor([[True, True], [False, False]])
    masked = IceNetAccuracyPerForecastDay(land_mask=ocean_mask)
    masked.update(predictions, truth, sample_weight=sample_weights)
    # Only ocean contributes, with 1 correct unit and 3 incorrect units.
    torch.testing.assert_close(masked.compute().reshape(-1), torch.tensor([25.0]))


@pytest.mark.parametrize("neighbourhood_size", [1, 3])
def test_fss_is_one_for_identical_ice_edges(neighbourhood_size: int) -> None:
    """Return perfect fractional skill for matching ice boundaries."""
    ice = torch.zeros((1, 2, 1, 5, 5))
    ice[0, 0, 0, 1, 1] = 1.0
    ice[0, 1, 0, 1:4, 1:4] = 1.0

    metric = FractionalSkillScorePerForecastDay(neighbourhood_size=neighbourhood_size)
    metric.update(ice, ice)

    torch.testing.assert_close(metric.compute(), torch.ones(2), atol=1e-6, rtol=0)


def test_fss_is_zero_for_disjoint_single_cell_edges_with_unit_neighbourhood() -> None:
    """Match the zero-skill reference for disjoint unit-neighbourhood edges."""
    predictions = torch.zeros((1, 1, 1, 5, 5))
    truth = torch.zeros_like(predictions)
    truth[0, 0, 0, 1, 1] = 1
    predictions[0, 0, 0, 3, 3] = 1

    metric = FractionalSkillScorePerForecastDay(neighbourhood_size=1)
    metric.update(predictions, truth)

    torch.testing.assert_close(metric.compute().reshape(-1), torch.zeros(1))


def test_fss_is_undefined_when_both_forecasts_have_no_ice_edge() -> None:
    """Mark FSS as undefined when the reference denominator is zero."""
    metric = FractionalSkillScorePerForecastDay(neighbourhood_size=1)
    empty = torch.zeros((1, 2, 1, 4, 4))
    metric.update(empty, empty)
    assert torch.isnan(metric.compute()).all()


@pytest.mark.parametrize("neighbourhood_size", [0, -1, 2, 4])
def test_fss_requires_positive_odd_neighbourhood(neighbourhood_size: int) -> None:
    """Reject invalid neighbourhood window sizes."""
    with pytest.raises(ValueError, match="positive odd integer"):
        FractionalSkillScorePerForecastDay(neighbourhood_size=neighbourhood_size)


@pytest.mark.parametrize("pixel_size", [5, 25])
def test_extent_bias_cancellation_does_not_imply_correct_ice_maps(
    pixel_size: int,
) -> None:
    """Opposing frame biases cancel; absolute extent error and IIEE remain positive."""
    pred = torch.zeros(2, 2, 1, 2, 2)
    truth = torch.zeros_like(pred)
    pred[0, 0, 0, 0, :] = 1
    truth[1, 0, 0, 0, :] = 1
    truth[:, 1, 0, 0, 0] = 1
    area = float(pixel_size**2)
    extent = SeaIceExtentErrorPerForecastDay(pixel_size=pixel_size)
    disagreement = IntegratedIceEdgeErrorPerForecastDay(pixel_size=pixel_size)
    for metric in (extent, disagreement):
        metric.update(pred[:1], truth[:1])
        metric.compute()  # Exercise cache invalidation before the second batch.
        metric.update(pred[1:], truth[1:])

    torch.testing.assert_close(extent.compute(), torch.tensor([0.0, -area]))
    torch.testing.assert_close(disagreement.compute(), torch.tensor([2 * area, area]))
    per_frame_absolute_extent = torch.tensor([[2 * area, area], [2 * area, area]])
    assert float(per_frame_absolute_extent[:, 0].mean()) == 2 * area
    reverse = SeaIceExtentErrorPerForecastDay(pixel_size=pixel_size)
    reverse.update(truth, pred)
    torch.testing.assert_close(reverse.compute(), -extent.compute())


@pytest.mark.parametrize(
    ("metric_type", "cell_errors"),
    [
        (SeaIceExtentErrorPerForecastDay, [0.0, 2.0]),
        (IntegratedIceEdgeErrorPerForecastDay, [4 / 3, 2.0]),
    ],
)
def test_extent_metrics_weight_unequal_batches_by_frame_count(
    metric_type: type[MeanIceAreaMetric], cell_errors: list[float]
) -> None:
    """A one-frame batch and a two-frame batch receive weights one and two."""
    pred = torch.zeros(3, 2, 1, 2, 2)
    truth = torch.zeros_like(pred)
    pred[0, 0, 0, 0, :] = 1
    truth[1:, 0, 0, 0, 0] = 1
    pred[1, 1, 0, 0, :] = 1
    pred[2, 1] = 1
    metric = metric_type(pixel_size=3)
    metric.update(pred[:1], truth[:1])
    metric.update(pred[1:], truth[1:])

    torch.testing.assert_close(metric.compute(), torch.tensor(cell_errors) * 9)
    assert metric.sample_count == 3


def _unequal_edge_support() -> tuple[torch.Tensor, torch.Tensor]:
    """One perfect four-edge forecast followed by two displaced one-edge forecasts."""
    pred = torch.zeros(3, 1, 1, 5, 5)
    truth = torch.zeros_like(pred)
    for y, x in ((1, 1), (1, 3), (3, 1), (3, 3)):
        pred[0, 0, 0, y, x] = truth[0, 0, 0, y, x] = 1
    pred[1:, 0, 0, 3, 3] = 1
    truth[1:, 0, 0, 1, 1] = 1
    return pred, truth


def test_diiee_pools_area_and_edge_length_before_dividing() -> None:
    """Use 2*(0+2+2)*p^2/((8+2+2)*p), not a mean of frame ratios."""
    pred, truth = _unequal_edge_support()
    metric = DistanceAveragedIceEdgeErrorPerForecastDay(pixel_size=9)
    metric.update(pred[:1], truth[:1])
    metric.update(pred[1:], truth[1:])

    torch.testing.assert_close(metric.compute().reshape(-1), torch.tensor([6.0]))


def test_fss_pools_mse_and_reference_before_dividing() -> None:
    """Known sparse-edge supports give 1-(0+2+2)/(8+2+2), not (1+0+0)/3."""
    pred, truth = _unequal_edge_support()
    metric = FractionalSkillScorePerForecastDay(neighbourhood_size=1)
    metric.update(pred[:1], truth[:1])
    metric.update(pred[1:], truth[1:])

    torch.testing.assert_close(metric.compute().reshape(-1), torch.tensor([2 / 3]))


@pytest.mark.parametrize(("size", "expected"), [(1, 0.0), (3, 2 / 3)])
def test_fss_shifted_singletons_match_neighbourhood_overlap(
    size: int, expected: float
) -> None:
    """The two 3x3 fraction footprints overlap at six of nine cells on a 7x9 grid."""
    pred = torch.zeros(1, 2, 1, 7, 9)
    truth = torch.zeros_like(pred)
    pred[:, :, :, 3, 4] = 1
    truth[:, :, :, 3, 3] = 1
    metric = FractionalSkillScorePerForecastDay(neighbourhood_size=size)
    metric.update(pred, truth)

    torch.testing.assert_close(metric.compute(), torch.full((2,), expected))


@pytest.mark.parametrize(
    "metric_type",
    [
        SeaIceExtentErrorPerForecastDay,
        IntegratedIceEdgeErrorPerForecastDay,
        DistanceAveragedIceEdgeErrorPerForecastDay,
        FractionalSkillScorePerForecastDay,
        IceNetAccuracyPerForecastDay,
    ],
)
def test_nonfinite_land_cannot_change_extent_edge_or_accuracy_scores(
    metric_type: type,
) -> None:
    """NaN and both infinities on excluded cells leave the ocean calculation unchanged."""
    pred, truth = _unequal_edge_support()
    pred = pred.expand(-1, 2, -1, -1, -1).clone()
    truth = truth.expand_as(pred).clone()
    mask = torch.ones(5, 5, dtype=torch.bool)
    mask[0, :3] = False
    reference = metric_type(land_mask=mask)
    reference.update(pred, truth)
    pred[:, :, :, 0, :3] = torch.tensor([float("nan"), float("inf"), float("-inf")])
    truth[:, :, :, 0, :3] = torch.tensor([float("inf"), float("-inf"), float("nan")])
    actual = metric_type(land_mask=mask)
    actual.update(pred, truth)

    torch.testing.assert_close(actual.compute(), reference.compute())


def test_weighted_accuracy_accumulates_weight_mass_across_unequal_batches() -> None:
    """One correct weight unit and eight incorrect units give 100/9 percent."""
    truth = torch.ones(3, 2, 1, 1, 2)
    pred = torch.zeros_like(truth)
    pred[0] = 1
    weights = (
        torch.tensor([1.0, 3.0, 5.0]).reshape(3, 1, 1, 1, 1).expand_as(pred).clone()
    )
    weights[..., 1] = 1000
    metric = IceNetAccuracyPerForecastDay(land_mask=torch.tensor([[True, False]]))
    metric.update(pred[:1], truth[:1], sample_weight=weights[:1])
    metric.update(pred[1:], truth[1:], sample_weight=weights[1:])

    torch.testing.assert_close(metric.compute(), torch.full((2,), 100 / 9))


@pytest.mark.parametrize(
    "metric_type",
    [
        SeaIceExtentErrorPerForecastDay,
        IntegratedIceEdgeErrorPerForecastDay,
        DistanceAveragedIceEdgeErrorPerForecastDay,
        FractionalSkillScorePerForecastDay,
        IceNetAccuracyPerForecastDay,
    ],
)
def test_ice_metrics_reset_clears_sums_counts_and_cached_results(
    metric_type: type,
) -> None:
    """A new evaluation after reset must equal a fresh metric, including lead count."""
    pred, truth = _unequal_edge_support()
    metric = metric_type()
    metric.update(pred.expand(-1, 2, -1, -1, -1), truth.expand(-1, 2, -1, -1, -1))
    metric.compute()
    metric.reset()
    metric.update(truth[:1], truth[:1])
    reference = metric_type()
    reference.update(truth[:1], truth[:1])

    torch.testing.assert_close(metric.compute(), reference.compute())
