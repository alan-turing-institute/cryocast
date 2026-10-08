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
