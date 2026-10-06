import pytest
import torch
from torch.nn import functional

from icenet_mp.losses import UncertaintyWeightedLoss


def test_lower_uncertainty_gives_more_weight() -> None:
    """Give lower-uncertainty observations greater influence on the loss."""
    loss_fn = UncertaintyWeightedLoss(delta=0.5, power=2.0)
    preds = torch.tensor([[[[[1.0, 2.0]]]]])
    targets = torch.zeros_like(preds)
    uncertainty = torch.tensor([[[[[0.1, 0.2]]]]])

    result = loss_fn(preds, targets, uncertainty)

    pointwise = functional.huber_loss(preds, targets, reduction="none", delta=0.5)
    weights = torch.tensor([[[[[100.0, 25.0]]]]])
    expected = (pointwise * weights).sum() / weights.sum()
    torch.testing.assert_close(result, expected)
    assert result < pointwise.mean()


def test_invalid_and_sentinel_uncertainty_are_excluded() -> None:
    """Exclude invalid and sentinel-like uncertainty values from weighting."""
    loss_fn = UncertaintyWeightedLoss(max_uncertainty=1.0)
    preds = torch.tensor([[[[[1.0, 10.0, 10.0, 10.0]]]]])
    targets = torch.zeros_like(preds)
    uncertainty = torch.tensor([[[[[0.1, 0.0, float("nan"), 99.0]]]]])

    result = loss_fn(preds, targets, uncertainty)
    expected = functional.huber_loss(preds[..., :1], targets[..., :1], delta=0.5)

    torch.testing.assert_close(result, expected)


def test_all_invalid_uncertainty_falls_back_to_unweighted_huber() -> None:
    """Use ordinary Huber loss when a batch has no valid uncertainty values."""
    loss_fn = UncertaintyWeightedLoss()
    preds = torch.tensor([1.0, 2.0])
    targets = torch.zeros_like(preds)
    uncertainty = torch.tensor([0.0, 99.0])

    result = loss_fn(preds, targets, uncertainty)
    expected = functional.huber_loss(preds, targets, delta=0.5)

    torch.testing.assert_close(result, expected)


def test_min_uncertainty_clamps_extreme_weights() -> None:
    """Clamp very small valid uncertainty values before computing weights."""
    loss_fn = UncertaintyWeightedLoss(min_uncertainty=0.05)
    preds = torch.tensor([1.0, 2.0])
    targets = torch.zeros_like(preds)
    uncertainty = torch.tensor([1e-8, 0.05])

    result = loss_fn(preds, targets, uncertainty)
    pointwise = functional.huber_loss(preds, targets, reduction="none", delta=0.5)

    torch.testing.assert_close(result, pointwise.mean())


def test_requires_uncertainty_tensor() -> None:
    """Require an uncertainty tensor for uncertainty-weighted loss."""
    loss_fn = UncertaintyWeightedLoss()

    with pytest.raises(ValueError, match="requires an uncertainty tensor"):
        loss_fn(torch.ones(1), torch.zeros(1))


def test_rejects_shape_mismatch() -> None:
    """Reject uncertainty tensors that do not match the target shape."""
    loss_fn = UncertaintyWeightedLoss()

    with pytest.raises(ValueError, match="does not match target shape"):
        loss_fn(torch.zeros(2), torch.zeros(2), torch.ones(1))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delta": 0.0}, "delta"),
        ({"min_uncertainty": 0.0}, "min_uncertainty"),
        (
            {"min_uncertainty": 0.5, "max_uncertainty": 0.1},
            "max_uncertainty",
        ),
        ({"power": 0.0}, "power"),
        ({"uncertainty_variable": ""}, "uncertainty_variable"),
    ],
)
def test_rejects_invalid_configuration(
    kwargs: dict[str, float | str], message: str
) -> None:
    """Reject invalid uncertainty-loss configuration values."""
    with pytest.raises(ValueError, match=message):
        UncertaintyWeightedLoss(**kwargs)  # type: ignore[arg-type]
