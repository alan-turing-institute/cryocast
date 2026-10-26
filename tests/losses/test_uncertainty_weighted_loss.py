import pytest
import torch
from torch.nn import functional

from cryocast.losses import UncertaintyWeightedLoss


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


@pytest.mark.parametrize(
    "name", ["delta", "min_uncertainty", "max_uncertainty", "power"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_parameters_are_rejected(name: str, value: float) -> None:
    """Reject NaN and infinity before they can silently disable weighting."""
    with pytest.raises(ValueError, match=name):
        UncertaintyWeightedLoss(**{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision_large_images_have_finite_loss_and_gradient(
    dtype: torch.dtype,
) -> None:
    """A normal image must not overflow the inverse-variance weight reduction."""
    preds = torch.full((1, 1, 1, 64, 64), 0.5, dtype=dtype, requires_grad=True)
    targets = torch.zeros_like(preds)
    sigma = torch.full_like(preds, 0.01)
    value = UncertaintyWeightedLoss()(preds, targets, sigma)
    reference = functional.huber_loss(preds.float(), targets.float(), delta=0.5)
    torch.testing.assert_close(value.float(), reference, atol=1e-4, rtol=1e-4)
    value.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
    assert preds.grad.abs().sum() > 0


def test_invalid_uncertainty_excludes_nonfinite_pixels_before_huber() -> None:
    """A masked NaN observation must not poison the loss or valid gradients."""
    preds = torch.tensor([0.5, 0.2, 0.3], requires_grad=True)
    target = torch.tensor([0.0, float("nan"), float("inf")])
    sigma = torch.tensor([0.1, 0.0, float("nan")])
    result = UncertaintyWeightedLoss()(preds, target, sigma)
    torch.testing.assert_close(result, torch.tensor(0.125))
    result.backward()
    torch.testing.assert_close(preds.grad, torch.tensor([0.5, 0.0, 0.0]))


def test_high_power_preserves_uniform_weight_equivalence() -> None:
    """Uniform uncertainties still equal Huber when inverse powers would overflow."""
    preds = torch.tensor([0.2, 0.7], requires_grad=True)
    target = torch.zeros_like(preds)
    sigma = torch.full_like(preds, 0.01)
    value = UncertaintyWeightedLoss(power=64.0)(preds, target, sigma)
    expected = functional.huber_loss(preds, target, delta=0.5)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_float64_inputs_keep_double_precision() -> None:
    """Do not demote double precision during numerically safe accumulation."""
    preds = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(preds)
    sigma = torch.tensor([0.1, 0.2], dtype=torch.float64)
    actual = UncertaintyWeightedLoss()(preds, target, sigma)
    expected = (
        functional.huber_loss(preds, target, reduction="none", delta=0.5)
        * torch.tensor([4.0, 1.0], dtype=torch.float64)
    ).sum() / 5
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)


def test_uniform_uncertainty_matches_huber_gradients() -> None:
    """Uniform weighting leaves both the Huber objective and its gradient intact."""
    a = torch.tensor([0.1, 0.8, -0.2], dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    targets = torch.zeros_like(a)
    weighted = UncertaintyWeightedLoss()(a, targets, torch.full_like(a, 0.03))
    ordinary = functional.huber_loss(b, targets, delta=0.5)
    weighted.backward()
    ordinary.backward()
    torch.testing.assert_close(weighted, ordinary)
    torch.testing.assert_close(a.grad, b.grad)


def test_prediction_shape_mismatch_is_explicit() -> None:
    """Do not silently broadcast predictions before applying a pixel mask."""
    with pytest.raises(ValueError, match="Prediction shape"):
        UncertaintyWeightedLoss()(torch.zeros(1), torch.zeros(2), torch.ones(2))
