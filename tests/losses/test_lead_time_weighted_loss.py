import math

import pytest
import torch

from icenet_mp.losses import AMSELoss, LeadTimeWeightedLoss, RMSELoss
from icenet_mp.types import SupportsPerLeadTimeLoss

from .conftest import make_fields

# Four lead times, matching the hand-computed weights in the tests below
SHAPE = (2, 4, 1, 16, 16)


class TestLeadTimeWeightedLoss:
    @pytest.mark.parametrize(
        "exponent", [-1.0, 0.0, 1.0, 2.0], ids=lambda e: f"exponent{e:g}"
    )
    @pytest.mark.parametrize("n_steps", [1, 3, 7], ids=lambda n: f"steps{n}")
    def test_weights_have_mean_one(self, exponent: float, n_steps: int) -> None:
        weights = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent).weights(
            n_steps, torch.device("cpu")
        )
        assert weights.shape == (n_steps,)
        assert weights.mean().item() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("exponent", "expected"),
        [
            (1.0, [0.4, 0.8, 1.2, 1.6]),
            (-1.0, [48 / 25, 24 / 25, 16 / 25, 12 / 25]),
        ],
        ids=["positive", "negative"],
    )
    def test_weights_match_expected_values(
        self, exponent: float, expected: list[float]
    ) -> None:
        weights = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent).weights(
            4, torch.device("cpu")
        )
        assert torch.allclose(weights, torch.tensor(expected))

    @pytest.mark.parametrize(
        ("exponent", "increasing"),
        [(0.5, True), (2.0, True), (-0.5, False), (-2.0, False)],
        ids=["positive-half", "positive-two", "negative-half", "negative-two"],
    )
    def test_weights_monotonic_in_sign_of_exponent(
        self, *, exponent: float, increasing: bool
    ) -> None:
        weights = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent).weights(
            5, torch.device("cpu")
        )
        diffs = weights.diff()
        assert bool((diffs > 0).all()) is increasing
        assert bool((diffs < 0).all()) is not increasing

    @pytest.mark.parametrize(
        ("exponent", "n_steps"),
        [(30.0, 28), (20.0, 90), (-30.0, 28), (-20.0, 90), (1000.0, 7)],
        ids=[
            "pos30-steps28",
            "pos20-steps90",
            "neg30-steps28",
            "neg20-steps90",
            "pos1000-steps7",
        ],
    )
    def test_weights_finite_for_large_exponents(
        self, exponent: float, n_steps: int
    ) -> None:
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent)
        weights = loss_fn.weights(n_steps, torch.device("cpu"))
        assert torch.isfinite(weights).all()
        assert weights.mean().item() == pytest.approx(1.0)
        prediction, target = make_fields(shape=(2, n_steps, 1, 4, 4))
        assert torch.isfinite(loss_fn(prediction, target))

    def test_weights_cached_per_n_steps(self) -> None:
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), 1.5)
        cpu = torch.device("cpu")
        assert loss_fn.weights(7, cpu) is loss_fn.weights(7, cpu)
        assert loss_fn.weights(7, cpu) is not loss_fn.weights(8, cpu)

    def test_weights_cached_in_inference_mode_support_backward(self) -> None:
        # Mirrors Lightning's validation sanity check running before training
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), 2.5)
        prediction, target = make_fields(shape=(2, 5, 1, 4, 4))
        with torch.inference_mode():
            loss_fn(prediction, target)
        prediction.requires_grad_()
        loss_fn(prediction, target).backward()
        assert prediction.grad is not None

    @pytest.mark.parametrize(
        "exponent", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"]
    )
    def test_rejects_non_finite_exponent(self, exponent: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent)

    @pytest.mark.parametrize("reduction", ["sum", "none"])
    def test_rejects_non_mean_reduction(self, reduction: str) -> None:
        with pytest.raises(ValueError, match="mean-reduced"):
            LeadTimeWeightedLoss(torch.nn.MSELoss(reduction=reduction))

    @pytest.mark.parametrize(
        ("dtype", "rel"),
        [(torch.bfloat16, 1e-2), (torch.float16, 1e-3), (torch.float64, 1e-6)],
        ids=["bfloat16", "float16", "float64"],
    )
    @pytest.mark.parametrize(
        "base", [torch.nn.MSELoss(), AMSELoss()], ids=["mse", "amse"]
    )
    def test_supports_other_dtypes(
        self, dtype: torch.dtype, rel: float, base: torch.nn.Module
    ) -> None:
        prediction, target = make_fields(shape=SHAPE)
        loss_fn = LeadTimeWeightedLoss(base, exponent=1.0)
        reference = loss_fn(prediction, target)
        prediction = prediction.to(dtype).requires_grad_()
        loss = loss_fn(prediction, target.to(dtype))
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(reference.item(), rel=rel)
        loss.backward()
        assert prediction.grad is not None
        assert prediction.grad.dtype == dtype

    @pytest.mark.parametrize(
        "base",
        [
            torch.nn.MSELoss(),
            torch.nn.L1Loss(),
            torch.nn.HuberLoss(delta=0.5),
            torch.nn.SmoothL1Loss(beta=0.5),
            AMSELoss(mode="hybrid"),
            AMSELoss(mode="pure"),
            AMSELoss(wavenumber_weight="fastnet"),
        ],
        ids=[
            "mse",
            "mae",
            "huber",
            "smooth_l1",
            "amse-hybrid",
            "amse-pure",
            "amse-fastnet",
        ],
    )
    def test_zero_exponent_matches_base(self, base: torch.nn.Module) -> None:
        prediction, target = make_fields(shape=SHAPE)
        loss = LeadTimeWeightedLoss(base, exponent=0.0)(prediction, target)
        assert loss.item() == pytest.approx(base(prediction, target).item())

    def test_zero_exponent_approximates_rmse(self) -> None:
        # The mean of per-step RMSEs is at most the RMSE over all steps (Jensen),
        # with equality only when every step has the same MSE
        prediction, target = make_fields(shape=SHAPE)
        base = RMSELoss()
        loss = LeadTimeWeightedLoss(base, exponent=0.0)(prediction, target).item()
        reference = base(prediction, target).item()
        assert loss <= reference
        assert loss == pytest.approx(reference, rel=1e-3)

    def test_matches_manual_weighted_sum(self) -> None:
        prediction, target = make_fields(shape=SHAPE)
        base = torch.nn.MSELoss()
        per_step = torch.stack([base(prediction[:, t], target[:, t]) for t in range(4)])
        expected = (torch.tensor([0.4, 0.8, 1.2, 1.6]) * per_step).mean()
        loss = LeadTimeWeightedLoss(base, exponent=1.0)(prediction, target)
        assert loss.item() == pytest.approx(expected.item())

    @pytest.mark.parametrize(
        ("exponent", "late_costs_more"),
        [(1.0, True), (-1.0, False)],
        ids=["positive", "negative"],
    )
    def test_sign_of_exponent_sets_which_errors_cost_more(
        self, *, exponent: float, late_costs_more: bool
    ) -> None:
        _, target = make_fields(shape=SHAPE)
        early_error = target.clone()
        early_error[:, 0] += 0.5
        late_error = target.clone()
        late_error[:, -1] += 0.5
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent=exponent)
        late_loss, early_loss = (
            loss_fn(late_error, target),
            loss_fn(early_error, target),
        )
        assert bool(late_loss > early_loss) is late_costs_more
        assert bool(early_loss > late_loss) is not late_costs_more

    @pytest.mark.parametrize("base", [RMSELoss(), AMSELoss()], ids=["rmse", "amse"])
    def test_wraps_reducing_losses(self, base: torch.nn.Module) -> None:
        prediction, target = make_fields(shape=SHAPE)
        prediction.requires_grad_()
        loss = LeadTimeWeightedLoss(base, exponent=1.0)(prediction, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        loss.backward()
        assert prediction.grad is not None

    def test_uses_per_lead_time_loss_when_available(self) -> None:
        prediction, target = make_fields(shape=SHAPE)
        base = AMSELoss()
        assert isinstance(base, SupportsPerLeadTimeLoss)
        assert not isinstance(torch.nn.MSELoss(), SupportsPerLeadTimeLoss)
        per_step = torch.stack([base(prediction[:, t], target[:, t]) for t in range(4)])
        expected = (torch.tensor([0.4, 0.8, 1.2, 1.6]) * per_step).mean()
        loss = LeadTimeWeightedLoss(base, exponent=1.0)(prediction, target)
        assert loss.item() == pytest.approx(expected.item())

    def test_rejects_inputs_without_lead_time(self) -> None:
        prediction, target = make_fields(shape=(2, 1, 16, 16))
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent=1.0)
        with pytest.raises(ValueError, match="NTCHW"):
            loss_fn(prediction, target)

    @pytest.mark.parametrize("target_steps", [3, 5], ids=["fewer", "more"])
    def test_rejects_mismatched_shapes(self, target_steps: int) -> None:
        prediction, _ = make_fields(shape=SHAPE)
        _, target = make_fields(shape=(2, target_steps, 1, 16, 16))
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent=1.0)
        with pytest.raises(ValueError, match="same shape"):
            loss_fn(prediction, target)
