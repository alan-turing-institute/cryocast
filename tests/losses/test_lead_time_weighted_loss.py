import pytest
import torch

from icenet_mp.losses import LeadTimeWeightedLoss
from icenet_mp.losses.amse_loss import AMSELoss
from icenet_mp.losses.lead_time_weighted_loss import SupportsPerLeadTimeLoss
from icenet_mp.losses.rmse_loss import RMSELoss


def make_fields(
    seed: int = 0, shape: tuple[int, ...] = (2, 4, 1, 16, 16)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded (prediction, target) pair of random NTCHW fields."""
    generator = torch.Generator().manual_seed(seed)
    prediction = torch.rand(*shape, generator=generator)
    target = torch.rand(*shape, generator=generator)
    return prediction, target


class TestLeadTimeWeightedLoss:
    @pytest.mark.parametrize("exponent", [-1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("n_steps", [1, 3, 7])
    def test_weights_have_mean_one(self, exponent: float, n_steps: int) -> None:
        weights = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent).weights(
            n_steps, torch.device("cpu")
        )
        assert weights.shape == (n_steps,)
        assert weights.mean().item() == pytest.approx(1.0)

    def test_weights_increase_with_lead_time(self) -> None:
        weights = LeadTimeWeightedLoss(torch.nn.MSELoss(), 1.0).weights(
            4, torch.device("cpu")
        )
        assert torch.allclose(weights, torch.tensor([0.4, 0.8, 1.2, 1.6]))

    @pytest.mark.parametrize(
        "base", [torch.nn.MSELoss(), torch.nn.L1Loss()], ids=["mse", "mae"]
    )
    def test_zero_exponent_matches_base(self, base: torch.nn.Module) -> None:
        prediction, target = make_fields()
        loss = LeadTimeWeightedLoss(base, exponent=0.0)(prediction, target)
        assert loss.item() == pytest.approx(base(prediction, target).item())

    def test_matches_manual_weighted_sum(self) -> None:
        prediction, target = make_fields()
        base = torch.nn.MSELoss()
        per_step = torch.stack([base(prediction[:, t], target[:, t]) for t in range(4)])
        expected = (torch.tensor([0.4, 0.8, 1.2, 1.6]) * per_step).mean()
        loss = LeadTimeWeightedLoss(base, exponent=1.0)(prediction, target)
        assert loss.item() == pytest.approx(expected.item())

    def test_late_errors_cost_more(self) -> None:
        _, target = make_fields()
        early_error = target.clone()
        early_error[:, 0] += 0.5
        late_error = target.clone()
        late_error[:, -1] += 0.5
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent=1.0)
        assert loss_fn(late_error, target) > loss_fn(early_error, target)

    @pytest.mark.parametrize("base", [RMSELoss(), AMSELoss()], ids=["rmse", "amse"])
    def test_wraps_reducing_losses(self, base: torch.nn.Module) -> None:
        prediction, target = make_fields()
        prediction.requires_grad_()
        loss = LeadTimeWeightedLoss(base, exponent=1.0)(prediction, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        loss.backward()
        assert prediction.grad is not None

    def test_uses_per_lead_time_loss_when_available(self) -> None:
        prediction, target = make_fields()
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
        prediction, _ = make_fields(shape=(2, 4, 1, 16, 16))
        _, target = make_fields(shape=(2, target_steps, 1, 16, 16))
        loss_fn = LeadTimeWeightedLoss(torch.nn.MSELoss(), exponent=1.0)
        with pytest.raises(ValueError, match="same shape"):
            loss_fn(prediction, target)
