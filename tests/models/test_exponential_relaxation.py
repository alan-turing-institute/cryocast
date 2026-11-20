import math

import pytest
import torch

from cryocast.models.common import ExponentialRelaxation


@pytest.fixture
def state() -> torch.Tensor:
    """Return a state in [0, 1]."""
    return torch.rand(2, 3, 4, 5, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def equilibrium() -> torch.Tensor:
    """Return an equilibrium in [0, 1]."""
    return torch.rand(2, 3, 4, 5, generator=torch.Generator().manual_seed(1))


class TestExponentialRelaxationConstruction:
    def test_rate_matches_tau(self) -> None:
        relaxation = ExponentialRelaxation(tau=30)
        assert relaxation.tau == 30.0
        assert relaxation.rate == pytest.approx(1.0 - math.exp(-1.0 / 30.0))

    @pytest.mark.parametrize("tau", [0.0, -1.0], ids=["zero", "negative"])
    def test_rejects_non_positive_tau(self, tau: float) -> None:
        with pytest.raises(ValueError, match="tau must be positive"):
            ExponentialRelaxation(tau=tau)

    def test_has_no_state(self) -> None:
        # No parameters or buffers, so adding the module never changes a checkpoint
        assert ExponentialRelaxation(tau=30.0).state_dict() == {}


class TestExponentialRelaxationForward:
    def test_full_rate_returns_equilibrium(
        self, state: torch.Tensor, equilibrium: torch.Tensor
    ) -> None:
        # A very short tau gives lambda = 1 exactly in floating point
        relaxation = ExponentialRelaxation(tau=1e-3)
        assert relaxation.rate == 1.0
        assert torch.equal(relaxation(state, equilibrium), equilibrium)

    def test_anomaly_decays_exponentially(
        self, state: torch.Tensor, equilibrium: torch.Tensor
    ) -> None:
        # With no other change per step and a constant equilibrium, the anomaly
        # after k steps is exp(-k / tau) times the initial anomaly
        tau = 4.0
        relaxation = ExponentialRelaxation(tau=tau)
        x = state
        for k in range(1, 31):
            x = relaxation(x, equilibrium)
            expected = equilibrium + math.exp(-k / tau) * (state - equilibrium)
            torch.testing.assert_close(x, expected)

    def test_output_stays_between_state_and_equilibrium(self) -> None:
        # Values at the edges of [0, 1] must not overshoot over a long rollout
        state = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        equilibrium = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        lower = torch.minimum(state, equilibrium)
        upper = torch.maximum(state, equilibrium)
        relaxation = ExponentialRelaxation(tau=2.0)
        x = state
        for _ in range(90):
            x = relaxation(x, equilibrium)
            assert torch.all(x >= lower)
            assert torch.all(x <= upper)

    def test_gradient_flows_to_state(
        self, state: torch.Tensor, equilibrium: torch.Tensor
    ) -> None:
        relaxation = ExponentialRelaxation(tau=5.0)
        state = state.clone().requires_grad_()
        relaxation(state, equilibrium).sum().backward()
        assert state.grad is not None
        torch.testing.assert_close(
            state.grad, torch.full_like(state, 1.0 - relaxation.rate)
        )

    @pytest.mark.parametrize(
        ("state_dtype", "equilibrium_dtype"),
        [
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
        ],
        ids=["bf16-state", "bf16-equilibrium"],
    )
    def test_mixed_dtypes_use_the_wider_dtype(
        self,
        state_dtype: torch.dtype,
        equilibrium_dtype: torch.dtype,
    ) -> None:
        # Under bf16-mixed, the state can be bf16 and the climatology float32. A small
        # step must not round to no change, so the result is float32.
        state = torch.full((1, 1, 2, 2), 0.5, dtype=state_dtype)
        equilibrium = torch.full((1, 1, 2, 2), 0.4, dtype=equilibrium_dtype)
        relaxation = ExponentialRelaxation(tau=90.0)
        relaxed = relaxation(state, equilibrium)
        assert relaxed.dtype == torch.float32
        assert torch.all(relaxed < state.float())

    def test_rejects_mismatched_equilibrium_shape(self, state: torch.Tensor) -> None:
        relaxation = ExponentialRelaxation(tau=30.0)
        with pytest.raises(ValueError, match="does not match state shape"):
            relaxation(state, torch.zeros(2, 3, 4, 4))
