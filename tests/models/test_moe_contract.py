"""Analytical routing, supervision-isolation and precision checks for dense MoE."""

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from cryocast.models.processors import BaseProcessor, MixtureOfExpertsProcessor
from cryocast.types import DataSpace, ProcessorOutput, TensorNTCHW


class ScaleExpert(BaseProcessor):
    """An expert with an exactly differentiable, constant-in-time forecast."""

    def __init__(self, *, gain: float = 1.0, **kwargs: Any) -> None:
        """Create one trainable gain."""
        super().__init__(**kwargs)
        self.gain = nn.Parameter(torch.tensor(gain))

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        del y
        prediction = self.gain * x[:, -1:].expand(-1, self.n_forecast_steps, -1, -1, -1)
        return ProcessorOutput(prediction=prediction)


class TargetReadingExpert(ScaleExpert):
    """Reveal any future-target forwarding by the prediction-only wrapper."""

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        if y is not None:
            return ProcessorOutput(prediction=y)
        return super().rollout(x)


class MalformedExpert(ScaleExpert):
    """Return internally consistent but contract-invalid forecasts."""

    def __init__(self, *, axis: int, **kwargs: Any) -> None:
        """Choose the axis to truncate."""
        super().__init__(**kwargs)
        self.axis = axis

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        result = super().rollout(x, y).prediction
        return ProcessorOutput(prediction=result.narrow(self.axis, 0, 1))


class HiddenLossExpert(ScaleExpert):
    """Exercise runtime enforcement even when the expert's flag is incorrect."""

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        return ProcessorOutput(
            prediction=super().rollout(x, y).prediction, loss=x.sum()
        )


class NonProcessor:
    """Accept Hydra's constructor arguments without implementing the interface."""

    def __init__(self, **kwargs: Any) -> None:
        """Accept but ignore shared processor settings."""
        del kwargs


def _spec(cls: type = ScaleExpert, **options: Any) -> DictConfig:
    config = {"_target_": f"{__name__}.{cls.__name__}"}
    config.update(options)
    return DictConfig(config)


def _mixture(
    experts: list[DictConfig] | None = None, **kwargs: Any
) -> MixtureOfExpertsProcessor:
    options = {
        "data_space": DataSpace(name="combined", channels=2, shape=(4, 4)),
        "n_history_steps": 2,
        "n_forecast_steps": 3,
        "gate_hidden_channels": 4,
        **kwargs,
    }
    return MixtureOfExpertsProcessor(
        experts=[_spec(gain=2.0), _spec(gain=-1.0)] if experts is None else experts,
        **options,
    )


def _fixed_gate(model: MixtureOfExpertsProcessor) -> nn.Linear:
    with torch.no_grad():
        for parameter in model.gate.parameters():
            parameter.zero_()
        final = model.gate[-1]
        assert isinstance(final, nn.Linear)
        assert final.bias is not None
        final.bias.copy_(torch.tensor([math.log(3.0), 0.0], dtype=final.bias.dtype))
    return final


def test_exact_convex_mixture_and_analytic_gradients() -> None:
    """Check both expert and softmax-gate derivatives against their closed forms."""
    model = _mixture().double()
    final = _fixed_gate(model)
    x = torch.linspace(0.1, 1.0, 128, dtype=torch.float64).reshape(2, 2, 2, 4, 4)
    x.requires_grad_()
    result = model.rollout(x)
    weights = torch.tensor([0.75, 0.25], dtype=torch.float64)
    gains = torch.tensor([2.0, -1.0], dtype=torch.float64)
    combined_gain = (weights * gains).sum()
    expected = combined_gain * x[:, -1:].expand(-1, 3, -1, -1, -1)
    torch.testing.assert_close(result.prediction, expected, atol=1e-14, rtol=1e-14)
    assert result.loss is None
    result.prediction.square().mean().backward()
    common = 2 * combined_gain * x[:, -1].detach().square().mean()
    for index, expert in enumerate(model.experts):
        assert isinstance(expert, ScaleExpert)
        torch.testing.assert_close(expert.gain.grad, common * weights[index])
    assert final.bias is not None
    torch.testing.assert_close(
        final.bias.grad, common * weights * (gains - combined_gain)
    )
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[:, 0]) == 0
    torch.testing.assert_close(
        x.grad[:, -1], 2 * combined_gain.square() * x[:, -1] / x[:, -1].numel()
    )


def test_prediction_only_experts_never_receive_future_targets() -> None:
    """Supplying y must not enable an expert to cheat during training."""
    model = _mixture([_spec(TargetReadingExpert)])
    history = torch.rand(2, 2, 2, 4, 4, requires_grad=True)
    future = torch.full((2, 3, 2, 4, 4), 50.0, requires_grad=True)
    result = model.rollout(history, future).prediction
    torch.testing.assert_close(
        result, model.rollout(history).prediction, atol=0, rtol=0
    )
    result.sum().backward()
    assert future.grad is None
    assert history.grad is not None


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4])
def test_each_expert_must_return_full_forecast_shape(axis: int) -> None:
    """Even identical wrong shapes from every expert must fail before blending."""
    model = _mixture([_spec(MalformedExpert, axis=axis)] * 2)
    with pytest.raises(ValueError, match=r"Expert.*shape"):
        model.rollout(torch.zeros(2, 2, 2, 4, 4))


@pytest.mark.parametrize(
    "shape",
    [(2, 2, 4, 4), (2, 1, 2, 4, 4), (2, 2, 3, 4, 4), (2, 2, 2, 2, 8), (2, 0, 2, 4, 4)],
)
def test_history_axes_are_checked_before_routing(shape: tuple[int, ...]) -> None:
    """Reject malformed histories rather than silently pooling the wrong axes."""
    model = _mixture()
    with pytest.raises(ValueError, match=r"history.*shape"):
        model.rollout(torch.zeros(shape))


@pytest.mark.parametrize("value", [True, 1.5, "4", 0, -1])
def test_hidden_width_is_a_positive_integer(value: object) -> None:
    """Reject truncation, Boolean widths and misleading low-level type errors."""
    with pytest.raises(ValueError, match="gate_hidden_channels"):
        _mixture(gate_hidden_channels=value)


def test_wrapper_cannot_claim_to_compute_latent_loss() -> None:
    """Avoid freezing the decoder for a wrapper that only supplies predictions."""
    with pytest.raises(ValueError, match="prediction-only"):
        _mixture(computes_loss_in_latent_space=True)


def test_latent_loss_expert_is_rejected() -> None:
    """A mixture loss policy is required before accepting latent-loss experts."""
    with pytest.raises(ValueError, match="latent-space loss"):
        _mixture([_spec(computes_loss_in_latent_space=True)])


def test_undeclared_custom_loss_is_rejected_at_runtime() -> None:
    """Do not silently discard an unexpected expert objective."""
    model = _mixture([_spec(HiddenLossExpert)])
    with pytest.raises(ValueError, match="prediction-only"):
        model.rollout(torch.zeros(2, 2, 2, 4, 4))


def test_nonprocessor_expert_is_rejected() -> None:
    """Hydra targets must actually implement BaseProcessor."""
    with pytest.raises(TypeError, match="inherit BaseProcessor"):
        _mixture([_spec(NonProcessor)])


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_gate_precision_and_output_dtype(dtype: torch.dtype) -> None:
    """Pool and project safely while preserving the experts' output precision."""
    model = _mixture().to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.gate.parameters():
            parameter.fill_(100.0)
    history = torch.full((2, 2, 2, 4, 4), 100.0, dtype=dtype)
    weights = model.expert_weights(history)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(
        weights.double(), torch.full((2, 2), 0.5, dtype=torch.float64)
    )
    output = model.rollout(history).prediction
    assert output.dtype == dtype
    torch.testing.assert_close(output, torch.full_like(output, 50.0), rtol=0, atol=0)


def test_cpu_autocast_does_not_overflow_gate() -> None:
    """Disabling autocast only after a projection has overflowed is too late."""
    model = _mixture()
    with torch.no_grad():
        for parameter in model.gate.parameters():
            parameter.fill_(100.0)
    history = torch.full((2, 2, 2, 4, 4), 100.0)
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        weights = model.expert_weights(history)
        output = model.rollout(history).prediction
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(output, torch.full_like(output, 50.0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_work_with_float32_gate(dtype: torch.dtype) -> None:
    """A float32 gate must accept low-precision encoder outputs without autocast."""
    model = _mixture()
    history = torch.randn(2, 2, 2, 4, 4).to(dtype)
    output = model.rollout(history).prediction
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("seed", [0, 490, 1234])
def test_routing_is_per_sample_and_uses_only_latest_frame(seed: int) -> None:
    """Batch neighbours, permutation and earlier history cannot alter a gate."""
    torch.manual_seed(seed)
    model = _mixture().eval()
    history = torch.randn(3, 2, 2, 4, 4)
    weights = model.expert_weights(history)
    torch.testing.assert_close(
        weights, torch.cat([model.expert_weights(row[None]) for row in history])
    )
    modified = history.clone()
    modified[:, 0] = 999
    torch.testing.assert_close(model.expert_weights(modified), weights, rtol=0, atol=0)
    torch.testing.assert_close(model.expert_weights(history.flip(0)), weights.flip(0))
    torch.testing.assert_close(weights.sum(-1), torch.ones(3))
    assert not torch.equal(weights[0], weights[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_expert_matches_direct_forecast_and_gradients(
    dtype: torch.dtype,
) -> None:
    """With one expert there is no routing effect or invented auxiliary objective."""
    mixture = _mixture([_spec(gain=1.75)]).to(dtype)
    expert = mixture.experts[0]
    assert isinstance(expert, ScaleExpert)
    x = torch.randn(2, 2, 2, 4, 4, dtype=dtype, requires_grad=True)
    actual = mixture.rollout(x).prediction
    actual.sum().backward()
    actual_gradient = x.grad.clone() if x.grad is not None else None
    assert actual_gradient is not None
    x.grad = None
    expected = expert.rollout(x).prediction
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, x.grad, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 490])
def test_checkpoint_restores_prediction_and_next_adam_step(
    tmp_path: Path, seed: int
) -> None:
    """The routing distribution and expert parameters survive a safe roundtrip."""
    torch.manual_seed(seed)
    original = _mixture().double()
    optimiser = torch.optim.Adam(original.parameters(), lr=1e-3)
    history = torch.randn(2, 2, 2, 4, 4, dtype=torch.float64)

    def step(model: MixtureOfExpertsProcessor, opt: torch.optim.Optimizer) -> None:
        opt.zero_grad()
        model.rollout(history).prediction.square().mean().backward()
        opt.step()

    step(original, optimiser)
    path = tmp_path / "mixture.pt"
    torch.save(
        {"model": original.state_dict(), "optimizer": optimiser.state_dict()}, path
    )
    checkpoint = torch.load(path, weights_only=True)
    restored = _mixture().double()
    restored.load_state_dict(checkpoint["model"], strict=True)
    resumed = torch.optim.Adam(restored.parameters(), lr=1e-3)
    resumed.load_state_dict(checkpoint["optimizer"])
    torch.testing.assert_close(
        original.rollout(history).prediction,
        restored.rollout(history).prediction,
        rtol=0,
        atol=0,
    )
    step(original, optimiser)
    step(restored, resumed)
    torch.testing.assert_close(
        original.state_dict(), restored.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        optimiser.state_dict(), resumed.state_dict(), rtol=0, atol=0
    )
