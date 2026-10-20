"""Analytical and sampler regressions for diffusion schedules."""

import math

import numpy as np
import pytest
import torch

from cryocast.models.diffusion import GaussianDiffusion
from cryocast.models.processors import DiffusionProcessor
from cryocast.types import BetaSchedule, DataSpace


@pytest.mark.parametrize("schedule", list(BetaSchedule))
@pytest.mark.parametrize(
    "steps", [0, -1, True, False, 1.5, float("nan"), float("inf"), "4", None]
)
def test_schedule_rejects_non_positive_or_non_integer_lengths(
    schedule: BetaSchedule, steps: object
) -> None:
    """Reject invalid lengths before creating empty arrays or dividing by zero."""
    with pytest.raises(ValueError, match="timesteps must be a positive integer"):
        GaussianDiffusion(timesteps=steps, beta_schedule=schedule)  # type: ignore[arg-type]


@pytest.mark.parametrize("schedule", list(BetaSchedule))
def test_schedule_accepts_numpy_integer_length(schedule: BetaSchedule) -> None:
    """Integer scalar configuration values retain the same schedule."""
    actual = GaussianDiffusion(timesteps=np.int64(4), beta_schedule=schedule)  # type: ignore[arg-type]
    expected = GaussianDiffusion(timesteps=4, beta_schedule=schedule)
    torch.testing.assert_close(actual.betas, expected.betas, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [1, 2, 4, 10, 19, 20, 21, 50, 150, 1000])
def test_linear_schedule_matches_independent_float64_reference(steps: int) -> None:
    """Check beta scaling and posterior coefficients against independent arithmetic."""
    d = GaussianDiffusion(timesteps=steps, beta_schedule=BetaSchedule.LINEAR)
    beta = np.minimum(
        np.array([20.0])
        if steps == 1
        else np.linspace(0.1 / steps, 20.0 / steps, steps),
        0.999,
    )
    alpha = 1.0 - beta
    product = np.cumprod(alpha)
    previous = np.concatenate(([1.0], product[:-1]))
    variance = beta * (1.0 - previous) / (1.0 - product)
    expected = {
        "betas": beta,
        "alphas": alpha,
        "alphas_cumprod": product,
        "alphas_cumprod_prev": previous,
        "sqrt_alphas_cumprod": np.sqrt(product),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - product),
        "sqrt_recip_alphas": np.sqrt(1.0 / alpha),
        "posterior_variance": variance,
        "posterior_log_variance_clipped": np.log(np.maximum(variance, 1e-20)),
        "posterior_mean_coef1": beta * np.sqrt(previous) / (1.0 - product),
        "posterior_mean_coef2": (1.0 - previous) * np.sqrt(alpha) / (1.0 - product),
    }
    for name, reference in expected.items():
        actual = getattr(d, name)
        assert actual.shape == (steps,), name
        assert torch.isfinite(actual).all(), name
        # Float32 1-alpha cancellation is largest at the first 1000-step beta.
        torch.testing.assert_close(
            actual.double(), torch.from_numpy(reference), rtol=3e-4, atol=2e-7
        )
    assert d.posterior_variance[0] == 0
    assert torch.all(d.posterior_variance >= 0)
    assert torch.all(d.posterior_variance <= d.betas)


@pytest.mark.parametrize("steps", [2, 4, 10, 20, 50, 150, 1000])
def test_terminal_forward_corruption_uses_the_scaled_schedule(steps: int) -> None:
    """Test the actual terminal forward method, not only the coefficient table."""
    diffusion = GaussianDiffusion(timesteps=steps, beta_schedule=BetaSchedule.LINEAR)
    clean = torch.linspace(-1, 1, 32).reshape(2, 1, 4, 4)
    noise = torch.ones_like(clean) * 0.25
    times = torch.full((2,), steps - 1, dtype=torch.long)
    product = float(diffusion.alphas_cumprod[-1])
    expected = math.sqrt(product) * clean + math.sqrt(1 - product) * noise
    noisy = diffusion.q_sample(clean, times, noise)
    torch.testing.assert_close(noisy, expected)
    assert product / (1 - product) < 1e-3
    velocity = diffusion.calculate_v(clean, noise, times)
    recovered = math.sqrt(product) * noisy - math.sqrt(1 - product) * velocity
    torch.testing.assert_close(recovered, clean, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("schedule", list(BetaSchedule))
@pytest.mark.parametrize("steps", [1, 2, 4, 20, 50, 150, 1000])
def test_schedule_tables_and_reverse_step_are_finite(
    schedule: BetaSchedule, steps: int
) -> None:
    """Exercise every trained index, including one-step posterior edge cases."""
    diffusion = GaussianDiffusion(timesteps=steps, beta_schedule=schedule)
    for name, value in vars(diffusion).items():
        if isinstance(value, torch.Tensor):
            assert value.shape == (steps,), name
            assert torch.isfinite(value).all(), name
    times = torch.arange(steps)
    noisy = torch.ones(steps, 1, 2, 2)
    velocity = torch.full_like(noisy, 0.2)
    sampled = diffusion.p_sample(noisy, times, velocity)
    assert sampled.shape == noisy.shape
    assert torch.isfinite(sampled).all()


def test_canonical_linear_schedule_is_bitwise_unchanged() -> None:
    """Keep the existing 1000-step linear coefficients exactly unchanged."""
    d = GaussianDiffusion(timesteps=1000, beta_schedule=BetaSchedule.LINEAR)
    betas = torch.linspace(1e-4, 0.02, 1000)
    assert torch.equal(d.betas, betas)
    assert torch.equal(d.alphas_cumprod, torch.cumprod(1.0 - betas, dim=0))


@pytest.mark.parametrize("steps", [1, 4, 50, 150, 1000])
def test_cosine_schedule_is_bitwise_unchanged(steps: int) -> None:
    """Keep the old cosine construction, including shipped 150/1000-step presets."""
    d = GaussianDiffusion(timesteps=steps, beta_schedule=BetaSchedule.COSINE)
    t = torch.linspace(0, 1, steps + 1)
    product = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
    product = product / product[0]
    betas = torch.clip(1 - product[1:] / product[:-1], 0, 0.999)
    assert torch.equal(d.betas, betas)
    assert torch.equal(d.alphas_cumprod, torch.cumprod(1.0 - betas, dim=0))


@pytest.mark.parametrize("steps", [1, 2, 4, 20])
@pytest.mark.parametrize("use_ddim", [False, True], ids=["ddpm", "ddim"])
@pytest.mark.parametrize(
    "autoregressive", [False, True], ids=["parallel", "autoregressive"]
)
def test_short_linear_processor_training_and_sampling(
    steps: int, *, use_ddim: bool, autoregressive: bool
) -> None:
    """Exercise real UNet gradients and both sampler paths on short linear chains."""
    processor = DiffusionProcessor(
        data_space=DataSpace(name="combined", channels=2, shape=(16, 16)),
        data_space_target=DataSpace(name="target", channels=1, shape=(16, 16)),
        target_channel_offset=1,
        n_history_steps=1,
        n_forecast_steps=2,
        timesteps=steps,
        ddim_steps=max(1, steps // 2) if use_ddim else None,
        eta=0.0 if use_ddim else 1.0,
        beta_schedule="linear",
        start_out_channels=4,
        dropout_rate=0.0,
        use_autoregressive=autoregressive,
        loss=torch.nn.MSELoss(),
    )
    generator = torch.Generator().manual_seed(591)
    inputs = torch.randn(2, 1, 2, 16, 16, generator=generator)
    targets = torch.randn(2, 2, 1, 16, 16, generator=generator)
    result = processor.rollout(inputs, targets)
    assert result.loss is not None
    assert torch.isfinite(result.loss)
    result.loss.backward()
    gradients = [p.grad for p in processor.model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert any(grad.abs().sum() > 0 for grad in gradients)
    processor.eval()
    with torch.inference_mode():
        output = processor.rollout(inputs).prediction
    assert output.shape == (2, 2, 2, 16, 16)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :, 0], inputs[:, -1:, 0].expand(-1, 2, -1, -1))
    assert processor._ddim_timesteps[0] == steps - 1
