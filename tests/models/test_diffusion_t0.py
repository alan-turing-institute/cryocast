"""Forward, velocity and reverse consistency at the first stored noise level."""

import pytest
import torch
from omegaconf import DictConfig

from cryocast.models import DDPM
from cryocast.models.diffusion import GaussianDiffusion
from cryocast.models.processors import DiffusionProcessor
from cryocast.types import BetaSchedule, DataSpace


@pytest.mark.parametrize("schedule", list(BetaSchedule))
@pytest.mark.parametrize("steps", [1, 2, 4, 150, 1000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_t0_forward_velocity_rotation_and_inverse(
    schedule: BetaSchedule, steps: int, dtype: torch.dtype
) -> None:
    """Use the same scheduled rotation at zero as at every other timestep."""
    diffusion = GaussianDiffusion(timesteps=steps, beta_schedule=schedule)
    generator = torch.Generator().manual_seed(585)
    clean = torch.randn(2, 3, 4, 4, generator=generator, dtype=dtype)
    noise = torch.randn(clean.shape, generator=generator, dtype=dtype)
    t = torch.zeros(2, dtype=torch.long)
    a = diffusion.sqrt_alphas_cumprod[0]
    b = diffusion.sqrt_one_minus_alphas_cumprod[0]

    noisy = diffusion.q_sample(clean, t, noise)
    velocity = diffusion.calculate_v(clean, noise, t)

    torch.testing.assert_close(noisy, a * clean + b * noise, rtol=0, atol=0)
    torch.testing.assert_close(velocity, a * noise - b * clean, rtol=0, atol=0)
    torch.testing.assert_close(a * noisy - b * velocity, clean, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(b * noisy + a * velocity, noise, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        noisy.square() + velocity.square(),
        clean.square() + noise.square(),
        rtol=2e-6,
        atol=2e-6,
    )
    assert not torch.equal(noisy, clean)


@pytest.mark.parametrize("schedule", list(BetaSchedule))
@pytest.mark.parametrize("shape", [(3, 5), (3, 2, 4, 4), (3, 2, 1, 4, 4)])
def test_zero_and_nonzero_timesteps_share_broadcasting(
    schedule: BetaSchedule, shape: tuple[int, ...]
) -> None:
    """Mixed batches use per-example coefficients without crossing sample axes."""
    diffusion = GaussianDiffusion(timesteps=4, beta_schedule=schedule)
    generator = torch.Generator().manual_seed(585)
    clean = torch.randn(shape, generator=generator)
    noise = torch.randn(shape, generator=generator)
    t = torch.tensor([0, 1, 3])
    noisy = diffusion.q_sample(clean, t, noise)
    for i in range(3):
        a = diffusion.sqrt_alphas_cumprod[t[i]]
        b = diffusion.sqrt_one_minus_alphas_cumprod[t[i]]
        torch.testing.assert_close(
            noisy[i], a * clean[i] + b * noise[i], rtol=0, atol=0
        )


@pytest.mark.parametrize("schedule", list(BetaSchedule))
@pytest.mark.parametrize("steps", [1, 4, 150])
def test_t0_gradients_match_the_scheduled_rotation(
    schedule: BetaSchedule, steps: int
) -> None:
    """The first forward step retains gradients through both signal and noise."""
    diffusion = GaussianDiffusion(timesteps=steps, beta_schedule=schedule)
    clean = torch.ones(2, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    noise = torch.full_like(clean, 0.5, requires_grad=True)
    t = torch.zeros(2, dtype=torch.long)
    noisy = diffusion.q_sample(clean, t, noise)
    velocity = diffusion.calculate_v(clean, noise, t)
    q_clean, q_noise = torch.autograd.grad(noisy.sum(), (clean, noise))
    v_clean, v_noise = torch.autograd.grad(velocity.sum(), (clean, noise))
    a, b = diffusion.sqrt_alphas_cumprod[0], diffusion.sqrt_one_minus_alphas_cumprod[0]
    for actual, coefficient in (
        (q_clean, a),
        (q_noise, b),
        (v_clean, -b),
        (v_noise, a),
    ):
        torch.testing.assert_close(
            actual, torch.full_like(actual, float(coefficient)), rtol=0, atol=0
        )


@pytest.mark.parametrize("schedule", list(BetaSchedule))
def test_omitted_noise_is_sampled_at_zero(schedule: BetaSchedule) -> None:
    """Implicit noise and explicit noise have identical values and RNG consumption."""
    diffusion = GaussianDiffusion(timesteps=4, beta_schedule=schedule)
    clean = torch.ones(2, 1, 4, 4)
    t = torch.tensor([0, 3])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(585)
        noise = torch.randn_like(clean)
        expected = diffusion.q_sample(clean, t, noise)
        state = torch.random.get_rng_state()
        torch.manual_seed(585)
        actual = diffusion.q_sample(clean, t)
        assert torch.equal(torch.random.get_rng_state(), state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual[0], clean[0])


@pytest.mark.parametrize("schedule", list(BetaSchedule))
@pytest.mark.parametrize("steps", [1, 2, 4, 150, 1000])
def test_t0_exact_velocity_reverses_to_clean_without_fresh_noise(
    schedule: BetaSchedule, steps: int
) -> None:
    """The final reverse step recovers the clean field with scheduled precision."""
    diffusion = GaussianDiffusion(timesteps=steps, beta_schedule=schedule)
    clean = torch.tensor([[[[0.25, -0.5], [0.75, -1.0]]]])
    noise = torch.tensor([[[[1.0, 0.5], [-0.5, -1.0]]]])
    t = torch.zeros(1, dtype=torch.long)
    noisy = diffusion.q_sample(clean, t, noise)
    velocity = diffusion.calculate_v(clean, noise, t)
    first = diffusion.p_sample(noisy, t, velocity)
    second = diffusion.p_sample(noisy, t, velocity)
    assert torch.equal(first, second)
    torch.testing.assert_close(
        first, diffusion.posterior_mean_coef1[0] * clean, rtol=2e-6, atol=2e-6
    )
    # Existing float32 posterior coefficients can round slightly away from one.
    torch.testing.assert_close(first, clean, rtol=5e-4, atol=1e-6)


def _observe_zero_corruption(
    diffusion: GaussianDiffusion, monkeypatch: pytest.MonkeyPatch
) -> list[torch.Tensor]:
    """Record real forward corruption while forcing the sampled training index to zero."""
    calls: list[torch.Tensor] = []
    original = diffusion.q_sample

    def zero_steps(
        low: int, high: int, size: tuple[int, ...], *, device: torch.device
    ) -> torch.Tensor:
        assert low == 0
        assert high == diffusion.timesteps
        return torch.zeros(size, dtype=torch.long, device=device)

    def check_forward(
        x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert noise is not None
        assert torch.equal(t, torch.zeros_like(t))
        result = original(x_start, t, noise)
        expected = (
            diffusion.sqrt_alphas_cumprod[0] * x_start
            + diffusion.sqrt_one_minus_alphas_cumprod[0] * noise
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert not torch.equal(result, x_start)
        calls.append(result.detach())
        return result

    monkeypatch.setattr(torch, "randint", zero_steps)
    monkeypatch.setattr(diffusion, "q_sample", check_forward)
    return calls


@pytest.mark.parametrize("schedule", list(BetaSchedule))
@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize(
    "autoregressive", [False, True], ids=["parallel", "autoregressive"]
)
def test_processor_trains_at_t0_and_samples_both_samplers(
    schedule: BetaSchedule,
    steps: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    autoregressive: bool,
) -> None:
    """Exercise corrected training, real UNet gradients and DDPM/DDIM inference."""
    processor = DiffusionProcessor(
        data_space=DataSpace(name="combined", channels=2, shape=(16, 16)),
        data_space_target=DataSpace(name="target", channels=1, shape=(16, 16)),
        target_channel_offset=1,
        n_history_steps=1,
        n_forecast_steps=2,
        timesteps=steps,
        beta_schedule=schedule.value,
        start_out_channels=4,
        dropout_rate=0.0,
        use_autoregressive=autoregressive,
        loss=torch.nn.MSELoss(),
    )
    calls = _observe_zero_corruption(processor.diffusion, monkeypatch)
    inputs = torch.randn(2, 1, 2, 16, 16)
    targets = torch.randn(2, 2, 1, 16, 16)
    trained = processor.rollout(inputs, targets)
    assert len(calls) == 1
    assert trained.loss is not None
    assert torch.isfinite(trained.loss)
    trained.loss.backward()
    grads = [p.grad for p in processor.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    processor.eval()
    for ddim_steps, eta in ((None, 1.0), (1, 0.0)):
        processor.set_sampler(ddim_steps=ddim_steps, eta=eta)
        with torch.no_grad():
            output = processor.rollout(inputs).prediction
        assert output.shape == (2, 2, 2, 16, 16)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            output[:, :, 0], inputs[:, -1:, 0].expand(-1, 2, -1, -1)
        )


@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize(
    "autoregressive", [False, True], ids=["parallel", "autoregressive"]
)
def test_standalone_ddpm_training_uses_corrected_t0(
    steps: int, monkeypatch: pytest.MonkeyPatch, *, autoregressive: bool
) -> None:
    """The standalone DDPM reaches the same real forward process at timestep zero."""
    model = DDPM(
        name="t0-test",
        hemisphere="north",
        input_spaces=[
            {"channels": 1, "name": "sic", "shape": (16, 16)},
            {"channels": 1, "name": "era5", "shape": (8, 8)},
        ],
        output_space={"channels": 1, "name": "sic", "shape": (16, 16)},
        n_history_steps=1,
        n_forecast_steps=2,
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        metrics=[],
        optimizer={},
        scheduler={},
        lr_scheduler={},
        start_out_channels=4,
        dropout_rate=0.0,
        timesteps=steps,
        use_autoregressive=autoregressive,
    )
    calls = _observe_zero_corruption(model.diffusion, monkeypatch)
    result = model.training_step(
        {
            "sic": torch.randn(2, 1, 1, 16, 16),
            "era5": torch.randn(2, 1, 1, 8, 8),
            "target": torch.randn(2, 2, 1, 16, 16),
        },
        0,
    )
    assert len(calls) == 1
    assert torch.isfinite(result.loss)
    result.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
