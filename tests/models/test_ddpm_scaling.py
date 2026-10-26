from typing import TypedDict, Unpack

import pytest
import torch

from cryocast.models.processors import DiffusionProcessor
from cryocast.types import DataSpace


class _ScalingOptions(TypedDict, total=False):
    x0_scale: float
    n_forecast_steps: int
    n_history_steps: int
    use_autoregressive: bool
    ddim_steps: int | None
    eta: float


def _make_processor(
    *, spatial_size: int = 8, **kwargs: Unpack[_ScalingOptions]
) -> DiffusionProcessor:
    combined = DataSpace(
        name="combined", channels=4, shape=(spatial_size, spatial_size)
    )
    target = DataSpace(name="target", channels=2, shape=(spatial_size, spatial_size))
    options: _ScalingOptions = {
        "n_forecast_steps": 1,
        "n_history_steps": 1,
        "use_autoregressive": True,
    }
    options.update(kwargs)
    return DiffusionProcessor(
        data_space=combined,
        data_space_target=target,
        timesteps=2,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0.0,
        target_channel_offset=1,
        loss=torch.nn.MSELoss(),
        **options,
    )


def test_ddpm_rejects_non_positive_x0_scale() -> None:
    """Reject non-positive clean-target scaling factors."""
    with pytest.raises(ValueError, match="x0_scale must be positive"):
        _make_processor(x0_scale=0.0)


def test_ddpm_defaults_to_unscaled_x0() -> None:
    """Keep existing DDPM behaviour when x0_scale is omitted."""
    processor = _make_processor()
    assert processor.x0_scale == 1.0


def test_ddpm_scales_training_target_and_restores_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scale x0 before diffusion and restore metric predictions afterward."""
    processor = _make_processor(x0_scale=2.0)
    x = torch.zeros(1, 1, 4, 8, 8)
    y = torch.full((1, 1, 2, 8, 8), 4.0)
    captured: dict[str, torch.Tensor] = {}

    def fake_q_sample(
        x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        del t, noise
        captured["x0"] = x_start.clone()
        return torch.zeros_like(x_start)

    def fake_calculate_v(
        x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        del noise, t
        return x_start

    monkeypatch.setattr(processor.diffusion, "q_sample", fake_q_sample)
    monkeypatch.setattr(processor.diffusion, "calculate_v", fake_calculate_v)

    def fake_forward(
        noisy_y: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        del t, cond
        return torch.ones_like(noisy_y)

    monkeypatch.setattr(processor.model, "forward", fake_forward)

    output = processor._rollout_training(x, y)

    torch.testing.assert_close(captured["x0"], y[:, 0] / 2.0)
    target_prediction = output.prediction[
        ..., processor.target_slice_start : processor.target_slice_end, :, :
    ]
    torch.testing.assert_close(
        target_prediction, torch.full_like(target_prediction, 2.0)
    )


def test_ddpm_restores_scale_after_reverse_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the original latent scale after reverse diffusion."""
    processor = _make_processor(x0_scale=2.5)
    noisy = torch.randn(1, 2, 8, 8)
    cond = torch.randn(1, 4, 8, 8)

    def fake_forward(
        y: torch.Tensor, t: torch.Tensor, cond_input: torch.Tensor
    ) -> torch.Tensor:
        del t, cond_input
        return torch.zeros_like(y)

    def fake_p_sample(
        y: torch.Tensor, t: torch.Tensor, pred_v: torch.Tensor
    ) -> torch.Tensor:
        del t, pred_v
        return torch.ones_like(y)

    monkeypatch.setattr(processor.model, "forward", fake_forward)
    monkeypatch.setattr(processor.diffusion, "p_sample", fake_p_sample)

    output = processor._run_reverse_diffusion(noisy, cond)

    torch.testing.assert_close(output, torch.full_like(noisy, 2.5))


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_rejects_non_finite_or_negative_scale(scale: float) -> None:
    """Reject values that cannot define a positive finite target scale."""
    with pytest.raises(ValueError, match="x0_scale must be positive"):
        _make_processor(x0_scale=scale)


@pytest.mark.parametrize("autoregressive", [True, False], ids=["ar", "parallel"])
@pytest.mark.parametrize("scale", [0.25, 2.5])
def test_training_scales_only_targets_and_restores_all_metric_leads(
    monkeypatch: pytest.MonkeyPatch, *, autoregressive: bool, scale: float
) -> None:
    """Use scaled x0 for noising and velocity loss, preserving history channels."""
    processor = _make_processor(
        x0_scale=scale,
        use_autoregressive=autoregressive,
        n_forecast_steps=3,
        n_history_steps=2,
    )
    x = torch.linspace(-3, 7, 2 * 2 * 4 * 8 * 8).reshape(2, 2, 4, 8, 8)
    y = torch.linspace(-2, 5, 2 * 3 * 2 * 8 * 8).reshape(2, 3, 2, 8, 8)
    original_x, original_y = x.clone(), y.clone()
    captured: dict[str, torch.Tensor] = {}
    q_sample = processor.diffusion.q_sample

    def spy_q_sample(
        x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        captured.update(clean=x_start.clone(), t=t.clone(), noise=noise.clone())
        noisy = q_sample(x_start, t, noise)
        captured["noisy"] = noisy.clone()
        return noisy

    def fixed_velocity(
        noisy_y: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        del t
        captured["cond"] = cond.clone()
        return torch.full_like(noisy_y, 0.125)

    monkeypatch.setattr(processor.diffusion, "q_sample", spy_q_sample)
    monkeypatch.setattr(processor.model, "forward", fixed_velocity)
    output = processor.rollout(x, y)
    expected_clean = y[:, 0] / scale if autoregressive else (y / scale).flatten(1, 2)
    torch.testing.assert_close(captured["clean"], expected_clean)
    torch.testing.assert_close(captured["cond"], x.flatten(1, 2), rtol=0, atol=0)
    pred_v = torch.full_like(expected_clean, 0.125)
    expected_v = processor.diffusion.calculate_v(
        expected_clean, captured["noise"], captured["t"]
    )
    expected_loss = torch.nn.functional.mse_loss(pred_v, expected_v)
    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected_loss)
    reconstructed = (
        processor.diffusion.calculate_v(pred_v, captured["noisy"], captured["t"])
        * scale
    )
    expected_target = (
        reconstructed[:, None].expand(-1, 3, -1, -1, -1)
        if autoregressive
        else reconstructed.reshape(2, 3, 2, 8, 8)
    )
    expected = x[:, -1:, :, :, :].expand(-1, 3, -1, -1, -1).clone()
    expected[:, :, 1:3] = expected_target
    torch.testing.assert_close(output.prediction, expected)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(y, original_y, rtol=0, atol=0)
    assert not output.prediction.requires_grad


@pytest.mark.parametrize(("ddim_steps", "eta"), [(None, 1.0), (1, 0.0), (1, 0.5)])
@pytest.mark.parametrize("scale", [0.25, 2.5])
def test_reverse_samplers_restore_original_units_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    ddim_steps: int | None,
    eta: float,
    scale: float,
) -> None:
    """Exercise real DDPM/DDIM equations with paired inputs and random noise."""
    processor = _make_processor(x0_scale=scale, ddim_steps=ddim_steps, eta=eta)
    unit = _make_processor(ddim_steps=ddim_steps, eta=eta)
    unit.load_state_dict(processor.state_dict())
    noisy = torch.linspace(-1, 1, 128).reshape(1, 2, 8, 8)
    cond = torch.linspace(-2, 2, 256).reshape(1, 4, 8, 8)
    captured: list[torch.Tensor] = []

    def fixed_velocity(
        y: torch.Tensor, t: torch.Tensor, cond_input: torch.Tensor
    ) -> torch.Tensor:
        del t
        captured.append(cond_input.clone())
        return torch.full_like(y, 0.125)

    monkeypatch.setattr(processor.model, "forward", fixed_velocity)
    monkeypatch.setattr(unit.model, "forward", fixed_velocity)
    torch.manual_seed(478)
    expected = unit._run_reverse_diffusion(noisy.clone(), cond.clone()) * scale
    torch.manual_seed(478)
    actual = processor._run_reverse_diffusion(noisy.clone(), cond.clone())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(captured) == 2 * processor.ddim_steps
    for observed_cond in captured:
        torch.testing.assert_close(observed_cond, cond, rtol=0, atol=0)


@pytest.mark.parametrize("autoregressive", [True, False], ids=["ar", "parallel"])
@pytest.mark.parametrize(("ddim_steps", "eta"), [(None, 1.0), (1, 0.0)])
@pytest.mark.parametrize("scale", [0.25, 2.5])
def test_sampling_restores_target_scale_before_updating_history(
    monkeypatch: pytest.MonkeyPatch,
    *,
    autoregressive: bool,
    ddim_steps: int | None,
    eta: float,
    scale: float,
) -> None:
    """Preserve non-target channels and feed restored AR predictions into history."""
    processor = _make_processor(
        x0_scale=scale,
        use_autoregressive=autoregressive,
        n_forecast_steps=3,
        n_history_steps=2,
        ddim_steps=ddim_steps,
        eta=eta,
    )
    x = torch.arange(512, dtype=torch.float32).reshape(1, 2, 4, 8, 8)
    original = x.clone()
    conditions: list[torch.Tensor] = []
    raw_outputs: list[torch.Tensor] = []

    def sample_normalized(y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        conditions.append(cond.clone())
        raw = torch.arange(y.numel(), dtype=y.dtype, device=y.device).reshape_as(y)
        raw = raw / 16 + len(conditions)
        raw_outputs.append(raw)
        return raw

    method = (
        "_run_ddpm_reverse_diffusion"
        if processor.uses_ddpm_sampler
        else "_run_ddim_reverse_diffusion"
    )
    monkeypatch.setattr(processor, method, sample_normalized)
    result = processor.rollout(x)
    expected = x[:, -1:].expand(-1, 3, -1, -1, -1).clone()
    if autoregressive:
        assert len(conditions) == 3
        window = x.clone()
        for lead, (condition, raw) in enumerate(
            zip(conditions, raw_outputs, strict=True)
        ):
            torch.testing.assert_close(condition, window.flatten(1, 2), rtol=0, atol=0)
            expected[:, lead, 1:3] = raw * scale
            next_frame = window[:, -1:].clone()
            next_frame[:, 0, 1:3] = raw * scale
            window = torch.cat((window[:, 1:], next_frame), dim=1)
    else:
        assert len(conditions) == 1
        torch.testing.assert_close(conditions[0], x.flatten(1, 2), rtol=0, atol=0)
        expected[:, :, 1:3] = (raw_outputs[0] * scale).reshape(1, 3, 2, 8, 8)
    torch.testing.assert_close(result.prediction, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert result.loss is None


@pytest.mark.parametrize("autoregressive", [True, False], ids=["ar", "parallel"])
@pytest.mark.parametrize(("ddim_steps", "eta"), [(None, 1.0), (1, 0.0)])
def test_omitted_scale_matches_explicit_one_in_training_and_sampling(
    *,
    autoregressive: bool,
    ddim_steps: int | None,
    eta: float,
) -> None:
    """Omitting x0_scale preserves seeded outputs, losses and RNG consumption."""
    options: _ScalingOptions = {
        "use_autoregressive": autoregressive,
        "n_forecast_steps": 3,
        "n_history_steps": 2,
        "ddim_steps": ddim_steps,
        "eta": eta,
    }
    default = _make_processor(spatial_size=16, **options).eval()
    explicit_options: _ScalingOptions = {**options, "x0_scale": 1.0}
    explicit = _make_processor(spatial_size=16, **explicit_options).eval()
    explicit.load_state_dict(default.state_dict(), strict=True)
    x = torch.linspace(-1, 1, 2 * 4 * 16 * 16).reshape(1, 2, 4, 16, 16)
    y = torch.linspace(-2, 2, 3 * 2 * 16 * 16).reshape(1, 3, 2, 16, 16)
    with torch.no_grad():
        for target in (y, None):
            torch.manual_seed(478)
            expected = default.rollout(x, target)
            expected_rng = torch.random.get_rng_state()
            torch.manual_seed(478)
            actual = explicit.rollout(x, target)
            torch.testing.assert_close(
                actual.prediction, expected.prediction, rtol=0, atol=0
            )
            torch.testing.assert_close(
                torch.random.get_rng_state(), expected_rng, rtol=0, atol=0
            )
            if expected.loss is not None:
                assert actual.loss is not None
                torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
            else:
                assert actual.loss is None


@pytest.mark.parametrize("autoregressive", [True, False], ids=["ar", "parallel"])
@pytest.mark.parametrize("scale", [0.411207, 2.5])
def test_scaled_real_training_keeps_finite_gradients(
    *,
    autoregressive: bool,
    scale: float,
) -> None:
    """Backpropagate through actual noising, denoiser and velocity loss."""
    processor = _make_processor(
        spatial_size=16,
        x0_scale=scale,
        use_autoregressive=autoregressive,
        n_forecast_steps=3,
        n_history_steps=2,
    )
    x = torch.linspace(-1, 1, 2 * 4 * 16 * 16).reshape(1, 2, 4, 16, 16).requires_grad_()
    y = torch.linspace(-2, 2, 3 * 2 * 16 * 16).reshape(1, 3, 2, 16, 16).requires_grad_()
    torch.manual_seed(478)
    output = processor.rollout(x, y)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert torch.isfinite(output.prediction).all()
    for tensor in (x, y):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    gradients = [p.grad for p in processor.model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)
