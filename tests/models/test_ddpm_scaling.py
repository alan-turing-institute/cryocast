from typing import TypedDict, Unpack

import pytest
import torch

from cryocast.models.processors import DiffusionProcessor
from cryocast.types import DataSpace


class _ScalingOptions(TypedDict, total=False):
    x0_scale: float


def _make_processor(**kwargs: Unpack[_ScalingOptions]) -> DiffusionProcessor:
    combined = DataSpace(name="combined", channels=4, shape=(8, 8))
    target = DataSpace(name="target", channels=2, shape=(8, 8))
    return DiffusionProcessor(
        data_space=combined,
        data_space_target=target,
        n_forecast_steps=1,
        n_history_steps=1,
        timesteps=2,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0.0,
        use_autoregressive=True,
        target_channel_offset=1,
        loss=torch.nn.MSELoss(),
        **kwargs,
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
