"""Preserve shipped cosine configurations while linear scheduling remains opt-in."""

from collections.abc import Callable

import pytest
import torch
from omegaconf import DictConfig

from cryocast.models.diffusion import GaussianDiffusion
from cryocast.types import BetaSchedule


@pytest.mark.parametrize(
    ("model", "timesteps", "ddim_steps", "eta"),
    [("cnn_ddpm_cnn", 150, None, 1.0), ("cnn_ddim_cnn", 1000, 50, 0.0)],
)
def test_shipped_diffusion_presets_remain_cosine(
    compose_config: Callable[..., DictConfig],
    model: str,
    timesteps: int,
    ddim_steps: int | None,
    eta: float,
) -> None:
    """Keep training schedule and inference-only sampling settings separate."""
    cfg = compose_config("sample", overrides=[f"model={model}"])
    processor = cfg.model.processor
    assert processor.beta_schedule == "cosine"
    assert processor.timesteps == timesteps
    assert processor.ddim_steps == ddim_steps
    assert processor.eta == eta
    schedule = GaussianDiffusion(timesteps=timesteps)
    expected = GaussianDiffusion(timesteps=timesteps, beta_schedule=BetaSchedule.COSINE)
    assert torch.equal(schedule.betas, expected.betas)


def test_standalone_ddpm_keeps_default_schedule(
    compose_config: Callable[..., DictConfig],
) -> None:
    """The standalone 150-step preset still uses the Gaussian cosine default."""
    cfg = compose_config("sample", overrides=["model=ddpm"])
    assert cfg.model.timesteps == 150
    assert "beta_schedule" not in cfg.model
    default = GaussianDiffusion(timesteps=cfg.model.timesteps)
    cosine = GaussianDiffusion(timesteps=150, beta_schedule=BetaSchedule.COSINE)
    assert torch.equal(default.betas, cosine.betas)


def test_linear_schedule_requires_explicit_override(
    compose_config: Callable[..., DictConfig],
) -> None:
    """Selecting linear training leaves the default DDPM sampling controls alone."""
    cfg = compose_config(
        "sample",
        overrides=["model=cnn_ddpm_cnn", "model.processor.beta_schedule=linear"],
    )
    assert cfg.model.processor.beta_schedule == "linear"
    assert cfg.model.processor.timesteps == 150
    assert cfg.model.processor.ddim_steps is None
    assert cfg.model.processor.eta == 1.0
    diffusion = GaussianDiffusion(
        timesteps=cfg.model.processor.timesteps,
        beta_schedule=BetaSchedule(cfg.model.processor.beta_schedule),
    )
    assert diffusion.alphas_cumprod[-1] < 3e-5
