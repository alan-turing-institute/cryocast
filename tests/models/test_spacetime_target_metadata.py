"""Space-time prediction slices survive full-latent training interfaces."""

from typing import Any

import pytest
import torch

from cryocast.models.processors import SpaceTimeVitProcessor
from cryocast.types import DataSpace


def _processor(
    *, explicit: bool, offset: int, **overrides: Any
) -> SpaceTimeVitProcessor:
    """Construct either the legacy or explicit prediction-target interface."""
    combined = DataSpace(name="combined", channels=5, shape=(8, 8))
    target = DataSpace(name="sic", channels=2, shape=(8, 8))
    options: dict[str, Any] = {
        "data_space": combined,
        "data_space_target": combined if explicit else target,
        "target_channel_offset": offset,
        "n_history_steps": 3,
        "n_forecast_steps": 2,
        "dropout": 0.0,
        "emb_dim": 8,
        "heads": 2,
        "mlp_dim": 16,
        "patch_size": 4,
        "spatial_depth": 1,
        "temporal_depth": 1,
        "forecast_spatial_depth": 1,
    }
    if explicit:
        options["target_latent_space"] = target
    options.update(overrides)
    return SpaceTimeVitProcessor(**options)


@pytest.mark.parametrize("offset", [0, 1, 3])
@pytest.mark.parametrize("seed", [0, 516])
def test_explicit_target_metadata_preserves_legacy_network_and_forecasts(
    offset: int, seed: int
) -> None:
    """Metadata narrows only the prediction slice and does not change model weights."""
    torch.manual_seed(seed)
    legacy = _processor(explicit=False, offset=offset)
    torch.manual_seed(seed)
    explicit = _processor(explicit=True, offset=offset)
    assert explicit.c_target == 2
    assert explicit.c_combined == 5
    assert explicit.target_slice_start == offset
    assert explicit.target_slice_end == offset + 2
    assert not explicit.computes_loss_in_latent_space
    torch.testing.assert_close(
        explicit.state_dict(), legacy.state_dict(), rtol=0, atol=0
    )
    history = torch.randn(2, 3, 5, 8, 8)
    states, gradients, forecasts = [], [], []
    for processor in (legacy, explicit):
        processor.train()
        optimizer = torch.optim.Adam(processor.parameters(), lr=1e-3)
        x = history.clone().requires_grad_()
        forecast = processor.rollout(x).prediction
        forecast.square().mean().backward()
        gradients.append(
            {
                k: None if p.grad is None else p.grad.clone()
                for k, p in processor.named_parameters()
            }
        )
        assert x.grad is not None
        gradients[-1]["history"] = x.grad.clone()
        optimizer.step()
        states.append(processor.state_dict())
        processor.eval()
        with torch.no_grad():
            forecasts.append(processor.rollout(history).prediction)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    torch.testing.assert_close(forecasts[0], forecasts[1], rtol=0, atol=0)
    persistence = history[:, -1:].expand(-1, 2, -1, -1, -1)
    torch.testing.assert_close(
        forecasts[1][:, :, :offset], persistence[:, :, :offset], rtol=0, atol=0
    )
    torch.testing.assert_close(
        forecasts[1][:, :, offset + 2 :],
        persistence[:, :, offset + 2 :],
        rtol=0,
        atol=0,
    )


def test_explicit_prediction_target_requires_matching_latent_grid() -> None:
    """A physically shaped output must not be mistaken for an encoded target slice."""
    with pytest.raises(ValueError, match="shape"):
        _processor(
            explicit=True,
            offset=1,
            target_latent_space=DataSpace(name="sic", channels=2, shape=(4, 4)),
        )


@pytest.mark.parametrize("offset", [0, 1, 3])
def test_new_metadata_preserves_target_independence_and_horizon_prefix(
    offset: int,
) -> None:
    """Future labels never reach inference and longer horizons retain the same prefix."""
    short = _processor(explicit=True, offset=offset).eval()
    long = _processor(explicit=True, offset=offset, n_forecast_steps=5).eval()
    long.load_state_dict(short.state_dict(), strict=True)
    x = torch.randn(2, 3, 5, 8, 8)
    with torch.no_grad():
        expected = short.rollout(x).prediction
        future_a = short.rollout(x, torch.zeros(2, 2, 5, 8, 8)).prediction
        future_b = short.rollout(
            x, torch.full((2, 2, 5, 8, 8), float("nan"))
        ).prediction
        prefix = long.rollout(x).prediction[:, :2]
    torch.testing.assert_close(expected, future_a, rtol=0, atol=0)
    torch.testing.assert_close(expected, future_b, rtol=0, atol=0)
    torch.testing.assert_close(expected, prefix, rtol=1e-5, atol=1e-6)
