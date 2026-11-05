"""Missing-date semantics, causal forecasts and checkpoint regression tests."""

from pathlib import Path
from typing import Any

import pytest
import torch

from cryocast.models.encoders import CNNEncoder, MissingDataCNNEncoder
from cryocast.models.processors import SpaceTimeVitProcessor
from cryocast.types import DataSpace


def _encoder(**kwargs: Any) -> MissingDataCNNEncoder:
    """Use a compact missing-aware two-channel encoder."""
    return MissingDataCNNEncoder(
        data_space_in=DataSpace(name="float-argo", channels=2, shape=(8, 8)),
        latent_space=(4, 4),
        n_layers=1,
        **kwargs,
    )


def _processor(**overrides: Any) -> SpaceTimeVitProcessor:
    """Construct a small transformer with explicit conditioning/target slices."""
    options: dict[str, Any] = {
        "data_space": DataSpace(name="combined", channels=5, shape=(8, 8)),
        "data_space_target": DataSpace(name="target", channels=2, shape=(8, 8)),
        "target_channel_offset": 1,
        "n_forecast_steps": 3,
        "n_history_steps": 3,
        "emb_dim": 8,
        "heads": 2,
        "mlp_dim": 16,
        "patch_size": 4,
        "spatial_depth": 1,
        "temporal_depth": 1,
        "forecast_spatial_depth": 1,
        "dropout": 0.0,
    }
    options.update(overrides)
    return SpaceTimeVitProcessor(**options)


def test_missing_date_sentinel_does_not_remove_valid_individual_values() -> None:
    """A missing date is a full sentinel frame, not an isolated physical value."""
    encoder = _encoder(missing_fill_value=-1.0).eval()
    inputs = torch.ones(2, 2, 8, 8)
    inputs[0, 0, 0, 0] = -1.0
    inputs[1] = -1.0
    captured: list[torch.Tensor] = []
    handle = encoder.input_adapter.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    encoder(inputs)
    handle.remove()
    assert captured[0][0, 0, 0, 0] == -1.0
    assert torch.equal(captured[0][0, -1], torch.ones(8, 8))
    assert torch.count_nonzero(captured[0][1]) == 0


@pytest.mark.parametrize("seed", [0, 516])
def test_identity_adapter_does_not_change_downstream_random_initialisation(
    seed: int,
) -> None:
    """With complete input, adding the neutral adapter preserves matched RNG state."""
    torch.manual_seed(seed)
    standard = CNNEncoder(
        data_space_in=DataSpace(name="float-argo", channels=2, shape=(8, 8)),
        latent_space=(4, 4),
        n_layers=1,
    ).eval()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    missing = _encoder().eval()
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for name, value in standard.state_dict().items():
        torch.testing.assert_close(missing.state_dict()[name], value, atol=0, rtol=0)
    inputs = torch.randn(2, 2, 8, 8)
    torch.testing.assert_close(missing(inputs), standard(inputs), atol=0, rtol=0)


@pytest.mark.parametrize("fill", [float("nan"), float("inf"), -float("inf"), True])
def test_missing_sentinel_must_be_finite_or_none(fill: object) -> None:
    """Nonfinite values already have explicit missingness semantics."""
    with pytest.raises(ValueError, match="missing_fill_value"):
        _encoder(missing_fill_value=fill)


@pytest.mark.parametrize("probability", [True, float("nan"), float("inf")])
def test_conditioning_dropout_is_finite_real_probability(probability: object) -> None:
    """Do not silently interpret a Boolean switch as a stochastic probability."""
    with pytest.raises(ValueError, match="conditioning_dropout_probability"):
        _encoder(conditioning_dropout_probability=probability)


@pytest.mark.parametrize(
    "field",
    [
        "emb_dim",
        "heads",
        "mlp_dim",
        "patch_size",
        "spatial_depth",
        "temporal_depth",
        "forecast_spatial_depth",
        "n_history_steps",
        "n_forecast_steps",
        "target_channel_offset",
    ],
)
def test_transformer_counts_reject_boolean_values(field: str) -> None:
    """Reject Boolean dimensions before constructing partially valid networks."""
    with pytest.raises(ValueError, match=field):
        _processor(**{field: True})


@pytest.mark.parametrize(
    "field",
    [
        "emb_dim",
        "heads",
        "mlp_dim",
        "patch_size",
        "spatial_depth",
        "temporal_depth",
        "forecast_spatial_depth",
        "target_channel_offset",
    ],
)
def test_transformer_counts_reject_fractional_values(field: str) -> None:
    """Report configuration errors rather than low-level reshape/range failures."""
    with pytest.raises(ValueError, match=field):
        _processor(**{field: 1.5})


@pytest.mark.parametrize("bad_dropout", [float("nan"), True])
def test_transformer_dropout_rejects_invalid_real_values(bad_dropout: object) -> None:
    """An invalid probability must not create an unusable attention stack."""
    with pytest.raises(ValueError, match="dropout"):
        _processor(dropout=bad_dropout)


def test_nonfinite_missing_inputs_have_zero_gradients() -> None:
    """Missing observations cannot poison gradients or receive invented gradients."""
    encoder = _encoder().eval()
    values = torch.ones(2, 2, 8, 8)
    values[0, 0] = torch.nan
    values[1, 1] = torch.inf
    values.requires_grad_()
    result = encoder(values)
    result.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad[~torch.isfinite(values)]) == 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()
    )


@pytest.mark.parametrize("seed", [0, 516, 1234])
def test_prefix_values_and_history_gradients_are_horizon_invariant(seed: int) -> None:
    """Causality applies to gradients as well as the displayed forecast prefix."""
    torch.manual_seed(seed)
    short = _processor(n_forecast_steps=2).double().eval()
    long = _processor(n_forecast_steps=7).double().eval()
    long.load_state_dict(short.state_dict(), strict=True)
    history = torch.randn(2, 3, 5, 8, 8, dtype=torch.float64, requires_grad=True)
    other = history.detach().clone().requires_grad_()
    expected = short.rollout(history).prediction
    actual = long.rollout(other).prediction[:, :2]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    expected[:, :, 1:3].square().sum().backward()
    actual[:, :, 1:3].square().sum().backward()
    torch.testing.assert_close(history.grad, other.grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("history_steps", [1, 3, 15])
def test_untrained_conditioning_channels_are_exact_persistence(
    history_steps: int,
) -> None:
    """Changes in forecast horizon never promote Argo to prognostic channels."""
    model = _processor(n_history_steps=history_steps, n_forecast_steps=10).eval()
    inputs = torch.randn(2, history_steps, 5, 8, 8)
    prediction = model.rollout(inputs).prediction
    expected = inputs[:, -1:].expand(-1, 10, -1, -1, -1)
    torch.testing.assert_close(prediction[:, :, :1], expected[:, :, :1], atol=0, rtol=0)
    torch.testing.assert_close(prediction[:, :, 3:], expected[:, :, 3:], atol=0, rtol=0)


def test_processor_checkpoint_restores_forecasts_and_next_adam_update(
    tmp_path: Path,
) -> None:
    """Safe restoration preserves a trained state and its optimizer continuation."""
    torch.manual_seed(516)
    model = _processor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 3, 5, 8, 8)
    model.rollout(inputs).prediction.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "spacetime.ckpt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored = _processor()
    restored.load_state_dict(payload["model"], strict=True)
    resumed = torch.optim.Adam(restored.parameters(), lr=1e-3)
    resumed.load_state_dict(payload["optimizer"])
    torch.testing.assert_close(
        model.rollout(inputs).prediction,
        restored.rollout(inputs).prediction,
        atol=0,
        rtol=0,
    )
    for module, opt in ((model, optimizer), (restored, resumed)):
        opt.zero_grad(set_to_none=True)
        module.rollout(inputs).prediction.square().mean().backward()
        opt.step()
    torch.testing.assert_close(
        model.state_dict(), restored.state_dict(), atol=0, rtol=0
    )
