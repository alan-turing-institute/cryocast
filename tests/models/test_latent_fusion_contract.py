"""Numerical and analytical contracts for conditioning-only stream attention."""

import math
from pathlib import Path
from typing import Any, cast

import lightning
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import cryocast
from cryocast.models import EncodeProcessDecode
from cryocast.models.common import LatentFusion


@pytest.mark.parametrize(
    "temperature", [float("nan"), float("inf"), -float("inf"), True, False, 0.0, -1.0]
)
def test_temperature_must_be_finite_positive_real(temperature: float) -> None:
    """Reject invalid temperatures before they contaminate scores or gradients."""
    with pytest.raises(ValueError, match="temperature"):
        LatentFusion([1, 2], mode="attention", temperature=temperature)


@pytest.mark.parametrize("channels", [[True], [1.5], [0], [-1], []])
@pytest.mark.parametrize("mode", ["concat", "attention"])
def test_channel_counts_are_positive_integers(channels: list[Any], mode: str) -> None:
    """Do not truncate fractional dimensions or accept Boolean channel counts."""
    with pytest.raises(ValueError, match=r"channel|stream"):
        LatentFusion(channels, mode=mode)


def test_zero_score_initialisation_preserves_random_state() -> None:
    """Adding zero gates must not change downstream network initialisation."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(455)
        before = torch.get_rng_state().clone()
        LatentFusion([2, 3], mode="attention")
        assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_scores_do_not_overflow_low_precision(dtype: torch.dtype) -> None:
    """Large finite score heads still yield finite, representable outputs."""
    fusion = LatentFusion([1, 1], mode="attention").to(dtype=dtype)
    with torch.no_grad():
        for head in _heads(fusion):
            head.weight.fill_(60000)
    inputs = [torch.full((1, 1, 1, 2, 2), x, dtype=dtype) for x in (2.0, 1.0)]
    result = fusion(inputs)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result.float(),
        torch.cat([inputs[0].float() * 2, torch.zeros_like(inputs[1]).float()], dim=2),
    )


def test_cpu_autocast_keeps_gate_logits_finite() -> None:
    """Autocast must not reintroduce overflow inside the stable score path."""
    fusion = LatentFusion([1, 1], mode="attention")
    with torch.no_grad():
        for head in _heads(fusion):
            head.weight.fill_(60000)
    inputs = [torch.full((1, 1, 1, 2, 2), x) for x in (2.0, 1.0)]
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        result = fusion(inputs)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result, torch.cat([inputs[0] * 2, torch.zeros_like(inputs[1])], dim=2)
    )


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_two_stream_attention_and_gradients_match_closed_form(
    temperature: float,
) -> None:
    """Pin weights, input gradients and temperature-scaled gate derivatives."""
    fusion = LatentFusion([1, 1], mode="attention", temperature=temperature).double()
    with torch.no_grad():
        _heads(fusion)[1].bias.fill_(temperature * math.log(3))
    inputs = [
        torch.full((1, 1, 1, 1, 1), x, dtype=torch.float64, requires_grad=True)
        for x in (1.0, 3.0)
    ]
    result = fusion(inputs)
    torch.testing.assert_close(
        result.flatten(), torch.tensor([0.5, 4.5], dtype=torch.float64)
    )
    result.sum().backward()
    for tensor, expected in zip(inputs, (0.5, 1.5), strict=True):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, expected))
    for head, derivative, value in zip(
        _heads(fusion), (-0.75, 0.75), (1.0, 3.0), strict=True
    ):
        torch.testing.assert_close(
            head.bias.grad, torch.full_like(head.bias, derivative / temperature)
        )
        torch.testing.assert_close(
            head.weight.grad,
            torch.full_like(head.weight, derivative * value / temperature),
        )


@pytest.mark.parametrize("streams", [1, 2, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_attention_preserves_values_layout_and_dtype(
    streams: int, dtype: torch.dtype
) -> None:
    """Zero heads concatenate heterogeneous channels without reordering them."""
    channels = list(range(1, streams + 1))
    inputs = [torch.full((2, 3, n, 4, 2), float(n), dtype=dtype) for n in channels]
    fusion = LatentFusion(channels, mode="attention").to(dtype=dtype)
    torch.testing.assert_close(fusion(inputs), torch.cat(inputs, dim=2), rtol=0, atol=0)


def build_model(*, attention: bool, physical: bool = False) -> EncodeProcessDecode:
    """Construct a small two-stream model using production encoder/decoder classes."""
    return EncodeProcessDecode(
        name="fusion-contract",
        hemisphere="north",
        input_spaces=[
            {"name": "weather", "channels": 2, "shape": [4, 4]},
            {"name": "sic", "channels": 1, "shape": [4, 4]},
        ],
        output_space={"name": "sic", "channels": 1, "shape": [4, 4]},
        encoders=DictConfig(
            {
                "latent_space": [4, 4],
                "weather": {"_target_": "cryocast.models.encoders.NaiveLinearEncoder"},
                "sic": {"_target_": "cryocast.models.encoders.NaiveLinearEncoder"},
            }
        ),
        decoder=DictConfig({"_target_": "cryocast.models.decoders.NaiveLinearDecoder"}),
        processor=DictConfig({"_target_": "cryocast.models.processors.NullProcessor"}),
        fusion=DictConfig(
            {"_target_": "cryocast.models.common.LatentFusion", "mode": "attention"}
        )
        if attention
        else None,
        rollout_space="physical" if physical else "latent",
        n_history_steps=2,
        n_forecast_steps=2,
        target_variable_indices=[0],
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        metrics=[],
        optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
    )


@pytest.mark.parametrize("physical", [False, True])
def test_same_seed_zero_attention_matches_default_model(*, physical: bool) -> None:
    """A fresh zero gate preserves common weights and the complete forecast."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(455)
        default = build_model(attention=False, physical=physical).eval()
        torch.manual_seed(455)
        attention = build_model(attention=True, physical=physical).eval()
        inputs = {
            "weather": torch.randn(2, 2, 2, 4, 4),
            "sic": torch.randn(2, 2, 1, 4, 4),
        }
    for name, value in default.state_dict().items():
        torch.testing.assert_close(attention.state_dict()[name], value, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(attention(inputs), default(inputs), rtol=0, atol=0)


def test_clean_latent_encoding_is_independent_of_learned_conditioning_gates() -> None:
    """A raw target for full-latent training must stay in the decoder's basis."""
    model = build_model(attention=True).eval()
    inputs = {"weather": torch.randn(2, 2, 2, 4, 4), "sic": torch.randn(2, 2, 1, 4, 4)}
    with torch.no_grad():
        expected = torch.cat(
            [encoder.rollout(inputs[encoder.name]) for encoder in model.encoders], dim=2
        )
        _heads(model.fusion)[0].bias.fill_(2.0)
        actual = model._encode_inputs(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_conditioning_gates_change_forecasts_and_receive_gradients() -> None:
    """Nonzero gates affect the real model path without changing its clean latents."""
    model = build_model(attention=True).eval()
    inputs = {"weather": torch.randn(2, 2, 2, 4, 4), "sic": torch.randn(2, 2, 1, 4, 4)}
    before = model(inputs).detach()
    with torch.no_grad():
        _heads(model.fusion)[0].bias.fill_(2.0)
    after = model(inputs)
    assert not torch.equal(before, after)
    after.square().mean().backward()
    for head in _heads(model.fusion):
        assert head.bias.grad is not None
        assert torch.isfinite(head.bias.grad).all()
        assert head.bias.grad.abs().sum() > 0


def test_learned_attention_roundtrips_weights_only_checkpoint(tmp_path: Path) -> None:
    """Preserve the learned gates, configuration, predictions and next Adam update."""
    model = build_model(attention=True).train()
    batch = {"weather": torch.randn(2, 2, 2, 4, 4), "sic": torch.randn(2, 2, 1, 4, 4)}
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    model(batch).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": lightning.__version__,
        "optimizer_states": [optimizer.state_dict()],
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "attention.ckpt"
    torch.save(checkpoint, path)
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(path)
    restored = EncodeProcessDecode.load_from_checkpoint(
        path, **_loading_components(), weights_only=True
    )
    assert restored.fusion.mode == "attention"
    assert any(torch.count_nonzero(p) for p in restored.fusion.parameters())
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch), model(batch), rtol=0, atol=0)
    model.train()
    restored.train()
    restored_optimizer = torch.optim.Adam(
        (p for p in restored.parameters() if p.requires_grad), lr=1e-3
    )
    restored_optimizer.load_state_dict(
        torch.load(path, weights_only=True)["optimizer_states"][0]
    )
    for candidate, opt in ((model, optimizer), (restored, restored_optimizer)):
        candidate(batch).square().mean().backward()
        opt.step()
    torch.testing.assert_close(
        restored.state_dict(), model.state_dict(), rtol=0, atol=0
    )


def _loading_components() -> dict[str, Any]:
    return {
        "encoders": DictConfig(
            {
                "latent_space": [4, 4],
                "weather": {"_target_": "cryocast.models.encoders.NaiveLinearEncoder"},
                "sic": {"_target_": "cryocast.models.encoders.NaiveLinearEncoder"},
            }
        ),
        "processor": DictConfig(
            {"_target_": "cryocast.models.processors.NullProcessor"}
        ),
        "decoder": DictConfig(
            {"_target_": "cryocast.models.decoders.NaiveLinearDecoder"}
        ),
        "metrics": [],
    }


def test_checkpoint_without_fusion_configuration_remains_concat(tmp_path: Path) -> None:
    """The new module adds no tensors to ordinary concatenation checkpoints."""
    model = build_model(attention=False).eval()
    hyperparameters = dict(model.hparams)
    hyperparameters.pop("fusion", None)
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": hyperparameters,
        "pytorch-lightning_version": lightning.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "concat.ckpt"
    torch.save(checkpoint, path)
    loaded = EncodeProcessDecode.load_from_checkpoint(
        path, **_loading_components(), weights_only=True
    ).eval()
    assert loaded.fusion.mode == "concat"
    assert not loaded.fusion.state_dict()
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)


def test_attention_example_matches_underlying_model_configuration() -> None:
    """Keep the optional attention example matched as the base architecture evolves."""
    directory = str(Path(cryocast.__file__).parent / "config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        plain = compose(config_name="base", overrides=["model=cnn_unet_cnn"])
        gated = compose(config_name="base", overrides=["model=cnn_unet_cnn_attention"])
    plain_model = OmegaConf.to_container(plain.model, resolve=False)
    gated_model = OmegaConf.to_container(gated.model, resolve=False)
    assert isinstance(plain_model, dict)
    assert isinstance(gated_model, dict)
    gated_model.pop("fusion")
    gated_model["name"] = plain_model["name"]
    assert gated_model == plain_model


def _heads(fusion: LatentFusion) -> list[torch.nn.Linear]:
    return [cast("torch.nn.Linear", head) for head in fusion.score_heads]
