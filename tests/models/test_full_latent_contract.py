"""Reference cases for full-latent supervision, isolation and checkpoint loading."""

from pathlib import Path
from typing import Any

import lightning
import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from cryocast.data.combined_dataset import forecast_input_key
from cryocast.models import EncodeProcessDecode
from cryocast.models.encoders import BaseEncoder
from cryocast.models.processors import BaseProcessor, DiffusionProcessor
from cryocast.types import DataSpace, Metadata, ProcessorOutput, TensorNCHW, TensorNTCHW


class _ScaleEncoder(BaseEncoder):
    """Deterministic, differentiable encoding with an observable gradient context."""

    def __init__(self, name: str, channels: int, scale: float) -> None:
        super().__init__(
            data_space_in=DataSpace(name=name, channels=channels, shape=(16, 16)),
            latent_space=(16, 16),
            output_channels=channels,
        )
        self.scale = nn.Parameter(torch.tensor(scale))
        self.grad_modes: list[bool] = []

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        self.grad_modes.append(torch.is_grad_enabled())
        return self.scale * x


class _ReferenceProcessor(BaseProcessor):
    """Squared-error objective whose conditioning gradient has a closed form."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(computes_loss_in_latent_space=True, **kwargs)
        self.gain = nn.Parameter(torch.tensor(0.3))
        self.target: TensorNTCHW | None = None

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        prediction = self.gain * x[:, -1:].expand(-1, self.n_forecast_steps, -1, -1, -1)
        self.target = y
        loss = None if y is None else (prediction - y).square().mean()
        return ProcessorOutput(prediction=prediction, loss=loss)


def _model(*, target_is_input: bool = True, latent: bool = True) -> EncodeProcessDecode:
    target_name = "sic" if target_is_input else "diagnostic"
    processor: dict[str, Any] = {
        "_target_": f"{__name__}._ReferenceProcessor"
        if latent
        else "cryocast.models.processors.NullProcessor",
    }
    return EncodeProcessDecode(
        name="full-latent-reference",
        encoders=[_ScaleEncoder("weather", 1, 2.0), _ScaleEncoder("sic", 2, 1.5)],
        processor=DictConfig(processor),
        decoder=DictConfig({"_target_": "cryocast.models.decoders.NaiveLinearDecoder"}),
        target_variable_indices=[1] if target_is_input else [0],
        hemisphere="north",
        input_spaces=[
            {"name": "sic", "channels": 2, "shape": [16, 16]},
            {"name": "weather", "channels": 1, "shape": [16, 16]},
        ],
        output_space={"name": target_name, "channels": 1, "shape": [16, 16]},
        n_history_steps=2,
        n_forecast_steps=2,
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
        metrics=[],
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "weather": torch.full((2, 2, 1, 16, 16), 0.2, requires_grad=True),
        "sic": torch.full((2, 2, 2, 16, 16), 0.1, requires_grad=True),
        forecast_input_key("weather"): torch.full(
            (2, 2, 1, 16, 16), 0.8, requires_grad=True
        ),
        forecast_input_key("sic"): torch.full(
            (2, 2, 2, 16, 16), 0.7, requires_grad=True
        ),
        "target": torch.full((2, 2, 1, 16, 16), 0.7, requires_grad=True),
    }


def _processor(*, autoregressive: bool) -> DiffusionProcessor:
    return DiffusionProcessor(
        data_space=DataSpace(name="combined", channels=3, shape=(16, 16)),
        n_history_steps=2,
        n_forecast_steps=2,
        timesteps=4,
        ddim_steps=2,
        eta=0.0,
        start_out_channels=4,
        time_embed_dim=256,
        dropout_rate=0.0,
        use_autoregressive=autoregressive,
        loss=nn.MSELoss(),
    )


def test_future_target_uses_encoder_order_and_no_autograd_graph() -> None:
    """Encoder registration order, not dict order, defines the latent channel order."""
    model = _model()
    batch = _batch()
    for encoder in model.encoders:
        assert isinstance(encoder, _ScaleEncoder)
        encoder.grad_modes.clear()
    future = model._encode_forecast_inputs(batch)
    expected = torch.cat(
        [
            batch[forecast_input_key("sic")] * 1.5,
            batch[forecast_input_key("weather")] * 2.0,
        ],
        dim=2,
    )
    torch.testing.assert_close(future, expected)
    assert not future.requires_grad
    assert future.grad_fn is None
    for encoder in model.encoders:
        assert isinstance(encoder, _ScaleEncoder)
        assert encoder.grad_modes == [False]
        assert encoder.training
        assert encoder.scale.requires_grad


def test_conditioning_gradients_exclude_future_target_and_decoder() -> None:
    """The analytic gradient includes history encoding but no target-encoding path."""
    model = _model().train()
    batch = _batch()
    result = model.training_step(batch, 0)
    result.loss.backward()
    for encoder in model.encoders:
        assert isinstance(encoder, _ScaleEncoder)
        history = batch[encoder.name][:, -1:].expand(-1, 2, -1, -1, -1)
        future = batch[forecast_input_key(encoder.name)]
        # The objective averages over all three combined channels.
        expected = (
            2 * 0.3 * encoder.scale.detach() * (0.3 * history - future) * history
        ).sum() / (2 * 2 * 3 * 16 * 16)
        torch.testing.assert_close(encoder.scale.grad, expected)
        assert batch[encoder.name].grad is not None
        assert batch[forecast_input_key(encoder.name)].grad is None
    assert batch["target"].grad is None
    assert all(
        p.grad is None and not p.requires_grad for p in model.decoder.parameters()
    )
    assert isinstance(model.processor, _ReferenceProcessor)
    assert model.processor.gain.grad is not None


@pytest.mark.parametrize("target_is_input", [False, True])
def test_no_skip_connection_does_not_require_target_history(
    *, target_is_input: bool
) -> None:
    """Full-latent supervision supports an output group outside the model inputs."""
    model = _model(target_is_input=target_is_input)
    batch = _batch()
    result = model.training_step(batch, 0)
    assert result.prediction.shape == batch["target"].shape
    with torch.no_grad():
        prediction = model({key: batch[key] for key in ("sic", "weather")})
    assert prediction.shape == result.prediction.shape


@pytest.mark.parametrize("missing", ["sic", "weather"])
def test_missing_future_group_is_rejected_before_encoding(missing: str) -> None:
    """Every modality must supply its actual future window."""
    model = _model()
    batch = _batch()
    del batch[forecast_input_key(missing)]
    for encoder in model.encoders:
        assert isinstance(encoder, _ScaleEncoder)
        encoder.grad_modes.clear()
    with pytest.raises(ValueError, match="forecast input windows"):
        model._encode_forecast_inputs(batch)
    for encoder in model.encoders:
        assert isinstance(encoder, _ScaleEncoder)
        assert not encoder.grad_modes


@pytest.mark.parametrize(
    "shape", [(1, 2, 2, 16, 16), (2, 1, 2, 16, 16), (2, 2, 1, 16, 32), (2, 2, 2, 8, 32)]
)
def test_malformed_future_window_cannot_be_silently_reshaped(
    shape: tuple[int, ...],
) -> None:
    """Equal element counts must not conceal incorrect batch/channel/spatial axes."""
    model = _model()
    batch = _batch()
    batch[forecast_input_key("sic")] = torch.zeros(shape)
    with pytest.raises(ValueError, match=r"Forecast input.*sic.*shape"):
        model._encode_forecast_inputs(batch)


@pytest.mark.parametrize("autoregressive", [False, True])
@pytest.mark.parametrize(
    "shape", [(1, 2, 3, 16, 16), (2, 1, 3, 16, 16), (2, 2, 1, 16, 16), (2, 2, 3, 8, 32)]
)
def test_diffusion_requires_complete_matching_latent_shape(
    *, autoregressive: bool, shape: tuple[int, ...]
) -> None:
    """Partial targets cannot broadcast or reshape into full-latent supervision."""
    processor = _processor(autoregressive=autoregressive)
    history = torch.zeros(2, 2, 3, 16, 16)
    with pytest.raises(ValueError, match=r"Forecast latent.*shape"):
        processor.rollout(history, torch.zeros(shape))


@pytest.mark.parametrize("autoregressive", [False, True])
def test_real_diffusion_trains_all_combined_channels(*, autoregressive: bool) -> None:
    """A real UNet backpropagates through conditioning and all output channels."""
    torch.manual_seed(596)
    processor = _processor(autoregressive=autoregressive)
    history = torch.randn(2, 2, 3, 16, 16, requires_grad=True)
    target = torch.randn(2, 2, 3, 16, 16)
    result = processor.rollout(history, target)
    assert result.loss is not None
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert history.grad is not None
    assert (history.grad.abs().sum(dim=(0, 1, 3, 4)) > 0).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in processor.model.parameters()
        if p.requires_grad
    )
    assert result.prediction.shape == target.shape
    assert not result.prediction.requires_grad


@pytest.mark.parametrize("autoregressive", [False, True])
@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
def test_diffusion_inference_ignores_future_values_and_roundtrips_weights(
    *, autoregressive: bool, sampler: str
) -> None:
    """Strict state loading preserves seeded full-latent inference for both samplers."""
    processor = _processor(autoregressive=autoregressive).eval()
    restored = _processor(autoregressive=autoregressive).eval()
    restored.load_state_dict(processor.state_dict(), strict=True)
    for module in (processor, restored):
        module.set_sampler(
            ddim_steps=None if sampler == "ddpm" else 2,
            eta=1.0 if sampler == "ddpm" else 0.0,
        )
    history = torch.ones(1, 2, 3, 16, 16)
    with torch.no_grad():
        torch.manual_seed(596)
        expected = processor.rollout(history).prediction
        torch.manual_seed(596)
        actual = restored.rollout(history).prediction
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual[:, :, 2], torch.ones_like(actual[:, :, 2]))


def test_model_inference_does_not_read_future_targets() -> None:
    """Extra supervision keys cannot affect a forecast made from the same history."""
    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        expected = model({key: batch[key] for key in ("sic", "weather")})
        batch[forecast_input_key("sic")] = torch.full_like(batch["sic"], float("nan"))
        batch[forecast_input_key("weather")] = torch.full_like(
            batch["weather"], float("inf")
        )
        batch["target"] = torch.full_like(batch["target"], float("nan"))
        actual = model(batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint(model: EncodeProcessDecode) -> dict[str, Any]:
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    return checkpoint


def test_new_full_latent_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Versioned checkpoint reload is strict and preserves metadata and parameters."""
    original = _model()
    original.training_metadata = Metadata(training_start="2020-01-01")
    checkpoint = _checkpoint(original)
    assert checkpoint.get("latent_target_layout") == "full_combined_v1"
    path = tmp_path / "full.ckpt"
    torch.save(checkpoint, path)
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(path)
    template = _model()
    loaded = EncodeProcessDecode.load_from_checkpoint(
        path,
        encoders=template.encoders,
        processor=template.processor,
        decoder=template.decoder,
        metrics=[],
        weights_only=True,
    )
    torch.testing.assert_close(
        loaded.state_dict(), original.state_dict(), atol=0, rtol=0
    )
    assert loaded.training_metadata == original.training_metadata


@pytest.mark.parametrize("layout", [None, "target_slice_v0", "full_combined_v2"])
def test_unversioned_or_incompatible_latent_checkpoint_is_rejected(
    layout: str | None,
) -> None:
    """Even equal-size legacy weights cannot prove the changed latent-target contract."""
    model = _model()
    checkpoint = _checkpoint(model)
    checkpoint.pop("latent_target_layout", None)
    if layout is not None:
        checkpoint["latent_target_layout"] = layout
    with pytest.raises(ValueError, match=r"full combined-latent.*checkpoint"):
        model.on_load_checkpoint(checkpoint)


def test_legacy_nonlatent_checkpoint_drops_only_unused_target_encoder(
    tmp_path: Path,
) -> None:
    """Removing the old unused branch must not break ordinary forecasting checkpoints."""
    model = _model(latent=False).eval()
    checkpoint = _checkpoint(model)
    checkpoint["state_dict"]["target_encoder.unused.weight"] = torch.randn(1, 1)
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)
    template = _model(latent=False)
    loaded = EncodeProcessDecode.load_from_checkpoint(
        path,
        encoders=template.encoders,
        processor=template.processor,
        decoder=template.decoder,
        metrics=[],
        weights_only=True,
    ).eval()
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded(_batch()), model(_batch()), atol=0, rtol=0)
    # Migration is in memory; the original checkpoint retains its legacy weights.
    assert (
        "target_encoder.unused.weight"
        in torch.load(path, weights_only=True)["state_dict"]
    )


def test_nonlatent_checkpoint_still_rejects_unrelated_weights(tmp_path: Path) -> None:
    """Compatibility must not silently weaken strict state loading."""
    model = _model(latent=False)
    checkpoint = _checkpoint(model)
    checkpoint["state_dict"]["unrelated.weight"] = torch.zeros(1)
    path = tmp_path / "invalid.ckpt"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        EncodeProcessDecode.load_from_checkpoint(
            path,
            encoders=model.encoders,
            processor=model.processor,
            decoder=model.decoder,
            metrics=[],
            weights_only=True,
        )


@pytest.mark.parametrize("autoregressive", [False, True])
def test_trained_diffusion_model_checkpoint_restores_loss_and_forecast(
    tmp_path: Path,
    *,
    autoregressive: bool,
) -> None:
    """One optimizer update survives a real model checkpoint in both training modes."""
    torch.manual_seed(596)
    model = _model()
    model.processor = _processor(autoregressive=autoregressive)
    batch = _batch()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    model.training_step(batch, 0).loss.backward()
    optimizer.step()
    model.eval()
    path = tmp_path / "trained-latent.ckpt"
    torch.save(_checkpoint(model), path)
    template = _model()
    restored = EncodeProcessDecode.load_from_checkpoint(
        path,
        encoders=template.encoders,
        processor=_processor(autoregressive=autoregressive),
        decoder=template.decoder,
        metrics=[],
        weights_only=True,
    ).eval()
    with torch.no_grad():
        torch.manual_seed(91)
        loss = model.training_step(batch, 0).loss
        torch.manual_seed(91)
        restored_loss = restored.training_step(batch, 0).loss
        torch.testing.assert_close(restored_loss, loss, atol=0, rtol=0)
        torch.manual_seed(92)
        prediction = model(batch)
        torch.manual_seed(92)
        restored_prediction = restored(batch)
        torch.testing.assert_close(restored_prediction, prediction, atol=0, rtol=0)
    assert torch.isfinite(restored_prediction).all()


@pytest.mark.parametrize(
    "shape", [(2, 1, 3, 16, 16), (2, 2, 1, 16, 16), (2, 2, 3, 8, 32), (2, 3, 16, 16)]
)
def test_diffusion_checks_history_axes_before_denoising(shape: tuple[int, ...]) -> None:
    """Invalid histories fail before any stochastic sampling or reshape."""
    processor = _processor(autoregressive=True)
    with pytest.raises(ValueError, match="History latent has shape"):
        processor.rollout(torch.zeros(shape))
