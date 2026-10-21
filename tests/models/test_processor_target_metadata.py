"""Keep prediction slices distinct from full-latent supervision."""

from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from cryocast.models import EncodeProcessDecode
from cryocast.models.processors import BaseProcessor, DiffusionProcessor
from cryocast.types import DataSpace


@pytest.mark.parametrize("latent_loss", [False, True])
def test_target_metadata_does_not_narrow_latent_supervision(
    *, latent_loss: bool
) -> None:
    """A prediction slice is context, not the shape of the clean training target."""
    combined = DataSpace(name="combined", channels=6, shape=(16, 16))
    target = DataSpace(name="sic", channels=2, shape=(16, 16))
    processor = BaseProcessor(
        data_space=combined,
        data_space_target=combined,
        target_latent_space=target,
        target_channel_offset=3,
        computes_loss_in_latent_space=latent_loss,
        n_history_steps=2,
        n_forecast_steps=2,
    )
    assert processor.data_space_target == combined
    assert processor.target_latent_space == target
    assert processor.target_channel_offset == 3
    assert not processor.state_dict()


@pytest.mark.parametrize(
    ("channels", "shape", "offset"),
    [
        (2, (8, 8), 0),
        (7, (16, 16), 0),
        (2, (16, 16), -1),
        (2, (16, 16), 5),
        (2, (16, 16), None),
        (2, (16, 16), True),
        (2, (16, 16), 1.5),
    ],
)
def test_invalid_explicit_target_metadata_is_rejected(
    channels: int, shape: tuple[int, int], offset: object
) -> None:
    """Do not accept an absent, misaligned or out-of-bounds explicit slice."""
    with pytest.raises(ValueError, match="target"):
        BaseProcessor(
            data_space=DataSpace(name="combined", channels=6, shape=(16, 16)),
            target_latent_space=DataSpace(name="sic", channels=channels, shape=shape),
            target_channel_offset=cast("int | None", offset),
            n_history_steps=2,
            n_forecast_steps=2,
        )


def test_legacy_prediction_slice_remains_accepted() -> None:
    """Direct prediction-only callers can still supply the original target fields."""
    target = DataSpace(name="sic", channels=2, shape=(16, 16))
    processor = BaseProcessor(
        data_space=DataSpace(name="combined", channels=6, shape=(16, 16)),
        data_space_target=target,
        target_channel_offset=1,
        n_history_steps=2,
        n_forecast_steps=2,
    )
    assert processor.data_space_target == target
    assert processor.target_latent_space is None
    assert processor.target_channel_offset == 1


def _model(
    order: tuple[str, ...], *, diffusion: bool, output_name: str = "sic"
) -> EncodeProcessDecode:
    """Build unequal encoder widths with a physical target subset."""
    physical_channels = {"weather": 1, "sic": 3, "argo": 2}
    encoders: dict[str, Any] = {"latent_space": [16, 16]}
    # Reverse config order to verify that registered input order controls offsets.
    for name in reversed(order):
        encoders[name] = {
            "_target_": "cryocast.models.encoders.CNNEncoder",
            "n_layers": 1,
            "scale_factor": 2,
        }
    processor: dict[str, Any] = {"_target_": "cryocast.models.processors.NullProcessor"}
    if diffusion:
        processor = {
            "_target_": "cryocast.models.processors.DiffusionProcessor",
            "timesteps": 4,
            "ddim_steps": 2,
            "eta": 0.0,
            "start_out_channels": 4,
            "dropout_rate": 0.0,
            "loss": {"_target_": "torch.nn.MSELoss"},
        }
    return EncodeProcessDecode(
        name="target-metadata",
        encoders=DictConfig(encoders),
        processor=DictConfig(processor),
        decoder=DictConfig({"_target_": "cryocast.models.decoders.NaiveLinearDecoder"}),
        hemisphere="north",
        input_spaces=[
            {"name": n, "channels": physical_channels[n], "shape": [16, 16]}
            for n in order
        ],
        output_space={"name": output_name, "channels": 1, "shape": [16, 16]},
        target_variable_indices=[2] if output_name == "sic" else [0],
        n_history_steps=2,
        n_forecast_steps=2,
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        metrics=[],
        optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
    )


@pytest.mark.parametrize(
    "order",
    [
        ("sic", "argo", "weather"),
        ("argo", "sic", "weather"),
        ("weather", "argo", "sic"),
    ],
)
@pytest.mark.parametrize("diffusion", [False, True])
def test_model_derives_prediction_slice_from_encoded_widths(
    order: tuple[str, ...], *, diffusion: bool
) -> None:
    """Physical target indices must never be treated as latent-channel indices."""
    model = _model(order, diffusion=diffusion)
    target_encoder = next(e for e in model.encoders if e.name == "sic")
    offset = sum(
        e.data_space_out.channels for e in model.encoders[: order.index("sic")]
    )
    assert model.processor.target_latent_space == target_encoder.data_space_out
    assert model.processor.target_latent_space is not None
    assert model.processor.target_latent_space.channels == 6
    assert model.processor.target_channel_offset == offset
    assert model.processor.data_space_target == model.processor.data_space
    assert model.processor.data_space.channels == 12
    assert not hasattr(model, "target_encoder")


@pytest.mark.parametrize("diffusion", [False, True])
def test_output_outside_inputs_has_no_invented_prediction_slice(
    *, diffusion: bool
) -> None:
    """Full-latent models may decode a separate output without inventing its history."""
    model = _model(
        ("weather", "argo", "sic"), diffusion=diffusion, output_name="diagnostic"
    )
    assert model.processor.target_latent_space is None
    assert model.processor.target_channel_offset is None
    assert model.processor.data_space_target == model.processor.data_space


@pytest.mark.parametrize("autoregressive", [False, True])
@pytest.mark.parametrize(("ddim_steps", "eta"), [(None, 1.0), (2, 0.0)])
def test_diffusion_ignores_prediction_slice_for_training_and_sampling(
    *, autoregressive: bool, ddim_steps: int | None, eta: float
) -> None:
    """Adding metadata preserves full-latent loss, gradients, sampling and Adam state."""
    combined = DataSpace(name="combined", channels=3, shape=(16, 16))
    options: dict[str, Any] = {
        "data_space": combined,
        "data_space_target": combined,
        "n_history_steps": 2,
        "n_forecast_steps": 2,
        "timesteps": 4,
        "ddim_steps": ddim_steps,
        "eta": eta,
        "start_out_channels": 4,
        "dropout_rate": 0.0,
        "use_autoregressive": autoregressive,
        "loss": nn.MSELoss(),
    }
    torch.manual_seed(596)
    reference = DiffusionProcessor(**options)
    torch.manual_seed(596)
    candidate = DiffusionProcessor(
        **options,
        target_latent_space=DataSpace(name="sic", channels=1, shape=(16, 16)),
        target_channel_offset=1,
    )
    torch.testing.assert_close(
        candidate.state_dict(), reference.state_dict(), rtol=0, atol=0
    )
    assert candidate.data_space_target == combined
    assert candidate.c_combined == 3
    history = torch.randn(2, 2, 3, 16, 16)
    future = torch.randn(2, 2, 3, 16, 16)
    losses, gradients, forecasts = [], [], []
    for processor in (reference, candidate):
        optimizer = torch.optim.Adam(processor.parameters(), lr=1e-3)
        processor.train()
        torch.manual_seed(516)
        loss = processor.rollout(history, future).loss
        assert loss is not None
        loss.backward()
        losses.append(loss.detach())
        gradients.append(
            {
                name: None if p.grad is None else p.grad.clone()
                for name, p in processor.named_parameters()
            }
        )
        optimizer.step()
        processor.eval()
        with torch.no_grad():
            torch.manual_seed(243)
            forecasts.append(processor.rollout(history).prediction)
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    torch.testing.assert_close(forecasts[0], forecasts[1], rtol=0, atol=0)
    torch.testing.assert_close(
        candidate.state_dict(), reference.state_dict(), rtol=0, atol=0
    )
