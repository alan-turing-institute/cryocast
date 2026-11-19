import logging
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import lightning
import pytest
import torch
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from cryocast.losses import LeadTimeWeightedLoss
from cryocast.models import EncodeProcessDecode
from cryocast.models.decoders import BaseDecoder
from cryocast.models.encoders import BaseEncoder
from cryocast.models.processors import BaseProcessor
from cryocast.types import (
    DataSpace,
    Hemisphere,
    ModelStepOutput,
    ProcessorOutput,
    RolloutSpace,
    SkipAnchorType,
    TensorNCHW,
    TensorNTCHW,
)


class _ScaledEncoder(BaseEncoder):
    """Perfect encoder that represents a physical value x as the latent scale * x."""

    def __init__(self, *, data_space_in: DataSpace, scale: float) -> None:
        super().__init__(
            data_space_in=data_space_in,
            latent_space=data_space_in.shape,
            output_channels=data_space_in.channels,
        )
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.scale * x


class _InverseDecoder(BaseDecoder):
    """Decoder that divides one channel of the combined latent by scale."""

    def __init__(self, *, channel: int, scale: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.channel = channel
        self.scale = scale

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return x[:, self.channel : self.channel + 1] / self.scale


class _OracleProcessor(BaseProcessor):
    """Processor that forecasts the future latent it is given, i.e. a perfect forecast."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(computes_loss_in_latent_space=True, **kwargs)
        self.target: TensorNTCHW | None = None

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        assert y is not None
        assert y.shape[2] == x.shape[2]
        self.target = y
        return ProcessorOutput(prediction=y.clone(), loss=torch.zeros(()))


class _PredictionProcessor(BaseProcessor):
    """Processor that returns a zero latent loss and the prediction gain * x.

    The prediction is the last history latents times a trainable gain: one step in
    autoregressive mode, n_forecast_steps steps in parallel mode, or `n_steps` steps if
    set. Set `sample_mask` to choose the samples for the prediction loss (default: None,
    i.e. all samples).
    """

    def __init__(
        self,
        *,
        use_autoregressive: bool = True,
        n_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(computes_loss_in_latent_space=True, **kwargs)
        self.use_autoregressive = use_autoregressive
        self.n_steps = n_steps
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.sample_mask: torch.Tensor | None = None

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        assert y is not None
        n_steps = self.n_steps or (
            1 if self.use_autoregressive else self.n_forecast_steps
        )
        return ProcessorOutput(
            prediction=self.gain * x[:, -n_steps:],
            loss=torch.zeros(()),
            sample_mask=self.sample_mask,
        )


class _RepeatLastProcessor(BaseProcessor):
    """Latent-loss processor that repeats the last history latent at every step.

    Unlike _OracleProcessor it does not read the future latent, so the forward pass
    and the training step give the same prediction.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(computes_loss_in_latent_space=True, **kwargs)

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        return ProcessorOutput(
            prediction=x[:, -1:].repeat(1, self.n_forecast_steps, 1, 1, 1),
            loss=torch.zeros(()),
        )


class _ScaledDecoder(BaseDecoder):
    """Trainable decoder that multiplies one channel of the combined latent by scale."""

    def __init__(self, *, channel: int, scale: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.channel = channel
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return x[:, self.channel : self.channel + 1] * self.scale


class _BatchNormDecoder(_ScaledDecoder):
    """_ScaledDecoder followed by BatchNorm, to track the running statistics."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.norm(super().forward(x))


def _oracle_model(
    *,
    cfg_loss: DictConfig,
    cfg_metrics: list[dict[str, Any]],
    loss_weights: dict[str, float] | None = None,
    n_forecast_steps: int = 2,
    processor: DictConfig | None = None,
    decoder_scale: float = 2.0,
    skip_connection: dict[str, Any] | None = None,
    skip_anchor: str | None = None,
    relaxation: dict[str, Any] | None = None,
) -> EncodeProcessDecode:
    """Build a latent-loss model with a trainable encoder and a trainable decoder.

    The decoder doubles the target channel, so the reconstruction loss is not zero. The
    latent loss from the processor (default _OracleProcessor) is always zero.
    """
    shape = (8, 8)
    sic = DataSpace(channels=2, name="sic", shape=shape)
    output_space = DataSpace(channels=1, name="sic", shape=shape)
    return EncodeProcessDecode(
        name="oracle",
        encoders=[_ScaledEncoder(data_space_in=sic, scale=1.0)],
        processor=processor
        or DictConfig({"_target_": f"{__name__}.{_OracleProcessor.__name__}"}),
        decoder=_ScaledDecoder(
            channel=1,
            scale=decoder_scale,
            data_space_in=DataSpace(channels=2, name="combined", shape=shape),
            data_space_out=output_space,
            skip_connection=skip_connection,
        ),
        loss_weights=loss_weights,
        skip_anchor=skip_anchor,
        relaxation=relaxation,
        hemisphere="north",
        input_spaces=[sic.to_dict()],
        loss=cfg_loss,
        metrics=cfg_metrics,
        n_forecast_steps=n_forecast_steps,
        n_history_steps=2,
        output_space=output_space.to_dict(),
        optimizer=DictConfig({}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
        target_variable_indices=[1],
    )


def _oracle_batch(n_forecast_steps: int = 2) -> dict[str, TensorNTCHW]:
    """Build a batch for the model from `_oracle_model`."""
    future_sic = torch.rand(2, n_forecast_steps, 2, 8, 8)
    return {
        "sic": torch.rand(2, 2, 2, 8, 8),
        "future_sic": future_sic,
        "target": future_sic[:, :, 1:2].clone(),
    }


def _training_step(
    model: EncodeProcessDecode, batch: dict[str, TensorNTCHW]
) -> tuple[ModelStepOutput, dict[str, torch.Tensor]]:
    """Run a training step and return the output and the logged values."""
    with pytest.MonkeyPatch.context() as mp:
        mock_log = MagicMock()
        mp.setattr(model, "log", mock_log)
        result = model.training_step(batch, 0)
    return result, {c.args[0]: c.args[1] for c in mock_log.call_args_list}


class TestEncodeProcessDecode:
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2, 5], ids=["forecast1", "forecast2", "forecast5"]
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2, 5], ids=["history1", "history2", "history5"]
    )
    def test_init(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            loss=cfg_loss,
            metrics=cfg_metrics,
            target_variable_indices=[0],
        )

        assert model.name == "encode-null-decode"
        assert model.input_spaces == [DataSpace.from_dict(cfg_input_space)]
        assert model.n_forecast_steps == test_n_forecast_steps
        assert model.n_history_steps == test_n_history_steps
        assert model.output_space == DataSpace.from_dict(cfg_output_space)

    @pytest.mark.parametrize(
        ("test_computes_loss_in_latent_space", "expected_needs_future_inputs"),
        [(False, False), (True, True)],
        ids=["default", "latent-loss"],
    )
    def test_init_needs_future_inputs(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        *,
        test_computes_loss_in_latent_space: bool,
        expected_needs_future_inputs: bool,
    ) -> None:
        cfg_processor = DictConfig(
            {
                **cfg_processor,
                "computes_loss_in_latent_space": test_computes_loss_in_latent_space,
            }
        )
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        assert model.needs_future_inputs is expected_needs_future_inputs

    def test_init_processor_builds_its_own_loss(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        """A processor `loss` config reaches the processor uninstantiated.

        Hydra must not instantiate nested processor configs itself: a loss config
        carrying `lead_time_exponent` is only valid when built by `build_loss`.
        """
        loss = OmegaConf.merge(cfg_loss, {"lead_time_exponent": 1.0})
        output_space = DictConfig(
            {
                "channels": 1,
                "name": cfg_input_space["name"],
                "shape": cfg_input_space["shape"],
            }
        )
        processor = DictConfig(
            {
                "_target_": "cryocast.models.processors.DiffusionProcessor",
                "timesteps": 2,
                "start_out_channels": 8,
                "loss": loss,
            }
        )
        model = EncodeProcessDecode(
            name="diffusion-latent-loss",
            encoders=cfg_encoders,
            processor=processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            loss=loss,
            metrics=cfg_metrics,
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        assert isinstance(model.processor.loss_fn, LeadTimeWeightedLoss)
        assert model.needs_future_inputs

    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2, 5], ids=["forecast1", "forecast2", "forecast5"]
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2, 5], ids=["history1", "history2", "history5"]
    )
    @pytest.mark.parametrize(
        "test_batch_size", [1, 2, 5], ids=["batch1", "batch2", "batch5"]
    )
    def test_forward(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        result: torch.Tensor = model(
            {
                cfg_input_space["name"]: torch.randn(
                    test_batch_size,
                    test_n_history_steps,
                    cfg_input_space["channels"],
                    cfg_input_space["shape"][0],
                    cfg_input_space["shape"][1],
                ),
                cfg_output_space["name"]: torch.rand(
                    test_batch_size,
                    test_n_history_steps,
                    cfg_output_space["channels"],
                    cfg_output_space["shape"][0],
                    cfg_output_space["shape"][1],
                ),
            }
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            cfg_output_space["channels"],
            cfg_output_space["shape"][0],
            cfg_output_space["shape"][1],
        )

    def test_training_step_applies_lead_time_weighted_loss(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        """The decoded-loss path weights each lead time of the decoded prediction."""
        batch_size, n_forecast_steps, n_history_steps = 2, 3, 2
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            loss=OmegaConf.create(
                {"_target_": "torch.nn.MSELoss", "lead_time_exponent": 1.0}
            ),
            metrics=cfg_metrics,
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        assert isinstance(model.loss_fn, LeadTimeWeightedLoss)
        generator = torch.Generator().manual_seed(0)
        batch = {
            cfg_input_space["name"]: torch.randn(
                batch_size,
                n_history_steps,
                cfg_input_space["channels"],
                *cfg_input_space["shape"],
                generator=generator,
            ),
            cfg_output_space["name"]: torch.rand(
                batch_size,
                n_history_steps,
                cfg_output_space["channels"],
                *cfg_output_space["shape"],
                generator=generator,
            ),
            "target": torch.rand(
                batch_size,
                n_forecast_steps,
                cfg_output_space["channels"],
                *cfg_output_space["shape"],
                generator=generator,
            ),
        }

        result = model.training_step(batch, 0)

        # Weights (1, 2, 3) rescaled to mean 1 are (0.5, 1.0, 1.5)
        per_step = torch.stack(
            [
                torch.nn.functional.mse_loss(
                    result.prediction[:, t], result.target[:, t]
                )
                for t in range(n_forecast_steps)
            ]
        )
        expected = (torch.tensor([0.5, 1.0, 1.5]) * per_step).mean()
        assert result.prediction.shape == batch["target"].shape
        assert result.loss.item() == pytest.approx(expected.item())
        # Guard against a degenerate case where weighting makes no difference
        assert result.loss.item() != pytest.approx(per_step.mean().item())

    @pytest.fixture
    def latent_loss_model(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> EncodeProcessDecode:
        output_space = DictConfig(
            {
                "channels": 1,
                "name": cfg_input_space["name"],
                "shape": cfg_input_space["shape"],
            }
        )
        processor = DictConfig(
            {
                "_target_": "cryocast.models.processors.NullProcessor",
                "computes_loss_in_latent_space": True,
            }
        )
        model = EncodeProcessDecode(
            name="latent-loss",
            encoders=cfg_encoders,
            processor=processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=2,
            n_history_steps=2,
            output_space=output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[2],
        )
        model.eval()
        return model

    def test_latent_target_encodes_future_inputs(
        self, latent_loss_model: EncodeProcessDecode, cfg_input_space: DictConfig
    ) -> None:
        """Latent supervision is the encoded inputs over the forecast steps."""
        model = latent_loss_model
        name = cfg_input_space["name"]
        history = torch.rand(2, 2, cfg_input_space["channels"], 16, 16)
        future = torch.rand(2, 2, cfg_input_space["channels"], 16, 16)

        expected = model._encode_inputs({name: future})
        actual = model._encode_future_inputs(
            {name: history, f"future_{name}": future, "target": future[:, :, 2:3]}
        )

        torch.testing.assert_close(actual, expected)
        assert actual.shape[2] == model.processor.data_space.channels

        # The latent loss must not train the shared encoder through the future latent
        assert all(p.requires_grad for p in model.encoders[0].parameters())
        assert not actual.requires_grad

    def test_latent_target_requires_future_inputs(
        self, latent_loss_model: EncodeProcessDecode, cfg_input_space: DictConfig
    ) -> None:
        name = cfg_input_space["name"]
        batch = {
            name: torch.rand(2, 2, cfg_input_space["channels"], 16, 16),
            "target": torch.rand(2, 2, 1, 16, 16),
        }

        with pytest.raises(ValueError, match="needs future inputs"):
            latent_loss_model.training_step(batch, 0)

    def test_perfect_latent_forecast_decodes_to_target(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """A perfect latent forecast decodes to the target."""
        shape = (8, 8)
        era5 = DataSpace(channels=1, name="era5", shape=shape)
        sic = DataSpace(channels=2, name="sic", shape=shape)
        output_space = DataSpace(channels=1, name="sic", shape=shape)
        model = EncodeProcessDecode(
            name="oracle",
            encoders=[
                _ScaledEncoder(data_space_in=era5, scale=1.0),
                _ScaledEncoder(data_space_in=sic, scale=1.0),
            ],
            processor=DictConfig(
                {"_target_": f"{__name__}.{_OracleProcessor.__name__}"}
            ),
            # Combined latent is [era5, sic_0, sic_1]: the target variable is channel 2
            decoder=_InverseDecoder(
                channel=2,
                scale=1.0,
                data_space_in=DataSpace(channels=3, name="combined", shape=shape),
                data_space_out=output_space,
            ),
            hemisphere="north",
            input_spaces=[era5.to_dict(), sic.to_dict()],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=2,
            n_history_steps=2,
            output_space=output_space.to_dict(),
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[1],
        )
        future_sic = torch.rand(2, 2, 2, *shape)
        batch = {
            "era5": torch.rand(2, 2, 1, *shape),
            "sic": torch.rand(2, 2, 2, *shape),
            "future_era5": torch.rand(2, 2, 1, *shape),
            "future_sic": future_sic,
            "target": future_sic[:, :, 1:2].clone(),
        }

        result = model.training_step(batch, 0)

        mae = (result.prediction - result.target).abs().mean()
        assert mae.item() == pytest.approx(0.0, abs=1e-6)

        # The latent loss must not train the shared encoder through the future latent
        processor = model.processor
        assert isinstance(processor, _OracleProcessor)
        assert processor.target is not None
        assert all(p.requires_grad for e in model.encoders for p in e.parameters())
        assert not processor.target.requires_grad

    @pytest.mark.parametrize("test_key", ["reconstruction", "prediction"])
    def test_init_rejects_negative_loss_weight(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]], test_key: str
    ) -> None:
        with pytest.raises(ValueError, match=f"{test_key} must not be negative"):
            _oracle_model(
                cfg_loss=cfg_loss,
                cfg_metrics=cfg_metrics,
                loss_weights={test_key: -1.0},
            )

    def test_init_rejects_unknown_loss_weight(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        with pytest.raises(ValueError, match="Unknown loss_weights"):
            _oracle_model(
                cfg_loss=cfg_loss,
                cfg_metrics=cfg_metrics,
                loss_weights={"autoencodr": 1.0},
            )

    @pytest.mark.parametrize(
        ("test_weights", "expected_weights"),
        [
            (None, {"reconstruction": 0.0, "prediction": 0.0}),
            ({}, {"reconstruction": 0.0, "prediction": 0.0}),
            ({"reconstruction": 0.5}, {"reconstruction": 0.5, "prediction": 0.0}),
            ({"prediction": 0.5}, {"reconstruction": 0.0, "prediction": 0.5}),
        ],
        ids=["none", "empty", "reconstruction-half", "prediction-half"],
    )
    def test_init_loss_weights(
        self,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        test_weights: dict[str, float] | None,
        expected_weights: dict[str, float],
    ) -> None:
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights=test_weights,
        )
        assert model.loss_weights == expected_weights

    def test_zero_weights_freeze_decoder_and_use_latent_loss_only(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """With zero loss weights, the decoder is frozen and only the latent loss is used.

        The decoder stays in eval mode, and the latent loss does not go through the
        encoders or the decoder.
        """
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights={"reconstruction": 0.0},
        )
        model.train()

        result, logged = _training_step(model, _oracle_batch())

        # Only the latent loss from _OracleProcessor (zero) remains
        assert result.loss.item() == 0.0
        assert not result.loss.requires_grad
        assert list(logged) == ["train_loss"]

        # The decoder is frozen, and the encoders are not
        assert not any(p.requires_grad for p in model.decoder.parameters())
        assert not model.decoder.training
        assert all(p.requires_grad for e in model.encoders for p in e.parameters())

        # The latent target has no gradients
        processor = model.processor
        assert isinstance(processor, _OracleProcessor)
        assert processor.target is not None
        assert not processor.target.requires_grad

    def test_reconstruction_loss_trains_encoders_and_decoder(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """The reconstruction loss trains the decoder and the encoders.

        The latent loss is zero, so all of the loss comes from the reconstruction loss.
        """
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights={"reconstruction": 0.5},
        )
        model.train()
        batch = _oracle_batch()
        reconstruction_loss = model.loss(2.0 * batch["target"], batch["target"])

        result, logged = _training_step(model, batch)
        result.loss.backward()

        torch.testing.assert_close(result.loss, 0.5 * reconstruction_loss)
        assert logged["train_latent_loss"].item() == 0.0
        torch.testing.assert_close(
            logged["train_reconstruction_loss"], reconstruction_loss
        )

        # The decoder is trainable, and the gradient reaches it and the encoder
        decoder, encoder = model.decoder, model.encoders[0]
        assert isinstance(decoder, _ScaledDecoder)
        assert isinstance(encoder, _ScaledEncoder)
        assert decoder.training
        assert decoder.scale.grad is not None
        assert decoder.scale.grad.abs() > 0
        assert encoder.scale.grad is not None
        assert encoder.scale.grad.abs() > 0

        # The latent loss must not train the shared encoder through the target latent
        processor = model.processor
        assert isinstance(processor, _OracleProcessor)
        assert processor.target is not None
        assert not processor.target.requires_grad

    def test_checkpoint_loads_with_weights_only(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_processor: DictConfig,
        tmp_path: Path,
    ) -> None:
        """Checkpoints contain only types that torch can load with weights_only."""
        model = EncodeProcessDecode(
            name="weights-only",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere=Hemisphere.NORTH,
            input_spaces=[cfg_input_space],
            loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
            metrics=[],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        samples: list[dict[str, torch.Tensor]] = [
            {
                cfg_input_space["name"]: torch.rand(1, 4, 16, 16),
                "target": torch.rand(1, 1, 16, 16),
            }
        ]
        # A list of samples is a valid map-style dataset
        batches: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            samples,  # type: ignore[arg-type]
            batch_size=1,
        )
        trainer = Trainer(
            accelerator="cpu",
            default_root_dir=tmp_path,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=0,
            logger=False,
            max_steps=1,
        )
        trainer.fit(model, batches)
        checkpoint_file = tmp_path / "model.ckpt"
        trainer.save_checkpoint(checkpoint_file)

        assert (
            torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_file) == []
        )
        loaded = EncodeProcessDecode.load_from_checkpoint(
            checkpoint_file,
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            metrics=[],
            weights_only=True,
        )
        assert loaded.hemisphere is Hemisphere.NORTH
        assert loaded.rollout_strategy.space is RolloutSpace.LATENT

    def test_load_state_dict_ignores_removed_target_encoder(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_processor: DictConfig,
    ) -> None:
        """Checkpoints saved with the removed target encoder still load strictly."""
        model = EncodeProcessDecode(
            name="legacy-target-encoder",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere=Hemisphere.NORTH,
            input_spaces=[cfg_input_space],
            loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
            metrics=[],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        state_dict = model.state_dict()
        legacy_state_dict = {
            **state_dict,
            "target_encoder.model.0.weight": torch.zeros(1),
            "target_encoder.model.0.bias": torch.zeros(1),
        }

        result = model.load_state_dict(legacy_state_dict, strict=True)

        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert "target_encoder" not in dict(model.named_children())


class TestPredictionLoss:
    """The prediction loss compares the decoded differentiable prediction to the target.

    With _PredictionProcessor, _ScaledEncoder (scale 1) and _ScaledDecoder (scale 2), the
    decoded prediction is 2 * sic[:, -n_steps:, 1:2]. The latent loss is zero.
    """

    @staticmethod
    def _model(
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        loss_weights: dict[str, float],
        *,
        use_autoregressive: bool = True,
        n_steps: int | None = None,
        n_forecast_steps: int = 2,
    ) -> EncodeProcessDecode:
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights=loss_weights,
            n_forecast_steps=n_forecast_steps,
            processor=DictConfig(
                {
                    "_target_": f"{__name__}.{_PredictionProcessor.__name__}",
                    "use_autoregressive": use_autoregressive,
                    "n_steps": n_steps,
                }
            ),
        )
        model.train()
        return model

    def test_autoregressive_compares_first_step_only(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss,
            cfg_metrics,
            {"reconstruction": 0.0, "prediction": 0.5},
            use_autoregressive=True,
        )
        batch = _oracle_batch()
        expected = model.loss(2.0 * batch["sic"][:, -1:, 1:2], batch["target"][:, :1])

        result, logged = _training_step(model, batch)

        torch.testing.assert_close(logged["train_prediction_loss"], expected)
        torch.testing.assert_close(result.loss, 0.5 * expected)

        # Changing the later forecast steps of the target does not change the loss
        batch["target"][:, 1:] += 1.0
        changed, _ = _training_step(model, batch)
        torch.testing.assert_close(changed.loss, result.loss)

    def test_parallel_compares_all_steps(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss,
            cfg_metrics,
            {"reconstruction": 0.0, "prediction": 0.5},
            use_autoregressive=False,
        )
        batch = _oracle_batch()
        expected = model.loss(2.0 * batch["sic"][:, -2:, 1:2], batch["target"])

        result, logged = _training_step(model, batch)

        torch.testing.assert_close(logged["train_prediction_loss"], expected)
        torch.testing.assert_close(result.loss, 0.5 * expected)

    def test_zero_prediction_weight_skips_prediction_loss(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """With weight 0, the loss is the latent loss plus the reconstruction loss only."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.5, "prediction": 0.0}
        )
        batch = _oracle_batch()
        reconstruction_loss = model.loss(2.0 * batch["target"], batch["target"])

        result, logged = _training_step(model, batch)

        torch.testing.assert_close(result.loss, 0.5 * reconstruction_loss)
        assert "train_prediction_loss" not in logged

    def test_mask_selects_samples(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )
        processor = model.processor
        assert isinstance(processor, _PredictionProcessor)
        processor.sample_mask = torch.tensor([True, False])
        batch = _oracle_batch()
        expected = model.loss(2.0 * batch["sic"][:1, -1:, 1:2], batch["target"][:1, :1])

        result, _ = _training_step(model, batch)

        torch.testing.assert_close(result.loss, expected)

    def test_empty_mask_gives_zero_not_nan(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )
        processor = model.processor
        assert isinstance(processor, _PredictionProcessor)
        processor.sample_mask = torch.tensor([False, False])

        result, logged = _training_step(model, _oracle_batch())
        result.loss.backward()

        assert logged["train_prediction_loss"].item() == 0.0
        assert result.loss.item() == 0.0
        decoder = model.decoder
        assert isinstance(decoder, _ScaledDecoder)
        assert decoder.scale.grad is not None
        assert decoder.scale.grad.item() == 0.0

    def test_empty_mask_ignores_non_finite_decoded_values(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """An empty mask gives 0, also if decoding the prediction would give inf."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )
        processor = model.processor
        assert isinstance(processor, _PredictionProcessor)
        processor.sample_mask = torch.tensor([False, False])
        batch = _oracle_batch()
        batch["sic"][:, -1] = torch.inf

        _, logged = _training_step(model, batch)

        assert logged["train_prediction_loss"].item() == 0.0

    def test_decodes_only_masked_samples(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """The prediction loss decodes only the samples in the mask."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )
        processor = model.processor
        assert isinstance(processor, _PredictionProcessor)
        processor.sample_mask = torch.tensor([True, False])
        decoded_shapes: list[tuple[bool, torch.Size]] = []
        rollout = model.decoder.rollout

        def spy(
            x: TensorNTCHW, persistence: TensorNTCHW | None, **kwargs: Any
        ) -> TensorNTCHW:
            decoded_shapes.append((torch.is_grad_enabled(), x.shape))
            return rollout(x, persistence, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(model.decoder, "rollout", spy)
            _training_step(model, _oracle_batch())

        # With gradients: one masked sample. Without gradients: the metrics decode.
        assert decoded_shapes == [
            (True, torch.Size([1, 1, 2, 8, 8])),
            (False, torch.Size([2, 1, 2, 8, 8])),
        ]

    def test_metrics_decode_one_autoregressive_step(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """Decode the one predicted step once, then repeat it over the forecast steps."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )
        batch = _oracle_batch()
        expected = 2.0 * batch["sic"][:, -1:, 1:2].expand(-1, 2, -1, -1, -1)

        result, _ = _training_step(model, batch)

        torch.testing.assert_close(result.prediction, expected)
        assert result.prediction.is_contiguous()
        assert not result.prediction.requires_grad

    def test_metrics_pad_intermediate_length_prediction(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """Keep the predicted steps, then repeat the last one over the other steps."""
        model = self._model(
            cfg_loss,
            cfg_metrics,
            {"reconstruction": 0.0, "prediction": 1.0},
            n_steps=2,
            n_forecast_steps=4,
        )
        batch = _oracle_batch(n_forecast_steps=4)
        decoded = 2.0 * batch["sic"][:, -2:, 1:2]
        expected = torch.cat([decoded, decoded[:, -1:].expand(-1, 2, -1, -1, -1)], 1)

        result, logged = _training_step(model, batch)

        torch.testing.assert_close(
            logged["train_prediction_loss"],
            model.loss(decoded, batch["target"][:, :2]),
        )
        torch.testing.assert_close(result.prediction, expected)
        assert result.prediction.is_contiguous()
        assert not result.prediction.requires_grad

    def test_metrics_decode_keeps_batchnorm_statistics(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """The metrics-only decode does not update the BatchNorm running statistics."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )
        shape = (8, 8)
        model.decoder = _BatchNormDecoder(
            channel=1,
            scale=2.0,
            data_space_in=DataSpace(channels=2, name="combined", shape=shape),
            data_space_out=DataSpace(channels=1, name="sic", shape=shape),
        )
        model.train()
        processor = model.processor
        assert isinstance(processor, _PredictionProcessor)
        processor.sample_mask = torch.tensor([False, False])
        decoder = model.decoder
        assert isinstance(decoder, _BatchNormDecoder)
        norm = decoder.norm
        assert norm.running_mean is not None
        assert norm.running_var is not None
        assert norm.num_batches_tracked is not None
        running_mean = norm.running_mean.clone()
        running_var = norm.running_var.clone()

        _training_step(model, _oracle_batch())

        assert decoder.training
        torch.testing.assert_close(norm.running_mean, running_mean)
        torch.testing.assert_close(norm.running_var, running_var)
        assert norm.num_batches_tracked.item() == 0

    def test_rejects_prediction_longer_than_forecast(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss,
            cfg_metrics,
            {"reconstruction": 0.0, "prediction": 1.0},
            n_steps=2,
            n_forecast_steps=1,
        )

        with pytest.raises(ValueError, match="between 1 and 1"):
            _training_step(model, _oracle_batch(n_forecast_steps=1))

    def test_gradient_reaches_processor_decoder_and_encoders(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """All the loss comes from the prediction loss, which trains every part."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )

        result, _ = _training_step(model, _oracle_batch())
        result.loss.backward()

        processor, decoder, encoder = model.processor, model.decoder, model.encoders[0]
        assert isinstance(processor, _PredictionProcessor)
        assert isinstance(decoder, _ScaledDecoder)
        assert isinstance(encoder, _ScaledEncoder)
        for parameter in (processor.gain, decoder.scale, encoder.scale):
            assert parameter.grad is not None
            assert parameter.grad.abs() > 0

    @pytest.mark.parametrize(
        ("test_weights", "expected_frozen", "expected_warning"),
        [
            ({"reconstruction": 0.0, "prediction": 0.0}, True, True),
            ({"reconstruction": 0.0, "prediction": 1.0}, False, True),
            ({"reconstruction": 1.0, "prediction": 0.0}, False, False),
        ],
        ids=["both-zero", "prediction-only", "reconstruction-only"],
    )
    def test_freezes_decoder_and_warns(
        self,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        caplog: pytest.LogCaptureFixture,
        test_weights: dict[str, float],
        *,
        expected_frozen: bool,
        expected_warning: bool,
    ) -> None:
        """Freeze the decoder only if both weights are 0.

        Warn if the reconstruction weight is 0 while the encoders or decoder are trainable.
        The encoders are always trainable here.
        """
        with caplog.at_level(logging.WARNING):
            model = self._model(cfg_loss, cfg_metrics, test_weights)

        assert model.decoder.frozen is expected_frozen
        assert model.decoder.training is not expected_frozen
        assert ("latent space can drift" in caplog.text) is expected_warning

    def test_rejects_prediction_without_gradients(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights={"prediction": 1.0},
        )

        with pytest.raises(ValueError, match="returns a prediction without gradients"):
            model.training_step(_oracle_batch(), 0)


class TestSkipAnchor:
    """`skip_anchor` sets the skip-connection anchor wherever the latent rollout decodes.

    The decoder adds 0.1 * the target channel of each latent step to its anchor, so:
        initial   ->  output_k = clamp(observation + delta_k)
        previous  ->  output_k = clamp(output_{k-1} + delta_k)
    """

    SCALE = 0.1

    def _model(
        self,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        skip_anchor: str | None,
        **kwargs: Any,
    ) -> EncodeProcessDecode:
        kwargs.setdefault(
            "processor",
            DictConfig({"_target_": f"{__name__}.{_RepeatLastProcessor.__name__}"}),
        )
        return _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            decoder_scale=self.SCALE,
            skip_connection={"method": "additive"},
            skip_anchor=skip_anchor,
            **kwargs,
        )

    def _expected(
        self, observation: TensorNTCHW, deltas: TensorNTCHW, skip_anchor: str | None
    ) -> TensorNTCHW:
        """Apply the additive skip connection by hand for each forecast step."""
        anchor = observation[:, -1, 1:2]
        outputs = []
        for delta in (self.SCALE * deltas[:, :, 1:2]).unbind(dim=1):
            output = (anchor + delta).clamp(0.0, 1.0)
            outputs.append(output)
            if skip_anchor == "previous":
                anchor = output
        return torch.stack(outputs, dim=1)

    @pytest.mark.parametrize("skip_anchor", [None, "initial", "previous"])
    def test_training_step_matches_forward(
        self,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        skip_anchor: str | None,
    ) -> None:
        """The latent-loss metrics decode uses the same anchor as the forward pass."""
        model = self._model(cfg_loss, cfg_metrics, skip_anchor)
        model.train()
        batch = _oracle_batch()
        # The processor repeats the last history latent at every forecast step
        expected = self._expected(
            batch["sic"],
            batch["sic"][:, -1:].repeat(1, model.n_forecast_steps, 1, 1, 1),
            skip_anchor,
        )

        result, _ = _training_step(model, batch)
        model.eval()
        with torch.no_grad():
            forward = model(batch)

        torch.testing.assert_close(result.prediction, expected)
        torch.testing.assert_close(forward, expected)

    @pytest.mark.parametrize("skip_anchor", ["initial", "previous"])
    def test_prediction_loss_uses_skip_anchor(
        self,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        skip_anchor: str,
    ) -> None:
        """The prediction loss decodes the processor output with the same anchor."""
        model = self._model(
            cfg_loss,
            cfg_metrics,
            skip_anchor,
            loss_weights={"reconstruction": 0.0, "prediction": 1.0},
            processor=DictConfig(
                {
                    "_target_": f"{__name__}.{_PredictionProcessor.__name__}",
                    "use_autoregressive": False,
                }
            ),
        )
        model.train()
        batch = _oracle_batch()
        # The processor returns the last n_forecast_steps history latents
        expected = model.loss(
            self._expected(batch["sic"], batch["sic"], skip_anchor), batch["target"]
        )

        _, logged = _training_step(model, batch)

        torch.testing.assert_close(logged["train_prediction_loss"], expected)

    @pytest.mark.parametrize(
        ("rollout_space", "expected"),
        [
            (RolloutSpace.LATENT, SkipAnchorType.INITIAL),
            (RolloutSpace.PHYSICAL, SkipAnchorType.PREVIOUS),
        ],
        ids=["latent", "physical"],
    )
    def test_checkpoint_without_skip_anchor_keeps_old_behaviour(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_processor: DictConfig,
        tmp_path: Path,
        rollout_space: RolloutSpace,
        expected: SkipAnchorType,
    ) -> None:
        """Checkpoints saved before `skip_anchor` and `relaxation` existed still load."""
        model = EncodeProcessDecode(
            name="old-hparams",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere=Hemisphere.NORTH,
            input_spaces=[cfg_input_space],
            loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
            metrics=[],
            n_forecast_steps=1,
            n_history_steps=1,
            # A physical rollout needs the prediction target to be a model input
            output_space={**cfg_output_space, "name": cfg_input_space["name"]},
            optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            rollout_space=rollout_space,
            target_variable_indices=[0],
        )
        # Leave out the keys, as in a checkpoint saved before the options existed
        new_keys = {"skip_anchor", "relaxation"}
        assert all(model.hparams[key] is None for key in new_keys)
        checkpoint_file = tmp_path / "model.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": {
                    k: v for k, v in model.hparams.items() if k not in new_keys
                },
                "pytorch-lightning_version": lightning.__version__,
            },
            checkpoint_file,
        )

        loaded = EncodeProcessDecode.load_from_checkpoint(
            checkpoint_file,
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            metrics=[],
            weights_only=False,
        )
        assert loaded.rollout_strategy.space is rollout_space
        assert loaded.rollout_strategy.anchor is expected
        assert loaded.rollout_strategy.relaxation is None


class TestRelaxationInLatentLossTraining:
    """Relaxation applies wherever the latent-loss training step decodes a forecast.

    The decoder adds 0.1 * the target channel of each latent step to the previous
    output, then each step is relaxed towards a constant climatology c:
        output_k = lerp(clamp(output_{k-1} + delta_k), c, rate)
    The reconstruction loss decodes observations, not a forecast, so it is not relaxed.
    """

    SCALE = 0.1
    CLIMATOLOGY = 0.2
    RELAXATION: ClassVar[dict[str, Any]] = {"tau": 2.0}

    def _model(
        self,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
        **kwargs: Any,
    ) -> EncodeProcessDecode:
        kwargs.setdefault(
            "processor",
            DictConfig({"_target_": f"{__name__}.{_RepeatLastProcessor.__name__}"}),
        )
        kwargs.setdefault("relaxation", self.RELAXATION)
        return _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            decoder_scale=self.SCALE,
            skip_connection={"method": "additive"},
            skip_anchor="previous",
            **kwargs,
        )

    def _batch(self) -> dict[str, TensorNTCHW]:
        batch = _oracle_batch()
        batch["climatology"] = torch.full_like(batch["target"], self.CLIMATOLOGY)
        return batch

    def _expected(
        self,
        observation: TensorNTCHW,
        deltas: TensorNTCHW,
        rate: float | None,
    ) -> TensorNTCHW:
        """Apply the skip connection, and the relaxation if `rate` is set, by hand."""
        anchor = observation[:, -1, 1:2]
        outputs = []
        for delta in (self.SCALE * deltas[:, :, 1:2]).unbind(dim=1):
            output = (anchor + delta).clamp(0.0, 1.0)
            if rate is not None:
                output = output + rate * (self.CLIMATOLOGY - output)
            outputs.append(output)
            anchor = output
        return torch.stack(outputs, dim=1)

    def test_metrics_decode_matches_forward(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(cfg_loss, cfg_metrics)
        assert model.relaxation is not None
        batch = self._batch()
        expected = self._expected(
            batch["sic"],
            batch["sic"][:, -1:].repeat(1, model.n_forecast_steps, 1, 1, 1),
            model.relaxation.rate,
        )

        model.train()
        result, _ = _training_step(model, batch)
        model.eval()
        with torch.no_grad():
            forward = model(batch)

        torch.testing.assert_close(result.prediction, expected)
        torch.testing.assert_close(forward, expected)

    def test_prediction_loss_is_relaxed(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss,
            cfg_metrics,
            loss_weights={"reconstruction": 0.0, "prediction": 1.0},
            processor=DictConfig(
                {
                    "_target_": f"{__name__}.{_PredictionProcessor.__name__}",
                    "use_autoregressive": False,
                }
            ),
        )
        assert model.relaxation is not None
        model.train()
        batch = self._batch()
        # The processor returns the last n_forecast_steps history latents
        expected = model.loss(
            self._expected(batch["sic"], batch["sic"], model.relaxation.rate),
            batch["target"],
        )

        _, logged = _training_step(model, batch)

        torch.testing.assert_close(logged["train_prediction_loss"], expected)

    def test_reconstruction_loss_is_not_relaxed(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(cfg_loss, cfg_metrics, loss_weights={"reconstruction": 1.0})
        model.train()
        batch = self._batch()
        expected = model.loss(
            self._expected(batch["sic"], batch["future_sic"], None), batch["target"]
        )

        _, logged = _training_step(model, batch)

        torch.testing.assert_close(logged["train_reconstruction_loss"], expected)

    def test_off_in_training_ignores_climatology(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss,
            cfg_metrics,
            relaxation={**self.RELAXATION, "apply_in_training": False},
        )
        model.train()
        batch = _oracle_batch()  # no climatology entry
        expected = self._expected(
            batch["sic"],
            batch["sic"][:, -1:].repeat(1, model.n_forecast_steps, 1, 1, 1),
            None,
        )

        result, _ = _training_step(model, batch)

        torch.testing.assert_close(result.prediction, expected)
