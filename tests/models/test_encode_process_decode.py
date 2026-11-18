import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
    ProcessorOutput,
    ProcessorOutputDiffusion,
    RolloutSpace,
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
    autoregressive mode, or n_forecast_steps steps in parallel mode. Set `sample_mask`
    to choose the samples in `sample_mask` (default: all samples).
    """

    def __init__(self, *, use_autoregressive: bool = True, **kwargs: Any) -> None:
        super().__init__(computes_loss_in_latent_space=True, **kwargs)
        self.use_autoregressive = use_autoregressive
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.sample_mask: torch.Tensor | None = None

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        assert y is not None
        n_steps = 1 if self.use_autoregressive else self.n_forecast_steps
        mask = (
            torch.ones(x.shape[0], dtype=torch.bool)
            if self.sample_mask is None
            else self.sample_mask
        )
        return ProcessorOutputDiffusion(
            prediction=y.clone(),
            loss=torch.zeros(()),
            differentiable_prediction=self.gain * x[:, -n_steps:],
            sample_mask=mask,
        )


class _ScaledDecoder(BaseDecoder):
    """Trainable decoder that multiplies one channel of the combined latent by scale."""

    def __init__(self, *, channel: int, scale: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.channel = channel
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return x[:, self.channel : self.channel + 1] * self.scale


def _oracle_model(
    *,
    cfg_loss: DictConfig,
    cfg_metrics: list[dict[str, Any]],
    loss_weights: dict[str, float] | None = None,
    processor: DictConfig | None = None,
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
            scale=2.0,
            data_space_in=DataSpace(channels=2, name="combined", shape=shape),
            data_space_out=output_space,
        ),
        loss_weights=loss_weights,
        hemisphere="north",
        input_spaces=[sic.to_dict()],
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


def _oracle_batch() -> dict[str, TensorNTCHW]:
    """Build a batch for the model from `_oracle_model`."""
    future_sic = torch.rand(2, 2, 2, 8, 8)
    return {
        "sic": torch.rand(2, 2, 2, 8, 8),
        "future_sic": future_sic,
        "target": future_sic[:, :, 1:2].clone(),
    }


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
            (None, {"reconstruction": 1.0, "prediction": 0.0}),
            ({}, {"reconstruction": 1.0, "prediction": 0.0}),
            ({"reconstruction": 0}, {"reconstruction": 0.0, "prediction": 0.0}),
            ({"prediction": 0.5}, {"reconstruction": 1.0, "prediction": 0.5}),
        ],
        ids=["none", "empty", "reconstruction-zero", "prediction-half"],
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

    def test_reconstruction_loss_zero_weight_matches_main(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """Weight 0 gives the behaviour from before the reconstruction loss.

        The decoder is frozen and stays in eval mode, and the loss is only the latent
        loss, which does not go through the encoders or the decoder.
        """
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights={"reconstruction": 0.0},
        )
        model.train()

        with pytest.MonkeyPatch.context() as mp:
            mock_log = MagicMock()
            mp.setattr(model, "log", mock_log)
            result = model.training_step(_oracle_batch(), 0)

        # Only the latent loss from _OracleProcessor (zero) remains
        assert result.loss.item() == 0.0
        assert not result.loss.requires_grad
        assert [c.args[0] for c in mock_log.call_args_list] == ["train_loss"]

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

        with pytest.MonkeyPatch.context() as mp:
            mock_log = MagicMock()
            mp.setattr(model, "log", mock_log)
            result = model.training_step(batch, 0)
        result.loss.backward()

        torch.testing.assert_close(result.loss, 0.5 * reconstruction_loss)
        logged = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
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
        assert loaded.rollout_space is RolloutSpace.LATENT

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
    ) -> EncodeProcessDecode:
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights=loss_weights,
            processor=DictConfig(
                {
                    "_target_": f"{__name__}.{_PredictionProcessor.__name__}",
                    "use_autoregressive": use_autoregressive,
                }
            ),
        )
        model.train()
        return model

    @staticmethod
    def _training_step(
        model: EncodeProcessDecode, batch: dict[str, TensorNTCHW]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run a training step and return the loss and the logged values."""
        with pytest.MonkeyPatch.context() as mp:
            mock_log = MagicMock()
            mp.setattr(model, "log", mock_log)
            result = model.training_step(batch, 0)
        return result.loss, {c.args[0]: c.args[1] for c in mock_log.call_args_list}

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

        loss, logged = self._training_step(model, batch)

        torch.testing.assert_close(logged["train_prediction_loss"], expected)
        torch.testing.assert_close(loss, 0.5 * expected)

        # Changing the later forecast steps of the target does not change the loss
        batch["target"][:, 1:] += 1.0
        loss_changed_target, _ = self._training_step(model, batch)
        torch.testing.assert_close(loss_changed_target, loss)

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

        loss, logged = self._training_step(model, batch)

        torch.testing.assert_close(logged["train_prediction_loss"], expected)
        torch.testing.assert_close(loss, 0.5 * expected)

    def test_zero_weight_gives_same_loss_as_before(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """With weight 0, the loss is the latent loss plus the reconstruction loss only."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.5, "prediction": 0.0}
        )
        batch = _oracle_batch()
        reconstruction_loss = model.loss(2.0 * batch["target"], batch["target"])

        loss, logged = self._training_step(model, batch)

        torch.testing.assert_close(loss, 0.5 * reconstruction_loss)
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

        loss, _ = self._training_step(model, batch)

        torch.testing.assert_close(loss, expected)

    def test_empty_mask_gives_zero_not_nan(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )
        processor = model.processor
        assert isinstance(processor, _PredictionProcessor)
        processor.sample_mask = torch.tensor([False, False])

        loss, logged = self._training_step(model, _oracle_batch())
        loss.backward()

        assert logged["train_prediction_loss"].item() == 0.0
        assert loss.item() == 0.0
        decoder = model.decoder
        assert isinstance(decoder, _ScaledDecoder)
        assert decoder.scale.grad is not None
        assert decoder.scale.grad.item() == 0.0

    def test_gradient_reaches_processor_decoder_and_encoders(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """All the loss comes from the prediction loss, which trains every part."""
        model = self._model(
            cfg_loss, cfg_metrics, {"reconstruction": 0.0, "prediction": 1.0}
        )

        loss, _ = self._training_step(model, _oracle_batch())
        loss.backward()

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

    def test_rejects_processor_without_differentiable_prediction(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = _oracle_model(
            cfg_loss=cfg_loss,
            cfg_metrics=cfg_metrics,
            loss_weights={"prediction": 1.0},
        )

        with pytest.raises(
            TypeError, match="does not return a differentiable prediction"
        ):
            model.training_step(_oracle_batch(), 0)
