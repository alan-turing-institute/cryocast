from pathlib import Path
from typing import Any

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
        ("test_computes_loss_in_latent_space", "expected_multistage_only"),
        [(False, False), (True, True)],
        ids=["default", "latent-loss"],
    )
    def test_init_multistage_only(
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
        expected_multistage_only: bool,
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
        assert model.multistage_only is expected_multistage_only
        assert model.needs_future_inputs is expected_multistage_only

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
        assert model.multistage_only

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
        """A perfect latent forecast decodes to the target.

        The input encoders map `x` to `x`. The unused target encoder maps `x` to `2x`,
        so the forecast would be wrong if the latent target were built with it.
        """
        shape = (8, 8)
        era5 = DataSpace(channels=1, name="era5", shape=shape)
        sic = DataSpace(channels=2, name="sic", shape=shape)
        output_space = DataSpace(channels=1, name="sic", shape=shape)
        model = EncodeProcessDecode(
            name="oracle",
            encoders=[
                _ScaledEncoder(data_space_in=era5, scale=1.0),
                _ScaledEncoder(data_space_in=sic, scale=1.0),
                _ScaledEncoder(
                    data_space_in=DataSpace(channels=1, name="target", shape=shape),
                    scale=2.0,
                ),
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
