from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from cryocast.losses import LeadTimeWeightedLoss
from cryocast.models import EncodeProcessDecode
from cryocast.models.decoders import BaseDecoder
from cryocast.models.encoders import BaseEncoder
from cryocast.models.processors import BaseProcessor
from cryocast.types import DataSpace, ProcessorOutput, TensorNCHW, TensorNTCHW


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
    """Processor that forecasts the target latent it is given, i.e. a perfect forecast."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(computes_loss_in_latent_space=True, **kwargs)
        self.target: TensorNTCHW | None = None

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        assert y is not None
        assert self.target_channel_offset is not None
        self.target = y
        start = self.target_channel_offset
        end = start + self.data_space_target.channels
        prediction = x[:, -1:].expand(-1, self.n_forecast_steps, -1, -1, -1).clone()
        prediction[:, :, start:end] = y
        return ProcessorOutput(prediction=prediction, loss=torch.zeros(()))


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

    def test_latent_target_uses_target_input_encoder(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        """Latent supervision uses the same target-dataset encoder as history."""
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
            name="shared-target-latent",
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

        target_input_encoder = model.encoders[0]
        assert model.target_input_encoder is target_input_encoder
        assert model.processor.data_space_target == target_input_encoder.data_space_out
        assert model.processor.data_space_target != model.target_encoder.data_space_out

        history = torch.rand(2, 2, cfg_input_space["channels"], 16, 16)
        target = torch.rand(2, 2, 1, 16, 16)
        full_target = history[:, -1:].expand(-1, 2, -1, -1, -1).clone()
        full_target[:, :, 2:3] = target

        expected = target_input_encoder.rollout(full_target)
        actual = model._encode_target_latent({cfg_input_space["name"]: history}, target)

        torch.testing.assert_close(actual, expected)

        # The latent loss must not train the shared encoder through the target latent
        assert all(p.requires_grad for p in target_input_encoder.parameters())
        assert not actual.requires_grad

    def test_perfect_latent_forecast_decodes_to_target(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """A perfect latent forecast decodes to the target, even with mismatched encoders.

        The input and target encoders map `x` to `x` and `2x` respectively.
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
        batch = {
            "era5": torch.rand(2, 2, 1, *shape),
            "sic": torch.rand(2, 2, 2, *shape),
            "target": torch.rand(2, 2, 1, *shape),
        }

        result = model.training_step(batch, 0)

        mae = (result.prediction - result.target).abs().mean()
        assert mae.item() == pytest.approx(0.0, abs=1e-6)

        # The latent loss must not train the shared encoder through the target latent
        processor = model.processor
        assert isinstance(processor, _OracleProcessor)
        assert processor.target is not None
        assert all(p.requires_grad for e in model.encoders for p in e.parameters())
        assert not processor.target.requires_grad

        # The unused target encoder must be frozen (e.g. so that DDP does not fail)
        assert not any(p.requires_grad for p in model.target_encoder.parameters())
