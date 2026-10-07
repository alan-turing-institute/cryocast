from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from cryocast.data.combined_dataset import forecast_input_key
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
        del x
        assert y is not None
        self.target = y
        return ProcessorOutput(prediction=y, loss=torch.zeros(()))


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

    def test_latent_target_encodes_complete_forecast_inputs(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        """Latent supervision uses every future input channel and the history encoder."""
        output_space = DictConfig(
            {
                "channels": 1,
                "name": cfg_input_space["name"],
                "shape": cfg_input_space["shape"],
            }
        )
        model = EncodeProcessDecode(
            name="full-forecast-latent",
            encoders=cfg_encoders,
            processor=DictConfig(
                {"_target_": f"{__name__}.{_OracleProcessor.__name__}"}
            ),
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

        encoder = model.encoders[0]
        future = torch.rand(2, 2, cfg_input_space["channels"], 16, 16)
        batch = {forecast_input_key(cfg_input_space["name"]): future}

        expected = encoder.rollout(future).detach()
        actual = model._encode_forecast_inputs(batch)

        assert model.processor.data_space_target == model.processor.data_space
        torch.testing.assert_close(actual, expected)
        assert actual.shape[2] == model.processor.data_space.channels
        assert not actual.requires_grad

    def test_perfect_full_latent_forecast_decodes_to_target(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """A perfect combined-latent forecast decodes to the physical target."""
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
        future_era5 = torch.rand(2, 2, 1, *shape)
        future_sic = torch.rand(2, 2, 2, *shape)
        target = future_sic[:, :, 1:2].clone()
        batch = {
            "era5": torch.rand(2, 2, 1, *shape),
            "sic": torch.rand(2, 2, 2, *shape),
            "target": target,
            forecast_input_key("era5"): future_era5,
            forecast_input_key("sic"): future_sic,
        }

        result = model.training_step(batch, 0)

        torch.testing.assert_close(result.prediction, result.target)
        processor = model.processor
        assert isinstance(processor, _OracleProcessor)
        assert processor.target is not None
        assert processor.target.shape[2] == 3
        assert not processor.target.requires_grad
