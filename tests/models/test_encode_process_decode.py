import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import EncodeProcessDecode


@pytest.mark.parametrize("test_n_forecast_steps", [1, 2, 5])
@pytest.mark.parametrize("test_n_history_steps", [1, 2, 5])
class TestEncodeProcessDecode:
    def test_init(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
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
        assert model.input_spaces[0].channels == cfg_input_space["channels"]
        assert model.input_spaces[0].name == cfg_input_space["name"]
        assert model.input_spaces[0].shape == cfg_input_space["shape"]
        assert model.n_forecast_steps == test_n_forecast_steps
        assert model.n_history_steps == test_n_history_steps
        assert model.output_space.channels == cfg_output_space["channels"]
        assert model.output_space.name == cfg_output_space["name"]
        assert model.output_space.shape == cfg_output_space["shape"]

    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    def test_forward(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
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


class TestEncodeProcessDecodeMultistageOnly:
    @pytest.mark.parametrize(
        ("computes_loss_in_latent_space", "expected_multistage_only"),
        [(False, False), (True, True)],
        ids=["physical-loss", "latent-loss"],
    )
    def test_multistage_only_follows_processor_loss_space(
        self,
        *,
        computes_loss_in_latent_space: bool,
        expected_multistage_only: bool,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> None:
        """Only a processor that computes its loss in latent space needs multistage."""
        processor = DictConfig(
            {
                **cfg_processor,
                "computes_loss_in_latent_space": computes_loss_in_latent_space,
            }
        )
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=processor,
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
