import logging
from pathlib import Path
from typing import Any

import lightning
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cryocast.models import EncodeProcessDecode
from cryocast.models.multistage import EncoderStage
from cryocast.types import DataSpace


class TestEncoderStage:
    def test_forward_ignores_skip_connection_persistence_requirement(
        self,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        encoder_stage = EncoderStage(
            channel_names=["channel-0", "channel-1", "channel-2", "channel-3"],
            data_space_in=DataSpace.from_dict(cfg_input_space),
            encoder=cfg_encoders["test-input"],
            decoder=DictConfig(
                {
                    "_target_": "cryocast.models.decoders.NaiveLinearDecoder",
                    "skip_connection": {"method": "additive"},
                }
            ),
            latent_space=cfg_encoders["latent_space"],
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            name="test-input_encoder",
            optimizer=cfg_optimizer,
            output_space=cfg_input_space,
            scheduler=cfg_scheduler,
            lr_scheduler=cfg_lr_scheduler,
            loss=cfg_loss,
            metrics=cfg_metrics,
        )

        assert encoder_stage.decoder.skip_connection is None

        batch_size = 2
        result = encoder_stage(
            {
                "target": torch.rand(
                    batch_size,
                    1,
                    cfg_input_space["channels"],
                    *cfg_input_space["shape"],
                )
            }
        )

        assert result.shape == (
            batch_size,
            1,
            cfg_input_space["channels"],
            *cfg_input_space["shape"],
        )

    def test_process_batch_extracts_first_timestep(
        self, encoder_stage: EncoderStage, cfg_input_space: DictConfig
    ) -> None:
        batch_size = 2
        n_history_steps = 3
        test_input = torch.rand(
            batch_size,
            n_history_steps,
            cfg_input_space["channels"],
            *cfg_input_space["shape"],
        )

        processed = encoder_stage.process_batch({"test-input": test_input})

        assert torch.equal(processed["target"], test_input[:, 0].unsqueeze(1))

    def test_single_channel_metrics_are_disabled_for_multi_channel_input(
        self, encoder_stage: EncoderStage
    ) -> None:
        assert set(encoder_stage.validation_metrics.keys()) == {"mae", "rmse", "ssim"}

    def test_checkpoint_loads_with_weights_only(
        self,
        encoder_stage: EncoderStage,
        cfg_metrics: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """The input data space is saved as a plain dict and restored on loading."""
        checkpoint_path = tmp_path / "encoder.ckpt"
        torch.save(
            {
                "hyper_parameters": dict(encoder_stage.hparams),
                "pytorch-lightning_version": lightning.__version__,
                "state_dict": encoder_stage.state_dict(),
            },
            checkpoint_path,
        )

        loaded = EncoderStage.load_from_checkpoint(
            checkpoint_path, metrics=cfg_metrics, weights_only=True
        )

        assert isinstance(encoder_stage.hparams["data_space_in"], dict)
        assert loaded.encoder.data_space_in == encoder_stage.encoder.data_space_in

    def test_dataset_name_returns_input_space_name(
        self, encoder_stage: EncoderStage
    ) -> None:
        assert encoder_stage.dataset_name == "test-input"

    def test_from_template_builds_encoder_stage_from_encode_process_decode(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        template = EncodeProcessDecode(
            name="template",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=2,
            n_history_steps=3,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            loss=cfg_loss,
            metrics=cfg_metrics,
            target_variable_indices=[0],
        )

        encoder_stage = EncoderStage.from_template(
            channel_names=["channel-0", "channel-1", "channel-2", "channel-3"],
            data_space_in=DataSpace.from_dict(cfg_input_space),
            dataset="test-input",
            decoder=cfg_decoder,
            encoder=cfg_encoders["test-input"],
            template=template,
        )

        assert encoder_stage.hemisphere == template.hemisphere
        assert encoder_stage.n_forecast_steps == template.n_forecast_steps
        assert encoder_stage.n_history_steps == template.n_history_steps
        assert (
            encoder_stage.encoder.data_space_out.shape
            == template.encoders[0].data_space_out.shape
        )
        assert encoder_stage.name == "test_input_encoder"
        # The encoder stage reconstructs its own input, not the forecast target
        assert [s.to_dict() for s in encoder_stage.input_spaces] == [cfg_input_space]
        assert encoder_stage.output_space.to_dict() == cfg_input_space

    # Parametrizing `cfg_loss` overrides the shared fixture of that name, which the
    # `encoder_stage` fixture consumes even though this test does not request it. The
    # warning is logged while that fixture builds the stage, so it is read from the
    # "setup" phase records: if the stage is ever built in the test body instead,
    # switch to `caplog.records`.
    @pytest.mark.usefixtures("encoder_stage")
    @pytest.mark.parametrize(
        ("cfg_loss", "expect_warning"),
        [
            (
                OmegaConf.create(
                    {"_target_": "torch.nn.HuberLoss", "lead_time_exponent": 2.0}
                ),
                True,
            ),
            (OmegaConf.create({"_target_": "torch.nn.HuberLoss"}), False),
        ],
        ids=["weighted", "unweighted"],
    )
    def test_warns_when_lead_time_weighting_ignored(
        self,
        caplog: pytest.LogCaptureFixture,
        *,
        expect_warning: bool,
    ) -> None:
        records = [
            r
            for r in caplog.get_records("setup")
            if r.levelno == logging.WARNING
            and "has no effect on EncoderStage" in r.getMessage()
        ]
        assert bool(records) is expect_warning
