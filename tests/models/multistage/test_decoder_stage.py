import logging
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cryocast.models.multistage import DecoderStage, EncoderStage
from cryocast.types import DataSpace


class TestDecoderStage:
    def test_forward_shape(
        self,
        decoder_stage: DecoderStage,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        batch_size = 2
        result = decoder_stage(
            {
                "test-input": torch.rand(
                    batch_size,
                    1,
                    cfg_input_space["channels"],
                    *cfg_input_space["shape"],
                ),
                "persistence": torch.rand(
                    batch_size,
                    1,
                    cfg_output_space["channels"],
                    *cfg_output_space["shape"],
                ),
            }
        )
        assert result.shape == (
            batch_size,
            1,
            cfg_output_space["channels"],
            *cfg_output_space["shape"],
        )

    @pytest.mark.parametrize(
        "skip_connection",
        [None, {"method": "additive"}],
        ids=["no-skip", "additive-skip"],
    )
    @pytest.mark.parametrize("n_history_steps", [2, 3], ids=["2-steps", "3-steps"])
    def test_process_batch_aligns_timesteps(
        self,
        encoder_stage: EncoderStage,
        *,
        skip_connection: dict[str, str] | None,
        n_history_steps: int,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        # Predict only channel 2 of a 3-channel target group, so that the selection of
        # target variables is visible in the processed batch.
        decoder_stage = DecoderStage.from_template(
            decoder=DictConfig(
                {
                    "_target_": "cryocast.models.decoders.NaiveLinearDecoder",
                    "skip_connection": skip_connection,
                }
            ),
            encoders=[encoder_stage],
            output_space=DataSpace.from_dict(cfg_output_space),
            target_dataset_name="target",
            target_variable_indices=[2],
        )
        batch_size = 2
        test_input = torch.rand(
            batch_size,
            n_history_steps,
            cfg_input_space["channels"],
            *cfg_input_space["shape"],
        )
        target = torch.rand(batch_size, n_history_steps, 3, *cfg_output_space["shape"])

        processed = decoder_stage.process_batch(
            {"test-input": test_input, "target": target}
        )

        # The latent input and physical target must represent the same timestamp. If the
        # decoder is trained on t=-2 -> t=-1 instead, a future latent predicted by the
        # processor is advanced by the decoder a second time (issue #562). Persistence
        # comes from the previous state (t=-2) whether or not there is a skip connection,
        # so that a residual decoder does not learn that the increment is always zero.
        assert set(processed) == {"test-input", "target", "persistence"}
        assert torch.equal(processed["test-input"], test_input[:, -1].unsqueeze(1))
        assert torch.equal(processed["target"], target[:, -1, [2], :, :].unsqueeze(1))
        assert torch.equal(
            processed["persistence"], target[:, -2, [2], :, :].unsqueeze(1)
        )

        # The processed batch must be directly usable by forward()
        processed.pop("target")
        assert decoder_stage(processed).shape == (
            batch_size,
            1,
            cfg_output_space["channels"],
            *cfg_output_space["shape"],
        )

    def test_multiple_encoders_are_combined(
        self,
        encoder_stage: EncoderStage,
        second_encoder_stage: EncoderStage,
        *,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        decoder_stage = DecoderStage.from_template(
            decoder=cfg_decoder,
            encoders=[encoder_stage, second_encoder_stage],
            output_space=DataSpace.from_dict(cfg_output_space),
            target_dataset_name="target",
            target_variable_indices=[0],
        )
        # The decoder reads from the concatenation of all encoder latent spaces
        assert decoder_stage.encoder_names == ["test-input", "second-input"]
        assert decoder_stage.decoder.data_space_in.channels == (
            encoder_stage.encoder.data_space_out.channels
            + second_encoder_stage.encoder.data_space_out.channels
        )

        batch_size = 2
        n_history_steps = 2
        batch = {
            "test-input": torch.rand(
                batch_size,
                n_history_steps,
                cfg_input_space["channels"],
                *cfg_input_space["shape"],
            ),
            "second-input": torch.rand(
                batch_size,
                n_history_steps,
                second_encoder_stage.encoder.data_space_in.channels,
                *second_encoder_stage.encoder.data_space_in.shape,
            ),
            "target": torch.rand(
                batch_size,
                n_history_steps,
                cfg_output_space["channels"],
                *cfg_output_space["shape"],
            ),
        }

        processed = decoder_stage.process_batch(batch)

        # Every encoder input is sliced at the target timestamp
        for name in decoder_stage.encoder_names:
            assert torch.equal(processed[name], batch[name][:, -1].unsqueeze(1))
        processed.pop("target")
        assert decoder_stage(processed).shape == (
            batch_size,
            1,
            cfg_output_space["channels"],
            *cfg_output_space["shape"],
        )

    @pytest.mark.parametrize(
        ("n_history_steps", "target_variable_indices", "match"),
        [
            (1, [0], "at least two history steps"),
            (2, [0, 1], "target_variable_indices selects"),
        ],
        ids=["single-history-step", "variable-indices-channel-mismatch"],
    )
    def test_constructor_rejects_invalid_arguments(
        self,
        encoder_stage: EncoderStage,
        *,
        n_history_steps: int,
        target_variable_indices: list[int],
        match: str,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        with pytest.raises(ValueError, match=match):
            DecoderStage(
                decoder=cfg_decoder,
                encoders=[encoder_stage],
                target_dataset_name="target",
                target_variable_indices=target_variable_indices,
                hemisphere="north",
                input_spaces=[cfg_input_space],
                n_forecast_steps=1,
                n_history_steps=n_history_steps,
                name="test-target_decoder",
                optimizer=cfg_optimizer,
                output_space=cfg_output_space,
                scheduler=cfg_scheduler,
                lr_scheduler=cfg_lr_scheduler,
                loss=cfg_loss,
                metrics=cfg_metrics,
            )

    def test_encoder_parameters_are_frozen(self, decoder_stage: DecoderStage) -> None:
        assert all(
            not param.requires_grad
            for encoder in decoder_stage.encoders
            for param in encoder.parameters()
        )

    def test_train_keeps_frozen_encoders_in_eval_mode(
        self, decoder_stage: DecoderStage
    ) -> None:
        decoder_stage.eval()
        for encoder in decoder_stage.encoders:
            encoder.train()

        decoder_stage.train()

        assert decoder_stage.training
        assert all(not encoder.training for encoder in decoder_stage.encoders)

    def test_from_template_builds_decoder_stage_from_encoder_stages(
        self,
        encoder_stage: EncoderStage,
        *,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        decoder_stage = DecoderStage.from_template(
            decoder=cfg_decoder,
            encoders=[encoder_stage],
            output_space=DataSpace.from_dict(cfg_output_space),
            target_dataset_name="target",
            target_variable_indices=[0],
        )

        assert decoder_stage.hemisphere == encoder_stage.hemisphere
        assert decoder_stage.n_forecast_steps == encoder_stage.n_forecast_steps
        assert decoder_stage.n_history_steps == encoder_stage.n_history_steps
        assert [s.to_dict() for s in decoder_stage.input_spaces] == [cfg_input_space]
        assert decoder_stage.output_space.to_dict() == cfg_output_space
        assert decoder_stage.encoder_names == ["test-input"]
        # Single-channel metrics skipped by the multi-channel encoder stage are restored
        assert set(decoder_stage.validation_metrics.keys()) == {
            spec["name"] for spec in cfg_metrics
        }
        assert decoder_stage.name == "target_decoder"

    # Parametrizing `cfg_loss` overrides the shared fixture of that name, which the
    # `decoder_stage` fixture consumes even though this test does not request it. The
    # warning is logged while that fixture builds the stage, so it is read from the
    # "setup" phase records: if the stage is ever built in the test body instead,
    # switch to `caplog.records`.
    @pytest.mark.usefixtures("decoder_stage")
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
            and "has no effect on DecoderStage" in r.getMessage()
        ]
        assert bool(records) is expect_warning
