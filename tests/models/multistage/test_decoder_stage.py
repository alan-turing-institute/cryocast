import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models.multistage import DecoderStage, EncoderStage
from icenet_mp.types import DataSpace


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

    def test_process_batch_extracts_expected_timesteps(
        self,
        encoder_stage: EncoderStage,
        *,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        # Predict only channel 2 of a 3-channel target group, so that the selection of
        # target variables is visible in the processed batch
        decoder_stage = DecoderStage.from_template(
            decoder=cfg_decoder,
            encoders=[encoder_stage],
            output_space=DataSpace.from_dict(cfg_output_space),
            target_dataset_name="target",
            target_variable_indices=[2],
        )
        # t=-2 feeds the encoders and the persistence skip connection; t=-1 is the
        # forecast target. Using three history steps makes the -2/-1 split unambiguous.
        batch_size = 2
        n_history_steps = 3
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

        assert torch.equal(processed["test-input"], test_input[:, -2].unsqueeze(1))
        assert torch.equal(processed["target"], target[:, -1, [2], :, :].unsqueeze(1))
        assert torch.equal(
            processed["persistence"], target[:, -2, [2], :, :].unsqueeze(1)
        )

    @pytest.mark.parametrize(
        ("n_history_steps", "target_variable_indices", "expected_message"),
        [
            (1, [0], "at least two history steps"),
            (2, [0, 1], "target_variable_indices selects"),
        ],
        ids=["one-history-step", "indices-channel-mismatch"],
    )
    def test_rejects_invalid_configuration(
        self,
        encoder_stage: EncoderStage,
        *,
        n_history_steps: int,
        target_variable_indices: list[int],
        expected_message: str,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> None:
        with pytest.raises(ValueError, match=expected_message):
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
        cfg_metrics: list[str],
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
        assert set(decoder_stage.validation_metrics.keys()) == set(cfg_metrics)
        assert decoder_stage.name == "target_decoder"
