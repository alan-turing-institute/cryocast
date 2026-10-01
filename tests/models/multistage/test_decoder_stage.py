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
        decoder_stage: DecoderStage,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
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
        target = torch.rand(
            batch_size,
            n_history_steps,
            cfg_output_space["channels"],
            *cfg_output_space["shape"],
        )

        processed = decoder_stage.process_batch(
            {"test-input": test_input, "target": target}
        )

        assert torch.equal(processed["test-input"], test_input[:, -2].unsqueeze(1))
        assert torch.equal(processed["target"], target[:, -1, [0], :, :].unsqueeze(1))
        assert torch.equal(
            processed["persistence"], target[:, -2, [0], :, :].unsqueeze(1)
        )

    def test_requires_at_least_two_history_steps(
        self,
        encoder_stage: EncoderStage,
        *,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> None:
        with pytest.raises(ValueError, match="at least two history steps"):
            DecoderStage(
                decoder=cfg_decoder,
                encoders=[encoder_stage],
                target_dataset_name="target",
                target_variable_indices=[0],
                hemisphere="north",
                input_spaces=[cfg_input_space],
                n_forecast_steps=1,
                n_history_steps=1,
                name="test-target_decoder",
                optimizer=cfg_optimizer,
                output_space=cfg_output_space,
                scheduler=cfg_scheduler,
                lr_scheduler=cfg_lr_scheduler,
                loss=cfg_loss,
                metrics=cfg_metrics,
            )

    def test_variable_indices_channel_mismatch_raises(
        self,
        encoder_stage: EncoderStage,
        *,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> None:
        with pytest.raises(ValueError, match="target_variable_indices selects"):
            DecoderStage(
                decoder=cfg_decoder,
                encoders=[encoder_stage],
                target_dataset_name="target",
                target_variable_indices=[0, 1],
                hemisphere="north",
                input_spaces=[cfg_input_space],
                n_forecast_steps=1,
                n_history_steps=2,
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
        *,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> None:
        # from_template reads n_history_steps/n_forecast_steps/etc. off the source
        # EncoderStage, so it must be built with two history steps here (DecoderStage
        # itself requires at least two).
        source_encoder_stage = EncoderStage(
            channel_names=["channel-0", "channel-1", "channel-2", "channel-3"],
            data_space_in=DataSpace.from_dict(cfg_input_space),
            encoder=cfg_encoders["test-input"],
            decoder=cfg_decoder,
            latent_space=cfg_encoders["latent_space"],
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=2,
            name="test-input_encoder",
            optimizer=cfg_optimizer,
            output_space=cfg_output_space,
            scheduler=cfg_scheduler,
            lr_scheduler=cfg_lr_scheduler,
            loss=cfg_loss,
            metrics=cfg_metrics,
        )

        decoder_stage = DecoderStage.from_template(
            decoder=cfg_decoder,
            encoders=[source_encoder_stage],
            target_dataset_name="target",
            target_variable_indices=[0],
        )

        assert decoder_stage.hemisphere == source_encoder_stage.hemisphere
        assert decoder_stage.n_forecast_steps == source_encoder_stage.n_forecast_steps
        assert decoder_stage.n_history_steps == source_encoder_stage.n_history_steps
        assert decoder_stage.encoder_names == ["test-input"]
        assert decoder_stage.name == "target_decoder"
