from typing import Any

import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models.multistage import DecoderStage, EncoderStage, ProcessorStage
from icenet_mp.models.processors import BaseProcessor
from icenet_mp.types import DataSpace, ProcessorOutput, TensorNTCHW


class _FixedLossProcessor(BaseProcessor):
    """Test double whose rollout reports a fixed loss.

    Declares ``computes_loss_in_latent_space=True``, as any real processor returning a
    loss must: ``training_step`` only honours ``ProcessorOutput.loss`` for a processor
    that declares this, otherwise it goes through ``self(batch)`` and is silently
    dropped.
    """

    def __init__(self, *, loss: torch.Tensor, **kwargs: Any) -> None:
        kwargs.pop("computes_loss_in_latent_space", None)
        super().__init__(computes_loss_in_latent_space=True, **kwargs)
        self._loss = loss

    def rollout(
        self,
        x: TensorNTCHW,
        y: TensorNTCHW | None = None,  # noqa: ARG002
    ) -> ProcessorOutput:
        return ProcessorOutput(
            prediction=x[:, -self.n_forecast_steps :], loss=self._loss
        )


class TestProcessorStage:
    @pytest.fixture
    def target_encoder_stage(
        self,
        *,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
        cfg_decoder: DictConfig,
    ) -> EncoderStage:
        # The target encoder encodes the forecast target itself, not a raw input
        # dataset, so its data_space_in is built directly from the output space.
        target_space = DataSpace(
            channels=cfg_output_space["channels"],
            name="target",
            shape=cfg_output_space["shape"],
        )
        return EncoderStage(
            channel_names=["target-channel"],
            data_space_in=target_space,
            encoder=cfg_encoders["target"],
            decoder=cfg_decoder,
            latent_space=cfg_encoders["latent_space"],
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            name="target_encoder",
            optimizer=cfg_optimizer,
            output_space=cfg_output_space,
            scheduler=cfg_scheduler,
            lr_scheduler=cfg_lr_scheduler,
            loss=cfg_loss,
            metrics=cfg_metrics,
        )

    @pytest.fixture
    def processor_stage(
        self,
        decoder_stage: DecoderStage,
        target_encoder_stage: EncoderStage,
        *,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> ProcessorStage:
        return ProcessorStage(
            processor=cfg_processor,
            decoder_model=decoder_stage,
            target_encoder=target_encoder_stage,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=2,
            name="test-target_processor",
            optimizer=cfg_optimizer,
            output_space=cfg_output_space,
            scheduler=cfg_scheduler,
            lr_scheduler=cfg_lr_scheduler,
            loss=cfg_loss,
            metrics=cfg_metrics,
        )

    def test_forward_shape(
        self,
        processor_stage: ProcessorStage,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        batch_size = 2
        result = processor_stage(
            {
                "test-input": torch.rand(
                    batch_size,
                    2,
                    cfg_input_space["channels"],
                    *cfg_input_space["shape"],
                ),
                "target": torch.rand(
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

    def test_training_step_returns_prediction_target_and_loss(
        self,
        processor_stage: ProcessorStage,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        batch_size = 2
        batch = {
            "test-input": torch.rand(
                batch_size,
                2,
                cfg_input_space["channels"],
                *cfg_input_space["shape"],
            ),
            "target": torch.rand(
                batch_size,
                1,
                cfg_output_space["channels"],
                *cfg_output_space["shape"],
            ),
        }

        output = processor_stage.training_step(batch, 0)

        assert output.prediction.shape == (
            batch_size,
            1,
            cfg_output_space["channels"],
            *cfg_output_space["shape"],
        )
        assert torch.equal(output.target, batch["target"])
        assert output.loss.shape == torch.Size([])

    def test_training_step_raises_on_target_shape_mismatch(
        self,
        processor_stage: ProcessorStage,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        # The shape check only guards the custom loss path, which is the only one that
        # feeds the target through target_encoder; the standard path never touches it.
        processor_stage.processor = _FixedLossProcessor(
            data_space=processor_stage.processor.data_space,
            data_space_target=processor_stage.processor.data_space_target,
            n_forecast_steps=processor_stage.n_forecast_steps,
            n_history_steps=processor_stage.n_history_steps,
            loss=torch.tensor(0.5),
        )
        batch_size = 2
        batch = {
            "test-input": torch.rand(
                batch_size,
                2,
                cfg_input_space["channels"],
                *cfg_input_space["shape"],
            ),
            "target": torch.rand(
                batch_size,
                1,
                cfg_output_space["channels"] + 1,
                *cfg_output_space["shape"],
            ),
        }

        with pytest.raises(ValueError, match="does not match"):
            processor_stage.training_step(batch, 0)

    def test_training_step_uses_processor_supplied_loss(
        self,
        processor_stage: ProcessorStage,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        fixed_loss = torch.tensor(0.5)
        processor_stage.processor = _FixedLossProcessor(
            data_space=processor_stage.processor.data_space,
            data_space_target=processor_stage.processor.data_space_target,
            n_forecast_steps=processor_stage.n_forecast_steps,
            n_history_steps=processor_stage.n_history_steps,
            loss=fixed_loss,
        )
        batch_size = 2
        batch = {
            "test-input": torch.rand(
                batch_size,
                2,
                cfg_input_space["channels"],
                *cfg_input_space["shape"],
            ),
            "target": torch.rand(
                batch_size,
                1,
                cfg_output_space["channels"],
                *cfg_output_space["shape"],
            ),
        }

        output = processor_stage.training_step(batch, 0)

        assert output.loss is fixed_loss
        assert output.prediction.shape == (
            batch_size,
            1,
            cfg_output_space["channels"],
            *cfg_output_space["shape"],
        )

    def test_get_persistence_returns_tensor_when_decoder_has_skip_connection(
        self,
        encoder_stage: EncoderStage,
        target_encoder_stage: EncoderStage,
        *,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> None:
        skip_connection_decoder = DictConfig(
            {
                "_target_": "icenet_mp.models.decoders.NaiveLinearDecoder",
                "skip_connection": {"method": "additive"},
            }
        )
        decoder_stage = DecoderStage(
            decoder=skip_connection_decoder,
            encoders=[encoder_stage],
            target_dataset_name="target",
            target_variable_indices=[0],
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
        processor_stage = ProcessorStage(
            processor=cfg_processor,
            decoder_model=decoder_stage,
            target_encoder=target_encoder_stage,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=2,
            name="test-target_processor",
            optimizer=cfg_optimizer,
            output_space=cfg_output_space,
            scheduler=cfg_scheduler,
            lr_scheduler=cfg_lr_scheduler,
            loss=cfg_loss,
            metrics=cfg_metrics,
        )

        batch_size = 2
        inputs = {
            "target": torch.rand(
                batch_size,
                1,
                cfg_output_space["channels"],
                *cfg_output_space["shape"],
            ),
        }

        persistence = processor_stage._extract_anchor(inputs["target"])

        assert persistence is not None
        assert persistence.shape == (
            batch_size,
            len(decoder_stage.target_variable_indices),
            *cfg_output_space["shape"],
        )

    def test_encoders_and_decoder_parameters_are_frozen(
        self, processor_stage: ProcessorStage
    ) -> None:
        frozen_modules = (
            *processor_stage.encoders,
            processor_stage.target_encoder,
            processor_stage.decoder,
        )
        assert all(
            not param.requires_grad
            for module in frozen_modules
            for param in module.parameters()
        )

    def test_train_keeps_frozen_modules_in_eval_mode(
        self, processor_stage: ProcessorStage
    ) -> None:
        processor_stage.eval()
        for encoder in processor_stage.encoders:
            encoder.train()
        processor_stage.target_encoder.train()
        processor_stage.decoder.train()

        processor_stage.train()

        assert processor_stage.training
        assert all(not encoder.training for encoder in processor_stage.encoders)
        assert not processor_stage.target_encoder.training
        assert not processor_stage.decoder.training

    def test_from_template_builds_processor_stage_from_decoder_stage(
        self,
        decoder_stage: DecoderStage,
        target_encoder_stage: EncoderStage,
        cfg_processor: DictConfig,
    ) -> None:
        processor_stage = ProcessorStage.from_template(
            processor=cfg_processor,
            decoder_model=decoder_stage,
            target_encoder=target_encoder_stage,
        )

        assert processor_stage.hemisphere == decoder_stage.hemisphere
        assert processor_stage.n_forecast_steps == decoder_stage.n_forecast_steps
        assert processor_stage.n_history_steps == decoder_stage.n_history_steps
        assert processor_stage.name == (
            f"processor_{decoder_stage.n_history_steps}_to_"
            f"{decoder_stage.n_forecast_steps}"
        )
        assert processor_stage.target_encoder is target_encoder_stage.encoder
