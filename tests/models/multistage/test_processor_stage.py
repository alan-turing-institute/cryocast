from typing import Any

import pytest
import torch
from omegaconf import DictConfig

from cryocast.models.multistage import DecoderStage, EncoderStage, ProcessorStage
from cryocast.models.objectives import LatentObjective
from cryocast.models.processors import BaseProcessor
from cryocast.types import (
    DataSpace,
    ProcessorOutput,
    RolloutSpace,
    RolloutStrategy,
    SkipAnchorType,
    TensorNTCHW,
)


class _CaptureLatentLossProcessor(BaseProcessor):
    """Test double that records latent supervision passed to the processor."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(computes_loss_in_latent_space=True, **kwargs)
        self.target: TensorNTCHW | None = None

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        self.target = y
        prediction = x[:, -1:].expand(-1, self.n_forecast_steps, -1, -1, -1)
        return ProcessorOutput(prediction=prediction, loss=torch.tensor(0.0))


class TestProcessorStage:
    @pytest.fixture
    def processor_stage(
        self,
        decoder_stage: DecoderStage,
        *,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_lr_scheduler: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> ProcessorStage:
        return ProcessorStage(
            processor=cfg_processor,
            decoder_model=decoder_stage,
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

    def test_training_step_uses_processor_supplied_loss(
        self,
        processor_stage: ProcessorStage,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # training_step only honours ProcessorOutput.loss with a LatentObjective;
        # otherwise it goes through self(batch).
        fixed_loss = torch.tensor(0.5)
        processor = processor_stage.processor
        n_forecast_steps = processor_stage.n_forecast_steps

        def fixed_loss_rollout(
            x: TensorNTCHW,
            _y: TensorNTCHW | None = None,
        ) -> ProcessorOutput:
            return ProcessorOutput(prediction=x[:, -n_forecast_steps:], loss=fixed_loss)

        monkeypatch.setattr(
            processor_stage, "objective", LatentObjective(processor_stage)
        )
        monkeypatch.setattr(processor, "rollout", fixed_loss_rollout)
        batch_size = 2
        batch = {
            "test-input": torch.rand(
                batch_size,
                2,
                cfg_input_space["channels"],
                *cfg_input_space["shape"],
            ),
            "future_test-input": torch.rand(
                batch_size,
                n_forecast_steps,
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

    def test_latent_target_encodes_future_inputs(
        self,
        encoder_stage: EncoderStage,
        *,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
    ) -> None:
        decoder_stage = DecoderStage.from_template(
            decoder=cfg_decoder,
            encoders=[encoder_stage],
            output_space=DataSpace(
                channels=1, name=cfg_input_space["name"], shape=cfg_input_space["shape"]
            ),
            target_dataset_name=cfg_input_space["name"],
            target_variable_indices=[2],
        )
        processor_stage = ProcessorStage.from_template(
            processor=DictConfig(
                {"_target_": f"{__name__}.{_CaptureLatentLossProcessor.__name__}"}
            ),
            decoder_model=decoder_stage,
            rollout_strategy=RolloutStrategy.from_options(),
        )

        name = cfg_input_space["name"]
        input_encoder = processor_stage.encoders[0]
        history = torch.rand(2, 2, cfg_input_space["channels"], 16, 16)
        future = torch.rand(2, 1, cfg_input_space["channels"], 16, 16)
        expected_future_latent = input_encoder.rollout(future)

        processor_stage.training_step(
            {name: history, f"future_{name}": future, "target": future[:, :, 2:3]}, 0
        )

        capture = processor_stage.processor
        assert isinstance(capture, _CaptureLatentLossProcessor)
        assert capture.target is not None
        torch.testing.assert_close(capture.target, expected_future_latent)
        assert capture.target.shape[2] == processor_stage.processor.data_space.channels

    @pytest.mark.parametrize(
        "expected",
        [
            pytest.param(
                RolloutStrategy(RolloutSpace.LATENT, SkipAnchorType.INITIAL),
                id="latent-initial",
            ),
            pytest.param(
                RolloutStrategy(RolloutSpace.LATENT, SkipAnchorType.PREVIOUS),
                id="latent-previous",
            ),
            pytest.param(
                RolloutStrategy(RolloutSpace.PHYSICAL, SkipAnchorType.PREVIOUS),
                id="physical-previous",
            ),
        ],
    )
    def test_from_template_forwards_rollout_strategy(
        self,
        encoder_stage: EncoderStage,
        *,
        cfg_decoder: DictConfig,
        cfg_input_space: DictConfig,
        cfg_processor: DictConfig,
        expected: RolloutStrategy,
    ) -> None:
        # Physical rollout requires the target dataset to also be a model input
        decoder_stage = DecoderStage.from_template(
            decoder=cfg_decoder,
            encoders=[encoder_stage],
            output_space=DataSpace(
                channels=1, name=cfg_input_space["name"], shape=cfg_input_space["shape"]
            ),
            target_dataset_name=cfg_input_space["name"],
            target_variable_indices=[2],
        )
        processor_stage = ProcessorStage.from_template(
            processor=cfg_processor,
            decoder_model=decoder_stage,
            rollout_strategy=expected,
        )

        assert processor_stage.rollout_strategy == expected
        # The resolved options are saved, so that a checkpoint reloads the same strategy
        assert processor_stage.hparams["rollout_space"] == expected.space
        assert processor_stage.hparams["skip_anchor"] == expected.anchor

    def test_from_template_copies_frozen_finaliser_from_decoder_stage(
        self,
        encoder_stage: EncoderStage,
        *,
        cfg_processor: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        # The gated skip connection has weights
        decoder_stage = DecoderStage.from_template(
            decoder=DictConfig(
                {"_target_": "cryocast.models.decoders.NaiveLinearDecoder"}
            ),
            finaliser={"skip_connection": {"method": "gated"}},
            encoders=[encoder_stage],
            output_space=DataSpace.from_dict(cfg_output_space),
            target_dataset_name="target",
            target_variable_indices=[0],
        )
        processor_stage = ProcessorStage.from_template(
            processor=cfg_processor,
            decoder_model=decoder_stage,
            rollout_strategy=RolloutStrategy.from_options(),
        )
        target = torch.rand(
            2, 1, cfg_output_space["channels"], *cfg_output_space["shape"]
        )

        anchor = processor_stage._anchor_nchw(target)

        # The skip connection comes from the DecoderStage, so there is an anchor
        assert anchor is not None
        assert torch.equal(anchor, target[:, -1, [0]])
        # The skip connection weights must not train
        assert processor_stage.finaliser.frozen
        assert all(
            not param.requires_grad for param in processor_stage.finaliser.parameters()
        )

    def test_encoders_and_decoder_parameters_are_frozen(
        self, processor_stage: ProcessorStage
    ) -> None:
        frozen_modules = (
            *processor_stage.encoders,
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
        processor_stage.decoder.train()

        processor_stage.train()

        assert processor_stage.training
        assert all(not encoder.training for encoder in processor_stage.encoders)
        assert not processor_stage.decoder.training

    def test_from_template_builds_processor_stage_from_decoder_stage(
        self,
        decoder_stage: DecoderStage,
        cfg_processor: DictConfig,
    ) -> None:
        processor_stage = ProcessorStage.from_template(
            processor=cfg_processor,
            decoder_model=decoder_stage,
            rollout_strategy=RolloutStrategy.from_options(),
        )

        assert processor_stage.hemisphere == decoder_stage.hemisphere
        assert processor_stage.n_forecast_steps == decoder_stage.n_forecast_steps
        assert processor_stage.n_history_steps == decoder_stage.n_history_steps
        assert processor_stage.name == (
            f"processor_{decoder_stage.n_history_steps}_to_"
            f"{decoder_stage.n_forecast_steps}"
        )
