import pytest
from omegaconf import DictConfig

from icenet_mp.models.multistage import DecoderStage, EncoderStage
from icenet_mp.types import DataSpace


@pytest.fixture
def encoder_stage(
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
    """An EncoderStage for the "test-input" dataset."""
    return EncoderStage(
        channel_names=["channel-0", "channel-1", "channel-2", "channel-3"],
        data_space_in=DataSpace.from_dict(cfg_input_space),
        encoder=cfg_encoders["test-input"],
        decoder=cfg_decoder,
        latent_space=cfg_encoders["latent_space"],
        hemisphere="north",
        input_spaces=[cfg_input_space],
        n_forecast_steps=1,
        n_history_steps=1,
        name="test-input_encoder",
        optimizer=cfg_optimizer,
        output_space=cfg_output_space,
        scheduler=cfg_scheduler,
        lr_scheduler=cfg_lr_scheduler,
        loss=cfg_loss,
        metrics=cfg_metrics,
    )


@pytest.fixture
def decoder_stage(
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
) -> DecoderStage:
    """A DecoderStage wrapping `encoder_stage`. Requires two history steps."""
    return DecoderStage(
        decoder=cfg_decoder,
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
