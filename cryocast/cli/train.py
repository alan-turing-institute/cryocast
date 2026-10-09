import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from cryocast.model_service import CheckpointFile, ModelService

from .hydra import hydra_adaptor

# Create the typer app
training_cli = typer.Typer(help="Train models")

log = logging.getLogger(__name__)


@training_cli.command()
@hydra_adaptor
def train(
    config: DictConfig,
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Path to a directory of existing checkpoints to resume from. With "
                "--multistage, any component with a checkpoint in this directory will "
                "be loaded and training will be skipped for that component. Without "
                "--multistage, the directory must contain a 'last*.ckpt' file and "
                "training will resume from it."
            )
        ),
    ] = None,
    *,
    multistage: Annotated[
        bool,
        typer.Option(
            "--multistage",
            help=(
                "Train an EncodeProcessDecode model in multiple stages (encoders, "
                "decoder, processor, then finetune). Default is single-stage training."
            ),
        ),
    ] = False,
) -> None:
    """Train a model."""
    checkpoint_dir = checkpoint_dir.resolve() if checkpoint_dir else None
    # For multistage training, we pass the directory of checkpoints to train_multistage
    if multistage:
        model_service = ModelService.from_config(config)
        model_service.train_multistage(checkpoint_dir=checkpoint_dir)
    # For single-stage training, we resume from the most recent checkpoint
    elif checkpoint_dir:
        checkpoint_file = CheckpointFile.find_last(checkpoint_dir)
        ModelService.from_checkpoint(config, checkpoint_file).train()
    # ... or if no checkpoint is provided, we start training from scratch
    else:
        ModelService.from_config(config).train()


if __name__ == "__main__":
    training_cli()
