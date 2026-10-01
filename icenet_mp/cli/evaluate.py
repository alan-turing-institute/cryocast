import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService

from .hydra import hydra_adaptor

# Create the typer app
evaluation_cli = typer.Typer(help="Evaluate models")

log = logging.getLogger(__name__)


def _require_callback_config(
    config: DictConfig, callback_name: str, cli_flag: str
) -> DictConfig:
    """Return an evaluate callback's config node, or raise if it's not configured.

    Args:
        config: The full composed configuration.
        callback_name: Key of the callback within `evaluate.callbacks`.
        cli_flag: Name of the CLI flag that requires this callback, used in the
            error message.

    Raises:
        ValueError: If `callback_name` is missing from `evaluate.callbacks` for the
            composed config (e.g. an evaluate variant that overrides the callbacks
            list without including it).

    """
    callbacks = config.get("evaluate", {}).get("callbacks", {})
    if callback_name not in callbacks:
        msg = (
            f"'{cli_flag}' requires the '{callback_name}' callback in "
            f"'evaluate.callbacks', but it is missing from this config "
            f"(check for an 'override callbacks:' entry that excludes it)."
        )
        raise ValueError(msg)
    return callbacks[callback_name]


@evaluation_cli.command()
@hydra_adaptor
def evaluate(
    config: DictConfig,
    checkpoint: Annotated[str, typer.Option(help="Path of a trained model checkpoint")],
    save_layer: Annotated[
        list[str] | None,
        typer.Option(
            "--save-layer",
            help=(
                "Dotted path of a model submodule to hook (e.g. 'processor.conv1'). "
                "Repeat the flag to hook multiple layers. "
                "Values for each selected layer will be saved to disk each batch."
            ),
        ),
    ] = None,
    *,
    save_predictions: Annotated[
        bool,
        typer.Option(
            "--save-predictions",
            help=(
                "Write predictions for the configured test period to a NetCDF file "
                "in the run directory. Evaluation must use a single process."
            ),
        ),
    ] = False,
) -> None:
    """Evaluate a pre-trained model."""
    # If activation saving is enabled, then add requested layers
    if layer_paths := list(save_layer or []):
        _require_callback_config(config, "activation_saver", "--save-layer")[
            "layer_paths"
        ] = layer_paths

    # If prediction saving is enabled, mark the prediction writer callback as such.
    # ModelService.build_trainer sets the output path from the run directory.
    if save_predictions:
        _require_callback_config(config, "prediction_writer", "--save-predictions")[
            "enabled"
        ] = True

    model = ModelService.from_checkpoint(config, Path(checkpoint).resolve())
    model.evaluate()


if __name__ == "__main__":
    evaluation_cli()
