import logging
import os
from pathlib import Path
from typing import Annotated

import torch
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from optuna.trial import TrialState

from icenet_mp.model_service import ModelService
from icenet_mp.sweep import OptunaSweep

from .hydra import hydra_adaptor

# Configure logging
log = logging.getLogger(__name__)

# Create the typer app
sweep_cli = typer.Typer(help="Generate W&B sweeps with Optuna-sampled hyperparameters")


@sweep_cli.command()
@hydra_adaptor
def initialise(
    config: DictConfig,
    sweep_yaml: Annotated[
        Path,
        typer.Option(
            "--sweep-yaml", help="Full path to a sweep search-space YAML file"
        ),
    ],
) -> None:
    """Initialise a W&B sweep with Optuna-sampled hyperparameters.

    Create a W&B sweep with a grid search over a set of hyperparameter combinations
    specified by Optuna. Create the Optuna study directory and save the model and sweep
    configs.
    """
    sweep = OptunaSweep.from_yaml(sweep_yaml)

    # Fail here for typos or unsupported parameters, rather than creating a broken sweep
    sweep.validate_parameters(config)

    # Initialise a W&B sweep
    sweep_id = sweep.initialise_sweep(config)
    log.info("Initialised a W&B sweep with ID %s", sweep_id)

    # Initialise Optuna study
    sweep.initialise_study(config, sweep_id)
    log.info("Initialised an Optuna study at %s", sweep.study_path)


@sweep_cli.command()
def summarise(
    sweep_path: Annotated[
        Path,
        typer.Option(
            "--sweep-path",
            help="Full path to a local sweep directory",
        ),
    ],
) -> None:
    """Summarise the best parameters found in a W&B sweep."""
    sweep = OptunaSweep.from_path(sweep_path)
    trials = sweep.study.get_trials()
    n_completed = sum(1 for t in trials if t.state == TrialState.COMPLETE)
    n_failed = sum(1 for t in trials if t.state == TrialState.FAIL)
    n_pruned = sum(1 for t in trials if t.state == TrialState.PRUNED)
    n_running = sum(1 for t in trials if t.state == TrialState.RUNNING)
    log.info(
        "Study contains %d trial(s): %d completed, %d running, %d pruned, %d failed",
        len(trials),
        n_completed,
        n_running,
        n_pruned,
        n_failed,
    )
    if n_completed == 0:
        log.info("No trials have completed yet. Nothing to summarise.")
        return

    # Log the best trial and its parameters
    best = sweep.study.best_trial
    log.info("Trial %d performed best, with loss %f.", best.number, best.value)
    log.info("Best trial parameters:")
    for parameter_name, parameter_value in best.params.items():
        log.info("  %s: %s", parameter_name, parameter_value)

    # Log parameter importance, if available
    importances = sweep.parameter_importances()
    if not importances:
        log.info(
            "Could not estimate parameter importance for %d trials", sweep.n_trials
        )
        return
    log.info("Parameter importance:")
    name_width = max(len(name) for name in importances)
    log.info("  %-*s  importance", name_width, "parameter")
    for parameter_name, importance in importances.items():
        log.info("  %-*s  %.3f", name_width, parameter_name, importance)


@sweep_cli.command()
def trial(
    sweep_path: Annotated[
        Path,
        typer.Option(
            "--sweep-path",
            help="Full path to a local sweep directory",
        ),
    ],
    *,
    checkpoint_dir: Annotated[
        str | None,
        typer.Option(
            "--checkpoint-dir",
            help=(
                "Path to a directory of existing checkpoints to resume from. With "
                "--multistage, any component with a checkpoint in this directory will "
                "be loaded and training will be skipped for that component. Without "
                "--multistage, the directory must contain a 'last*.ckpt' file and "
                "training will resume from it."
            ),
        ),
    ] = None,
    multistage: Annotated[
        bool,
        typer.Option(
            "--multistage",
            help=(
                "Train an EncodeProcessDecode model in multiple stages (encoders, then "
                "decoder, then processor, then finetune). Default is single-stage "
                "training."
            ),
        ),
    ] = False,
) -> None:
    """Run a single trial from a W&B sweep."""
    # Load the Optuna sweep, start a trial, and get its parameter overrides
    sweep = OptunaSweep.from_path(sweep_path)
    trial, overrides = sweep.ask()

    try:
        # Generate a merged config for this trial
        log.info("Running trial %d with overrides:", trial.number)
        for parameter, value in overrides:
            log.info("  %s = %s", parameter.name, value)
        config = sweep.generate_trial_config(overrides)

        # Set W&B environment variables to ensure we are connected to the correct sweep
        os.environ["WANDB_SWEEP_ID"] = sweep.study_name
        os.environ["WANDB_ENTITY"] = sweep.entity
        os.environ["WANDB_PROJECT"] = "train"

        # Train the model for this trial
        model = ModelService.from_config(config)
        trainer = model.train(
            checkpoint_dir=Path(checkpoint_dir).resolve() if checkpoint_dir else None,
            multistage=multistage,
        )

        # If there is exactly one ModelCheckpoint callback then we will use that
        checkpoints = [
            ckpt
            for ckpt in trainer.checkpoint_callbacks
            if isinstance(ckpt, ModelCheckpoint)
        ]
        checkpoint = checkpoints[0] if len(checkpoints) == 1 else None
    except torch.OutOfMemoryError:
        # CUDA OOM is a routine outcome of hyperparameter sampling, not a bug. Skip the
        # full traceback dump and exit cleanly so logs stay readable.
        log.error(  # noqa: TRY400
            "Trial %d failed: ran out of GPU memory. This is likely because the "
            "sampled hyperparameters produced a model too large to fit alongside the "
            "configured batch size.",
            trial.number,
        )
        sweep.tell(trial, state=TrialState.PRUNED)
        raise typer.Exit(code=1) from None
    except Exception:
        # Mark the trial as failed after any exception before continuing
        log.exception("Trial %d failed.", trial.number)
        sweep.tell(trial, state=TrialState.FAIL)
        raise

    # Record the trial result and log the best trial
    if checkpoint is None:
        log.warning(
            "Trial %d failed: could not find a unique ModelCheckpoint callback, so the "
            "trial cannot be scored.",
            trial.number,
        )
        sweep.tell(trial, state=TrialState.FAIL)
        return
    if checkpoint.best_model_score is None:
        log.warning(
            "Trial %d failed: the checkpoint callback monitoring '%s' has no "
            "best_model_score, so it cannot be scored.",
            trial.number,
            checkpoint.monitor,
        )
        sweep.tell(trial, state=TrialState.FAIL)
        return
    result = sweep.tell(trial, checkpoint.best_model_score.item())
    log.info("Trial %d completed with value %f", result.number, result.value)
    best = sweep.study.best_trial
    log.info("Best trial (%d) completed with value %f.", best.number, best.value)
    log.info("Best trial parameters:")
    for parameter_name, parameter_value in best.params.items():
        log.info("  %s: %s", parameter_name, parameter_value)


if __name__ == "__main__":
    sweep_cli()
