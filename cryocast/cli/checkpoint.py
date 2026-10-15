import logging
from pathlib import Path
from pickle import UnpicklingError
from typing import Annotated

import typer

from cryocast.model_service import CheckpointFile, LegacyCheckpointFile

# Create the typer app
checkpoint_cli = typer.Typer(help="Manage model checkpoints")

log = logging.getLogger(__name__)


@checkpoint_cli.command()
def upgrade(
    checkpoints: Annotated[
        list[Path], typer.Argument(help="One or more checkpoint files to upgrade")
    ],
    *,
    trust: Annotated[
        bool,
        typer.Option(
            "--trust",
            help=(
                "Trust the checkpoints to run arbitrary code. Needed for checkpoints "
                "that cannot be loaded safely; only use this for files you trust."
            ),
        ),
    ] = False,
) -> None:
    """Upgrade checkpoints saved by older versions of the code.

    Each checkpoint, and the run's 'files/model_config.yaml', is rewritten so that it
    can be loaded safely, after backing up the original to '<name>.bak'. Files that
    are already up to date are left unchanged, so upgrading twice is safe.

    Checkpoints that cannot be loaded safely must be unpickled without restrictions
    to upgrade them, which can execute arbitrary code embedded in the file. This is
    refused unless '--trust' is given, so only pass it for checkpoints you trust.

    A checkpoint that cannot be upgraded is reported and skipped, so that the others
    are still upgraded, and the command exits with an error once all have been tried.
    """
    failed: list[Path] = []
    for path in checkpoints:
        try:
            LegacyCheckpointFile(CheckpointFile(path)).upgrade(trusted=trust)
        except (OSError, EOFError, RuntimeError, UnpicklingError, ValueError) as exc:
            log.error("Failed to upgrade %s: %s", path, exc)  # noqa: TRY400
            failed.append(path)
    if failed:
        log.error(
            "Failed to upgrade %d of %d checkpoints: %s",
            len(failed),
            len(checkpoints),
            ", ".join(map(str, failed)),
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    checkpoint_cli()
