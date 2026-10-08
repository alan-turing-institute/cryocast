from pathlib import Path
from typing import Annotated

import typer

from cryocast.model_service import CheckpointFile, LegacyCheckpointFile

# Create the typer app
checkpoint_cli = typer.Typer(help="Manage model checkpoints")


@checkpoint_cli.command()
def upgrade(
    checkpoints: Annotated[
        list[Path], typer.Argument(help="One or more checkpoint files to upgrade")
    ],
) -> None:
    """Upgrade checkpoints saved by older versions of CryoCast or by icenet-mp.

    Each checkpoint, and the run's 'files/model_config.yaml', is rewritten so that it
    can be loaded safely, after backing up the original to '<name>.bak'. Files that
    are already up to date are left unchanged, so upgrading twice is safe.
    """
    for path in checkpoints:
        LegacyCheckpointFile(CheckpointFile(path)).upgrade()


if __name__ == "__main__":
    checkpoint_cli()
