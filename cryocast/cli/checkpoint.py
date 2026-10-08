from pathlib import Path
from typing import Annotated

import typer

from cryocast.model_service.checkpoints import upgrade_checkpoint

# Create the typer app
checkpoint_cli = typer.Typer(help="Manage model checkpoints")


@checkpoint_cli.command()
def upgrade(
    checkpoints: Annotated[
        list[Path], typer.Argument(help="One or more checkpoint files to upgrade")
    ],
) -> None:
    """Upgrade checkpoints saved by older versions of CryoCast or by icenet-mp.

    Each checkpoint is backed up to '<name>.bak' and rewritten so that it can be
    loaded safely. The run's 'files/model_config.yaml' is also updated if needed.
    """
    for checkpoint in checkpoints:
        upgrade_checkpoint(checkpoint.resolve())


if __name__ == "__main__":
    checkpoint_cli()
