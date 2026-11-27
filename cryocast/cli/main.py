"""Main entrypoint for the CLI application."""

import logging
import warnings

import typer

from cryocast.compatibility.lightning import register_accelerators
from cryocast.compatibility.torch import patch_parameter_deepcopy
from cryocast.ingestion.filters import register_filters
from cryocast.ingestion.sources import register_sources
from cryocast.visualisations import register_animation_backends

from .checkpoint import checkpoint_cli
from .datasets import datasets_cli
from .evaluate import evaluation_cli
from .feature_importance import feature_importance_cli
from .sweep import sweep_cli
from .train import training_cli

# Configure logging
logging.basicConfig(
    format="🧊 [%(asctime)s] %(message)s",
    datefmt=r"%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)
log = logging.getLogger(__name__)

# Configure external libraries used by the application
log.debug("Configuring external libraries...")
patch_parameter_deepcopy()
register_accelerators()
register_animation_backends()
register_filters()
register_sources()
# Ignore warnings about known PyTorch issues
warnings.filterwarnings(
    "ignore",
    message=".*Using padding='same' with even kernel lengths and odd dilation.*",
)

# Create the typer app
app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Entrypoint for CryoCast CLI application.",
    no_args_is_help=True,
)
app.add_typer(checkpoint_cli, name="checkpoint")
app.add_typer(datasets_cli, name="datasets")
app.add_typer(evaluation_cli)
app.add_typer(feature_importance_cli)
app.add_typer(sweep_cli, name="sweep")
app.add_typer(training_cli)


def run() -> None:
    """Initialise and run the CLI application."""
    try:
        # Run the app
        app()
    except NotImplementedError as exc:
        # Catch MPS-not-implemented errors; anything else propagates as normal
        if "not currently implemented for the MPS device" not in str(exc):
            raise
        msg = (
            "WARNING: job failed due to running on MPS without CPU fallback enabled.\n"
            "Please rerun after setting the environment variable "
            "`PYTORCH_ENABLE_MPS_FALLBACK=1`. This *must* be set before starting the "
            "Python interpreter. It will be slower than running natively on MPS."
        )
        log.error(msg)  # noqa: TRY400
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
