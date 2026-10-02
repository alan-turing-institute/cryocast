"""``imp feature-importance`` - Random Forest feature importance for input variables."""

import typer
from omegaconf import DictConfig

from icenet_mp.feature_importance import compute_feature_importance

from .hydra import hydra_adaptor

# Create the typer app
feature_importance_cli = typer.Typer(
    help="Random Forest feature importance for input variables"
)


@feature_importance_cli.command(name="feature-importance")
@hydra_adaptor
def feature_importance(config: DictConfig) -> None:
    """Fit a Random Forest and print input variables ranked by importance."""
    ranked = compute_feature_importance(config)
    typer.echo(f"\n{'Rank':<5} {'Variable':<40} {'Importance':>12}")
    typer.echo("-" * 60)
    for rank, (name, score) in enumerate(ranked, start=1):
        typer.echo(f"{rank:<5} {name:<40} {score:>12.6f}")


if __name__ == "__main__":
    feature_importance_cli()
