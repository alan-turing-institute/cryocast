import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from omegaconf import DictConfig

from icenet_mp.data import (
    SingleDataset,
    generate_daily_climatology,
    save_daily_climatology,
)
from icenet_mp.ingestion import build_downloaders
from icenet_mp.visualisations import DatasetMediaWriter

from .hydra import hydra_adaptor

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")

logger = logging.getLogger(__name__)


@datasets_cli.command("create")
@hydra_adaptor
def create(
    config: DictConfig,
    *,
    overwrite: Annotated[
        bool, typer.Option(help="Specify whether to overwrite existing datasets")
    ] = False,
) -> None:
    """Create all datasets."""
    for downloader in build_downloaders(config):
        logger.info("Creating dataset %s.", downloader.name)
        try:
            downloader.create(overwrite=overwrite)
        except RuntimeError as exc:
            logger.error("Failed to create %s: %s", downloader.name, exc)  # noqa: TRY400
            raise typer.Exit(1) from exc


@datasets_cli.command("climatology")
@hydra_adaptor
def climatology(
    config: DictConfig,
    *,
    dataset: Annotated[
        str, typer.Option(help="Configured SIC dataset to use as the reference record")
    ],
    variable: Annotated[
        str, typer.Option(help="Variable to average into the daily climatology")
    ] = "ice_conc",
    years: Annotated[
        int, typer.Option(help="Number of complete reference years to average")
    ] = 35,
    reference_end_year: Annotated[
        int | None,
        typer.Option(help="Final reference year; defaults to the dataset's final year"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Output .npz path; defaults under data/climatology"),
    ] = None,
) -> None:
    """Generate a reusable daily SIC climatology baseline."""
    downloader = next(
        (item for item in build_downloaders(config) if item.name == dataset), None
    )
    if downloader is None:
        logger.error("Configured dataset %s was not found.", dataset)
        raise typer.Exit(1)
    if not downloader.path_dataset.exists():
        logger.error("Dataset %s not found at %s", dataset, downloader.path_dataset)
        raise typer.Exit(1)

    source = SingleDataset(
        name=dataset,
        input_files=[downloader.path_dataset],
        normalise=False,
    )
    if reference_end_year is None:
        reference_end_year = int(np.datetime_as_string(source.end_date, unit="Y"))

    result = generate_daily_climatology(
        source,
        variable=variable,
        reference_end_year=reference_end_year,
        years=years,
    )
    output_path = output or (
        Path(config["base_path"])
        / "data"
        / "climatology"
        / f"{dataset}-{variable}-{years}y-{reference_end_year}.npz"
    )
    saved = save_daily_climatology(result, output_path)
    logger.info(
        "Saved %d-day climatology for %s:%s to %s.",
        len(result.month_day),
        dataset,
        variable,
        saved,
    )


@datasets_cli.command("inspect")
@hydra_adaptor
def inspect(
    config: DictConfig,
    *,
    verbose: Annotated[
        bool, typer.Option(help="Show detailed dataset information")
    ] = False,
) -> None:
    """Inspect all datasets."""
    for downloader in build_downloaders(config):
        logger.info("Inspecting dataset %s.", downloader.name)
        try:
            downloader.inspect(verbose=verbose)
        except RuntimeError:
            logger.error("Inspecting dataset %s failed, skipping.", downloader.name)  # noqa: TRY400


@datasets_cli.command("plot")
@hydra_adaptor
def plot(
    config: DictConfig,
    *,
    dataset: Annotated[
        str | None, typer.Option(help="Only plot the named configured dataset")
    ] = None,
    timestep: Annotated[int, typer.Option(help="Dataset timestep index to plot")] = 0,
    video: Annotated[
        bool,
        typer.Option(
            help="Animate --n-steps consecutive timesteps instead of plotting one"
        ),
    ] = False,
    n_steps: Annotated[
        int,
        typer.Option(help="Number of consecutive timesteps to animate with --video"),
    ] = 10,
) -> None:
    """Plot one timestep of configured datasets."""
    writer = DatasetMediaWriter(Path(config["base_path"]).resolve())
    downloaders = [
        d for d in build_downloaders(config) if dataset is None or d.name == dataset
    ]
    if dataset is not None and not downloaders:
        logger.error("Configured dataset %s was not found.", dataset)
        raise typer.Exit(1)
    for downloader in downloaders:
        logger.info("Plotting dataset %s.", downloader.name)
        if downloader.path_dataset.exists():
            ds = SingleDataset(
                name=downloader.name,
                input_files=[downloader.path_dataset],
                group=downloader.group,
                normalise=False,
            )
            n_saved = (
                writer.video(
                    dataset=ds,
                    n_steps=n_steps,
                    timestep=timestep,
                )
                if video
                else writer.static(
                    dataset=ds,
                    timestep=timestep,
                )
            )
            logger.info("Saved %d plots for dataset %s.", n_saved, downloader.name)
        else:
            logger.error(
                "Dataset %s not found at %s", downloader.name, downloader.path_dataset
            )


@datasets_cli.command("masks")
@hydra_adaptor
def masks(
    config: DictConfig,
    *,
    overwrite: Annotated[
        bool, typer.Option(help="Specify whether to overwrite existing masks")
    ] = False,
) -> None:
    """Create land / active grid cell masks."""
    for downloader in build_downloaders(config):
        logger.info("Generating masks for dataset %s.", downloader.name)
        downloader.postprocessor.process(downloader.path_dataset, overwrite=overwrite)


if __name__ == "__main__":
    datasets_cli()
