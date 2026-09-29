import logging
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

import tqdm
from anemoi.datasets import open_dataset
from anemoi.datasets.commands.cleanup import Cleanup
from anemoi.datasets.commands.finalise import Finalise
from anemoi.datasets.commands.init import Init
from anemoi.datasets.commands.inspect import InspectZarr
from anemoi.datasets.commands.load import Load
from anemoi.datasets.create.recipe import Recipe
from anemoi.datasets.usage.gridded import MissingDateError
from zarr.errors import PathNotFoundError

from icenet_mp.types import (
    AnemoiCleanupArgs,
    AnemoiDatasetStatus,
    AnemoiFinaliseArgs,
    AnemoiInitArgs,
    AnemoiInspectArgs,
    AnemoiLoadArgs,
)

from .postprocessors import CompositePostprocessor
from .preprocessors import CompositePreprocessor

logger = logging.getLogger(__name__)


class DataDownloader:
    def __init__(
        self,
        name: str,
        base_path: Path,
        anemoi_config: dict[str, Any],
    ) -> None:
        """Initialise a DataDownloader from a config, and a preprocessor and postprocessor."""
        self.name = name
        self.group = anemoi_config.get("group_as", name)
        self.path_dataset = base_path / "data" / "anemoi" / f"{name}.zarr"
        self.recipe = Recipe(**anemoi_config)
        self.preprocessor = CompositePreprocessor(
            name, anemoi_config.get("preprocessors") or {}, base_path
        )
        self.postprocessor = CompositePostprocessor(
            name, anemoi_config.get("postprocessors") or {}, base_path
        )

    def artifacts(self) -> list[Path]:
        """Return a list of temporary artifacts created during the download and finalise process."""
        return [
            path
            for path in self.path_dataset.parent.glob(f"{self.path_dataset.stem}.*")
            if path != self.path_dataset
        ]

    def check_status(self) -> AnemoiDatasetStatus:
        """Return the status of the dataset."""
        try:
            ds_info = InspectZarr()._info(str(self.path_dataset))
            copy_in_progress = ds_info.copy_in_progress
            try:
                statistics_ready = ds_info.statistics_ready
            except KeyError:
                statistics_ready = False
            if copy_in_progress:
                # If a copy is in progress then the download is incomplete
                download_complete = False
            elif ds_info.dataset is None:
                # ... if there is no dataset object then the download is incomplete
                download_complete = False
            elif (build_flags := ds_info.build_flags) is not None:
                # If build flags are present and all true then the download is complete
                download_complete = len(build_flags) > 0 and bool(all(build_flags))
            elif statistics_ready or (ds_info.statistics_started is not None):
                # If statistics generation has started then the download is complete
                download_complete = True
            else:
                msg = (
                    f"Unable to determine readiness status for dataset {self.name} at "
                    f"{self.path_dataset}. Please check manually."
                )
                raise RuntimeError(msg)
        except (
            AttributeError,
            FileNotFoundError,
            KeyError,
            PathNotFoundError,
            ValueError,
        ) as exc:
            msg = f"Unable to get status for {self.name} at {self.path_dataset}"
            raise RuntimeError(msg) from exc
        return AnemoiDatasetStatus(
            copy_in_progress=copy_in_progress,
            download_complete=download_complete,
            is_finalised=download_complete and statistics_ready,
        )

    def create(self, *, overwrite: bool = False) -> None:
        """Ensure that a single Anemoi dataset exists."""
        # If we are overwriting we delete any existing dataset
        if overwrite:
            logger.info(
                "Overwrite set to true, redownloading %s to %s",
                self.name,
                self.path_dataset,
            )
            shutil.rmtree(self.path_dataset, ignore_errors=True)

        # Otherwise check whether a valid dataset exists
        elif self.path_dataset.exists():
            try:
                status = self.check_status()
            except RuntimeError as exc:
                msg = f"Status of dataset {self.name} at {self.path_dataset} could not be determined. Please check manually."
                raise RuntimeError(msg) from exc
            else:
                if status.copy_in_progress:
                    # If the dataset is being written to, we exit without error
                    logger.warning(
                        "Dataset %s at %s is currently being downloaded by another process.",
                        self.name,
                        self.path_dataset,
                    )
                    return
                if status.download_complete:
                    self.finalise(overwrite=overwrite, status=status)
                    try:
                        self.inspect()
                    except RuntimeError as exc:
                        msg = f"Dataset {self.name} at {self.path_dataset} could not be inspected. Please check manually."
                        raise RuntimeError(msg) from exc
                    # At this point we have a valid dataset so we exit without error
                    logger.info(
                        "Dataset %s at %s has been downloaded and seems to be valid.",
                        self.name,
                        self.path_dataset,
                    )
                    return

        # At this point there is no dataset at the requested path so we download to it
        self.download(overwrite=overwrite)

    def download(self, *, overwrite: bool) -> None:
        """Download an Anemoi dataset in parts."""
        self.preprocessor.process(overwrite=overwrite)
        logger.info("Creating dataset %s at %s.", self.name, self.path_dataset)
        # Initialise
        self.initialise()
        # Load in parts
        self.load_in_chunks()
        # Attempt to finalise, creating masks if necessary
        status = self.check_status()
        if status.copy_in_progress:
            logger.warning(
                "Dataset %s at %s is still being copied by another process, skipping finalise.",
                self.name,
                self.path_dataset,
            )
        elif status.download_complete:
            self.finalise(overwrite=overwrite, status=status)
        else:
            logger.warning(
                "Dataset %s at %s is not fully loaded, skipping finalise.",
                self.name,
                self.path_dataset,
            )

    def finalise(self, *, overwrite: bool, status: AnemoiDatasetStatus) -> None:
        """Finalise the segmented Anemoi dataset."""
        if not status.is_finalised:
            Finalise().run(
                AnemoiFinaliseArgs(
                    path=str(self.path_dataset),
                    recipe=self.recipe,
                )
            )
            logger.info("Finalised dataset %s at %s.", self.name, self.path_dataset)

        # Create active grid cell and land masks if appropriate
        self.postprocessor.process(self.path_dataset, overwrite=overwrite)

        # Cleanup any temporary artifacts created during the download and finalise process
        if self.artifacts():
            with suppress(ValueError):
                Cleanup().run(AnemoiCleanupArgs(path=str(self.path_dataset)))
            if remaining := self.artifacts():
                logger.warning("Residual artifacts for dataset %s:", self.name)
                for artifact in remaining:
                    logger.warning("... %s", artifact)
            else:
                logger.info("Cleaned up temporary artifacts for dataset %s.", self.name)

    def initialise(self) -> None:
        """Initialise an Anemoi dataset."""
        if self.path_dataset.exists():
            logger.info(
                "Dataset %s at %s is already initialised.", self.name, self.path_dataset
            )
            return
        try:
            Init().run(
                AnemoiInitArgs(
                    path=str(self.path_dataset),
                    recipe=self.recipe,
                )
            )
            logger.info("Initialised dataset %s at %s.", self.name, self.path_dataset)
        except (AttributeError, FileNotFoundError, PathNotFoundError) as exc:
            msg = f"Failed to initialise dataset {self.name} at {self.path_dataset}."
            raise RuntimeError(msg) from exc

    def inspect(
        self,
        *,
        verbose: bool = False,
    ) -> None:
        """Inspect an Anemoi dataset."""
        if not self.path_dataset.exists():
            msg = f"Dataset {self.name} not found at {self.path_dataset}"
            raise RuntimeError(msg)
        try:
            if verbose:
                InspectZarr().run(
                    AnemoiInspectArgs(
                        path=str(self.path_dataset),
                        detailed=True,
                        progress=False,
                        statistics=False,  # anemoi's brute-force stats crash on missing dates; integrity_check() reads the data instead
                        size=True,
                    )
                )
                self.integrity_check()
            else:
                ds_info = InspectZarr()._info(str(self.path_dataset))
                logger.info("  Path    : %s", ds_info.path)
                if (flags := ds_info.build_flags) is not None:
                    done = sum(flags)
                    total = len(flags)
                    completion_fraction = done / total if total else 0.0
                    chunk_info = f"{sum(flags)}/{len(flags)}"
                else:
                    completion_fraction = 1
                    chunk_info = "all"
                logger.info(
                    "  Progress: %.0f%% (%s chunks downloaded).",
                    100 * completion_fraction,
                    chunk_info,
                )
        except (ValueError, KeyError):
            logger.warning("Further dataset information unavailable.")
        except (AttributeError, FileNotFoundError, PathNotFoundError) as exc:
            msg = f"Failed to load dataset {self.name} at {self.path_dataset}"
            raise RuntimeError(msg) from exc

    def integrity_check(self) -> None:
        """Sequentially read every non-missing date to detect zarr corruption."""
        ds = open_dataset(self.path_dataset)
        missing_indices: set[int] = set(getattr(ds, "missing", None) or ())
        for idx in tqdm.tqdm(range(len(ds)), total=len(ds), desc="Integrity check"):
            try:
                if idx not in missing_indices:
                    ds[idx]
            except (MissingDateError, OSError) as exc:
                msg = f"❌ Integrity check for {self.name}: date {idx} unreadable."
                raise RuntimeError(msg) from exc
        logger.info(
            "✅ Integrity check for %s: %d/%d date(s) verified (%d known missing).",
            self.name,
            len(ds) - len(missing_indices),
            len(ds),
            len(missing_indices),
        )

    def load_in_chunks(self) -> None:
        """Download a single Anemoi dataset in chunks, skipping those already present."""
        Load().run(
            AnemoiLoadArgs(
                path=str(self.path_dataset),
                recipe=self.recipe,
            )
        )
