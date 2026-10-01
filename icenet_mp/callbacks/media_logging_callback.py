import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor
from torch.utils.data import DataLoader

from icenet_mp.data import CombinedDataset
from icenet_mp.models import BaseModel
from icenet_mp.types import (
    ArrayTHW,
    ModelStepOutput,
    PlotSpec,
    SupportsImageLogging,
    SupportsVideoLogging,
)
from icenet_mp.utils import (
    datetime_from_npdatetime,
    iso_from_date,
    npdatetime_from_datetime,
)
from icenet_mp.visualisations import MediaPublisher
from icenet_mp.visualisations.land_mask import LandMask

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class MediaLoggingCallback(Callback):
    """A callback to create and log images during evaluation."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        frequency: dict[str, int] | None = None,
        make_input_plots: bool = False,
        make_static_plots: bool = True,
        make_video_plots: bool = True,
        model_name: str | None = None,
        plot_spec: PlotSpec | None = None,
        prefix: str | None = None,
    ) -> None:
        """Create images during evaluation or training validation.

        Note that we do not plot during training as the data is shuffled so it would be
        difficult to work out which date corresponds to each batch.

        Args:
            frequency: A dictionary specifying how often to make plots, with keys:
                batch (plot every N batches)
                epoch (plot every N epochs)
                number (plot N sample batches evenly spaced across the epoch)
            make_input_plots: Whether to plot the raw inputs.
            make_static_plots: Whether to create static plots.
            make_video_plots: Whether to create video plots.
            model_name: The name of the model to include in plot subtitles.
            plot_spec: Plotting specification to use (contains difference settings, timestep selection, etc.).
            prefix: An optional prefix to add to all plot keys when logging.

        """
        super().__init__()
        # When to make plots
        self.frequency_batch = int((frequency or {}).get("batch", -1))
        self.frequency_epoch = int((frequency or {}).get("epoch", -1))
        self.frequency_number = int((frequency or {}).get("number", -1))

        # Which plots to make
        self.make_input_plots = make_input_plots
        self.make_static_plots = make_static_plots
        self.make_video_plots = make_video_plots

        # Plotting specification
        self._land_mask_cache: dict[Path | None, LandMask] = {}
        self._model_name: str | None = model_name
        self._plot_spec = PlotSpec() + plot_spec
        self.prefix: str | None = prefix

        # Cache the most recent batch
        self.cached_batch_idx_: int | None = None
        self.cached_dataloader_idx_: int | None = None
        self.cached_outputs_: ModelStepOutput | None = None

        # Track any plots that should only be logged once per dataloader
        self._is_logged: set[tuple[str, int | None, str]] = set()

    def cache_batch(
        self,
        batch_idx: int,
        dataloader_idx: int,
        outputs: Tensor | Mapping[str, Any] | None,
    ) -> None:
        """Cache the current batch information for use in epoch-end plotting."""
        if isinstance(outputs, Mapping):
            self.cached_outputs_ = ModelStepOutput(**outputs)
            self.cached_batch_idx_ = batch_idx
            self.cached_dataloader_idx_ = dataloader_idx

    def is_logged(self, kind: str, dates: list[datetime]) -> bool:
        """Record (kind, dataloader, date) as logged, returning True if already seen."""
        key = (kind, self.cached_dataloader_idx_, iso_from_date(dates[0]))
        if key in self._is_logged:
            return True
        self._is_logged.add(key)
        return False

    def is_sample_batch(self, batch_idx: int, total_batches: int | float) -> bool:  # noqa: PYI041
        """Return True if batch_idx is one of frequency_number equally-spaced targets."""
        if (
            self.frequency_number <= 0
            or not math.isfinite(total_batches)
            or total_batches <= 0
        ):
            return False
        n = int(min(self.frequency_number, total_batches))
        if n == 1:
            return batch_idx == total_batches - 1
        targets = {round(i * (total_batches - 1) / (n - 1)) for i in range(n)}
        return batch_idx in targets

    def load_dataset(
        self, dataloader: DataLoader | list[DataLoader] | None
    ) -> tuple[CombinedDataset, int] | None:
        """Load the dataset for the given dataloader index."""
        if dataloader is None or self.cached_dataloader_idx_ is None:
            return None
        dataloader = (
            dataloader[self.cached_dataloader_idx_]
            if isinstance(dataloader, Sequence)
            else dataloader
        )
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        if not isinstance(dataset, CombinedDataset):
            log.warning("Dataset is of type %s not CombinedDataset", type(dataset))
            return None
        if batch_size is None:
            log.warning("Dataloader does not have a batch size.")
            return None
        return (dataset, batch_size)

    def load_target_uncertainties(
        self, dataset: CombinedDataset, dates: list
    ) -> dict[int, ArrayTHW]:
        """Load uncertainty at the same normalised scale as the target."""
        try:
            uncertainties: dict[int, ArrayTHW] = {}
            for (
                target_variable,
                uncertainty_variable,
            ) in self._plot_spec.uncertainty_variables.items():
                # Attempt to load target index from the dataset
                if target_variable not in dataset.target.variable_names:
                    continue
                target_idx = dataset.target.variable_names.index(target_variable)

                # Attempt to load uncertainty from the dataset
                source = next(
                    (
                        input_ds
                        for input_ds in dataset.inputs
                        if input_ds.name == dataset.target.name
                        and uncertainty_variable in input_ds.variable_names
                    ),
                    None,
                )
                if source is None:
                    continue
                uncertainty_ds = source.subset(
                    variables=[uncertainty_variable], normalise=False
                )

                # Load uncertainties as fractions which must be in the range (0, 1]
                np_dates = [npdatetime_from_datetime(date) for date in dates]
                uncertainty = uncertainty_ds.get_tchw(np_dates)[:, 0]
                uncertainty = np.where(
                    np.isfinite(uncertainty) & (uncertainty > 0) & (uncertainty <= 1),
                    uncertainty,
                    np.nan,
                )

                # Scale uncertainties to the same range as the target, or return empty
                target_min = float(dataset.target.statistics["minimum"][target_idx])
                target_max = float(dataset.target.statistics["maximum"][target_idx])
                target_range = target_max - target_min
                if not np.isfinite(target_range) or target_range <= 0:
                    log.warning(
                        "Could not scale target uncertainty because target range is %s.",
                        target_range,
                    )
                    continue
                uncertainties[target_idx] = (uncertainty / target_range).astype(
                    np.float32
                )
        except (
            IndexError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError,
            MemoryError,
            OSError,
        ) as exc:
            log.warning("Could not load target uncertainty: %s", exc)
            return {}
        else:
            return uncertainties

    def make_plots(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        dataset: CombinedDataset,
        batch_size: int,
    ) -> None:
        # Ensure the module is a BaseModel
        if not isinstance(pl_module, BaseModel):
            msg = f"Lightning module is of type {type(pl_module)}, skipping plotting."
            log.warning(msg)
            return

        # Ensure that outputs is a ModelStepOutput
        if self.cached_outputs_ is None or self.cached_batch_idx_ is None:
            log.warning("Could not load outputs, skipping plotting.")
            return

        # Load land mask for plotting based on dataset
        datamodule = getattr(trainer, "datamodule", None)
        mask_directory = getattr(datamodule, "mask_directory", None)
        land_mask_path = mask_directory / "land_mask.npy" if mask_directory else None
        if land_mask_path not in self._land_mask_cache:
            self._land_mask_cache[land_mask_path] = LandMask(land_mask_path)

        # Use checkpoint epoch during testing and current epoch otherwise
        epoch = pl_module.checkpoint_epoch if trainer.testing else trainer.current_epoch

        # Construct a publisher to handle the actual plotting and logging of media.
        publisher = MediaPublisher(
            dataset=getattr(datamodule, "training_dataset", dataset),
            land_mask=self._land_mask_cache[land_mask_path],
            model_name=self._model_name,
            plot_spec=replace(self._plot_spec, hemisphere=pl_module.hemisphere),
            trained_epochs=None if epoch is None else epoch + 1,
        )

        # Load dates from the dataset
        start_date = dataset.dates[batch_size * self.cached_batch_idx_]
        history_dates = list(
            map(datetime_from_npdatetime, dataset.get_history_steps(start_date))
        )
        forecast_dates = list(
            map(datetime_from_npdatetime, dataset.get_forecast_steps(start_date))
        )

        # Get loggers that support image and video logging
        image_loggers: list[SupportsImageLogging] = [
            ll for ll in trainer.loggers if isinstance(ll, SupportsImageLogging)
        ]
        video_loggers: list[SupportsVideoLogging] = [
            ll for ll in trainer.loggers if isinstance(ll, SupportsVideoLogging)
        ]

        # Get channel names from the model
        channel_names = getattr(pl_module, "channel_names", ["sea-ice-concentration"])

        # Load uncertainties
        uncertainties = self.load_target_uncertainties(dataset, forecast_dates)
        climatology_tchw = dataset.climatology_for(start_date)

        if self.make_static_plots:
            publisher.log_static_outputs(
                self.cached_outputs_,
                image_loggers,
                channel_names=channel_names,
                climatology=climatology_tchw,
                forecast_dates=forecast_dates,
                history_dates=history_dates,
                compare_truth_climatology=(
                    self.make_input_plots
                    and not self.is_logged("climatology_static", forecast_dates)
                ),
                prefix=self.prefix,
                uncertainties=uncertainties,
            )
            if self.make_input_plots and not self.is_logged("static", forecast_dates):
                publisher.log_static_inputs(
                    dataset.inputs, forecast_dates, image_loggers, prefix=self.prefix
                )

        if self.make_video_plots:
            publisher.log_video_outputs(
                self.cached_outputs_,
                video_loggers,
                channel_names=channel_names,
                climatology=climatology_tchw,
                forecast_dates=forecast_dates,
                history_dates=history_dates,
                compare_truth_climatology=(
                    self.make_input_plots
                    and not self.is_logged("climatology_video", forecast_dates)
                ),
                prefix=self.prefix,
                uncertainties=uncertainties,
            )
            if self.make_input_plots and not self.is_logged("video", forecast_dates):
                publisher.log_video_inputs(
                    dataset.inputs, forecast_dates, video_loggers, prefix=self.prefix
                )

    def on_batch_end(  # noqa: PLR0913
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch_idx: int,
        dataloader_idx: int,
        *,
        is_last_batch: bool,
        num_batches: float,
        dataloaders: DataLoader | list[DataLoader] | None,
    ) -> None:
        """Shared test/validation batch-end handling: cache and maybe plot."""
        # Check whether this is a batch we want to plot based on the frequency settings
        is_per_epoch = is_last_batch
        is_per_batch = self.frequency_batch > 0 and not batch_idx % self.frequency_batch
        is_sampled_batch = self.is_sample_batch(batch_idx, num_batches)

        # Cache if this is a batch we want to plot
        if is_per_epoch or is_per_batch or is_sampled_batch:
            self.cache_batch(batch_idx, dataloader_idx, outputs)

        # If this is a selected batch then we will plot here
        if is_per_batch or is_sampled_batch:
            # Load the dataset
            if not (ds_tuple := self.load_dataset(dataloaders)):
                log.warning("Could not load dataset, skipping plotting.")
                return

            # Make the plots
            self.make_plots(trainer, pl_module, *ds_tuple)

    def on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloaders: DataLoader | list[DataLoader] | None,
    ) -> None:
        """Shared test/validation epoch-end handling: maybe plot."""
        # Only run plotting if this batch is at the specified frequency
        if self.frequency_epoch <= 0 or trainer.current_epoch % self.frequency_epoch:
            return

        # Load the dataset
        if not (ds_tuple := self.load_dataset(dataloaders)):
            log.warning("Could not load dataset, skipping plotting.")
            return

        # Make the plots
        self.make_plots(trainer, pl_module, *ds_tuple)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each test batch."""
        self.on_batch_end(
            trainer,
            pl_module,
            outputs,
            batch_idx,
            dataloader_idx,
            is_last_batch=trainer.is_last_batch,
            num_batches=trainer.num_test_batches[dataloader_idx],
            dataloaders=trainer.test_dataloaders,
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of each test epoch."""
        self.on_epoch_end(trainer, pl_module, trainer.test_dataloaders)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each validation batch."""
        # Ignore the initial sanity checking run
        if trainer.sanity_checking:
            return

        self.on_batch_end(
            trainer,
            pl_module,
            outputs,
            batch_idx,
            dataloader_idx,
            is_last_batch=trainer.fit_loop.epoch_loop.val_loop.batch_progress.is_last_batch,
            num_batches=trainer.num_val_batches[dataloader_idx],
            dataloaders=trainer.val_dataloaders,
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called at the end of each validation epoch."""
        self.on_epoch_end(trainer, pl_module, trainer.val_dataloaders)
