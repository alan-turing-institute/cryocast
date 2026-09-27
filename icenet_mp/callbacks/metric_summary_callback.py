import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, ClassVar

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.trainer.states import TrainerFn
from torch import Tensor
from torchmetrics import MetricCollection

from icenet_mp.utils import get_wandb_run

logger = logging.getLogger(__name__)


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics at the end of an epoch or a run."""

    _STAGES_BY_TRAINER_FN: ClassVar[dict[str, list[str]]] = {
        TrainerFn.FITTING.value: ["train", "validation"],
        TrainerFn.TESTING.value: ["test"],
    }

    def __init__(self, climatology_metrics: MetricCollection | None = None) -> None:
        """Initialise a MetricSummaryCallback.

        Args:
            climatology_metrics: Optional metric collection for accumulating climatology
                baseline metrics during a test run. When omitted, the collection is
                created lazily on the first test batch containing a ``climatology``
                entry, mirroring the model's test metrics.

        """
        super().__init__()
        self.climatology_metrics = climatology_metrics

    @staticmethod
    def _series_key(
        stage: str,
        metric_name: str,
        group_name: str,
        *,
        grouped: bool,
        multiple_stages: bool,
    ) -> str:
        """Series key for a per-forecast-day plot that is used by W&B legend."""
        if not grouped:
            return stage
        if not multiple_stages:
            return metric_name
        suffix = metric_name.removeprefix(f"{group_name}_")
        return f"{stage}_{suffix}"

    def _collect_values_per_forecast_day(
        self, metrics: dict[str, MetricCollection]
    ) -> dict[str, dict[str, Tensor]]:
        """Collect metric values that have a value for each forecast day."""
        values_per_forecast_day: dict[str, dict[str, Tensor]] = defaultdict(dict)
        for stage, metric_collection in metrics.items():
            for metric_name, metric in metric_collection.items():
                if not metric.update_called:
                    continue
                metric_tensor: Tensor = metric.compute()
                if metric_tensor.numel() > 1:
                    values_per_forecast_day[metric_name][stage] = metric_tensor
        return values_per_forecast_day

    def _metrics_for_stage(
        self, pl_module: LightningModule, stage: str
    ) -> MetricCollection | None:
        """Return a stage's metrics collection off pl_module, or None if unavailable."""
        metrics = getattr(pl_module, f"{stage}_metrics", None)
        return metrics if isinstance(metrics, MetricCollection) else None

    def _per_forecast_day_plots(
        self,
        values_per_forecast_day: dict[str, dict[str, Tensor]],
        *,
        multiple_stages: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build a per-forecast-day plot for each metric group.

        Metrics that should share a single plot are grouped; all other metrics get one
        plot each. Returns the plots keyed by name, and the metric names in the group,
        for use in the FSS-vs-neighbourhood-size plot.
        """
        # Group any metrics that belong to common groups (e.g. FSS and spatial mean)
        metric_names_by_group: dict[str, list[str]] = defaultdict(list)
        for metric_name in values_per_forecast_day:
            if metric_name.startswith("fss_"):
                metric_names_by_group["fss"].append(metric_name)
            elif metric_name.startswith("spatial_mean_"):
                metric_names_by_group["spatial_mean"].append(metric_name)
            else:
                metric_names_by_group[metric_name].append(metric_name)

        plots: dict[str, Any] = {}
        for group_name, metric_names in metric_names_by_group.items():
            grouped = len(metric_names) > 1
            series: dict[str, Tensor] = {
                self._series_key(
                    stage,
                    metric_name,
                    group_name,
                    grouped=grouped,
                    multiple_stages=multiple_stages,
                ): tensor
                for metric_name in metric_names
                for stage, tensor in values_per_forecast_day[metric_name].items()
            }

            keys = list(series.keys())
            days = list(range(1, len(series[keys[0]]) + 1))
            plot_name = f"{group_name}_per_forecast_day"
            plots[plot_name] = wandb.plot.line_series(
                xs=days,
                ys=[series[key].tolist() for key in keys],
                keys=keys,
                title=plot_name,
                xname="day",
            )

        return plots, metric_names_by_group.get("fss", [])

    def _fss_vs_neighbourhood_size_plot(
        self,
        values_per_forecast_day: dict[str, dict[str, Tensor]],
        fss_metric_names: list[str],
    ) -> dict[str, Any]:
        """Build a plot of mean FSS (over forecast days) against neighbourhood size."""
        sizes_and_names = sorted(
            (int(match.group(1)), name)
            for name in fss_metric_names
            if (match := re.match(r"^fss_neighbourhood_size_(\d+)$", name))
        )
        if not sizes_and_names:
            return {}

        stages = list(values_per_forecast_day[sizes_and_names[0][1]])
        sizes = [size for size, _ in sizes_and_names]
        ys = [
            [
                values_per_forecast_day[name][stage].mean().item()
                for _, name in sizes_and_names
            ]
            for stage in stages
        ]

        plot_name = "fss_vs_neighbourhood_size"
        return {
            plot_name: wandb.plot.line_series(
                xs=sizes,
                ys=ys,
                keys=stages,
                title=plot_name,
                xname="neighbourhood_size",
            )
        }

    def log_per_epoch_metrics(
        self, trainer: Trainer, metrics: MetricCollection, stage: str
    ) -> None:
        """Log per-epoch metrics to W&B."""
        # Skip logging during sanity checking to avoid logging incomplete metrics
        if trainer.sanity_checking:
            return

        # Compute the mean value of each metric (e.g., SIEError) across all days
        means = {
            f"{stage}_{name}_mean".lower(): metric.compute().mean().item()
            for name, metric in metrics.items()
            if metric.update_called
        }
        if not means:
            return

        for logger_ in trainer.loggers:
            logger_.log_metrics({**means, "epoch": trainer.current_epoch})

    def log_per_run_metrics(
        self, trainer: Trainer, metrics: dict[str, MetricCollection]
    ) -> None:
        """Log per-run metrics to W&B.

        Note that these will be based on metrics accumulated during the final epoch, due
        to the reset behaviour in log_per_epoch_metrics.
        """
        # Skip logging during sanity checking to avoid logging incomplete metrics
        if trainer.sanity_checking:
            return

        # Check that W&B is being used as a logger
        if not isinstance(run := get_wandb_run(trainer), wandb.Run):
            logger.warning(
                "W&B is not being used as a logger, cannot log per-run metrics!"
            )
            return

        # Extract the metric values (e.g., SIEError) across all batches
        # Only consider metrics that have a value for each forecast day
        values_per_forecast_day = self._collect_values_per_forecast_day(metrics)

        plots, fss_metric_names = self._per_forecast_day_plots(
            values_per_forecast_day, multiple_stages=len(metrics) > 1
        )
        plots.update(
            self._fss_vs_neighbourhood_size_plot(
                values_per_forecast_day, fss_metric_names
            )
        )
        if plots:
            run.log(plots)

    def _on_epoch_start(self, pl_module: LightningModule, stage: str) -> None:
        """Reset a stage's metrics collection, if present."""
        if (metrics := self._metrics_for_stage(pl_module, stage)) is not None:
            metrics.reset()

    def _on_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Log a stage's per-epoch metrics, warning if the collection is missing."""
        if (metrics := self._metrics_for_stage(pl_module, stage)) is not None:
            self.log_per_epoch_metrics(trainer, metrics, stage=stage)
        else:
            logger.warning("Could not load %s metrics!", stage)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        """Called at the start of a test epoch."""
        self._on_epoch_start(pl_module, "test")
        if self.climatology_metrics is not None:
            self.climatology_metrics.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Called at the end of each test batch.

        Accumulate climatology baseline metrics when the batch contains a
        ``climatology`` entry. The target is read from the model outputs, as the test
        step has already popped it from the batch.
        """
        if (
            not isinstance(batch, Mapping)
            or "climatology" not in batch
            or not isinstance(pl_module.test_metrics, MetricCollection)
            or not isinstance(outputs, Mapping)
            or "target" not in outputs
        ):
            return
        if self.climatology_metrics is None:
            self.climatology_metrics = MetricCollection(
                {
                    name: deepcopy(metric)
                    for name, metric in pl_module.test_metrics.items()
                }
            )
            # pl_module.test_metrics has already been updated with this batch's model
            # prediction by test_step (which runs before this hook), so the deep-copied
            # metrics start out contaminated with that update. Reset before accumulating
            # the climatology baseline so the two series stay independent.
            self.climatology_metrics.reset()
        self.climatology_metrics.update(batch["climatology"], outputs["target"])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a test epoch."""
        self._on_epoch_end(trainer, pl_module, "test")

    def on_train_epoch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Called at the start of a training epoch."""
        self._on_epoch_start(pl_module, "train")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a training epoch."""
        self._on_epoch_end(trainer, pl_module, "train")

    def on_validation_epoch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Called at the start of a validation epoch."""
        self._on_epoch_start(pl_module, "validation")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called at the end of a validation epoch."""
        self._on_epoch_end(trainer, pl_module, "validation")

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Called at the end of a run: log train/validation or test metrics."""
        metrics = {}

        for run_stage in self._STAGES_BY_TRAINER_FN.get(stage, []):
            if (m := self._metrics_for_stage(pl_module, run_stage)) is not None:
                metrics[run_stage] = m
            else:
                logger.warning("Could not load %s metrics!", run_stage)

        # Include the climatology baseline when it was accumulated during testing
        if (
            stage == TrainerFn.TESTING.value
            and self.climatology_metrics is not None
            and any(
                metric.update_called for metric in self.climatology_metrics.values()
            )
        ):
            metrics["climatology"] = self.climatology_metrics

        self.log_per_run_metrics(trainer, metrics)
