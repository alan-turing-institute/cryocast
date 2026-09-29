import logging
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, ClassVar, override

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.trainer.states import TrainerFn
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from icenet_mp.metrics import (
    FractionalSkillScorePerForecastDay,
    SpatialMeanGroundTruthPerForecastDay,
    SpatialMeanPredictionPerForecastDay,
)
from icenet_mp.utils import get_wandb_run

logger = logging.getLogger(__name__)


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics at the end of an epoch or a run."""

    STAGES_BY_TRAINER_FN: ClassVar[dict[str, list[str]]] = {
        TrainerFn.FITTING.value: ["train", "validation"],
        TrainerFn.TESTING.value: ["test"],
    }

    # Metric types whose per-forecast-day values share a single plot
    PLOT_GROUPS: ClassVar[dict[type[Metric], str]] = {
        FractionalSkillScorePerForecastDay: "fss",
        SpatialMeanGroundTruthPerForecastDay: "spatial_mean",
        SpatialMeanPredictionPerForecastDay: "spatial_mean",
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

    def _collect_per_run_values(
        self, metrics: dict[str, MetricCollection]
    ) -> tuple[dict[str, dict[str, dict[str, Tensor]]], dict[int, dict[str, float]]]:
        """Collect the values needed for the per-run plots.

        Only metrics with a value for each forecast day are included. Grouping uses
        metric types rather than names, since names are user-configurable.

        Returns:
            The per-forecast-day values of each metric, as ``{group: {metric name:
            {stage: values}}}``, and the mean FSS of each stage at each neighbourhood
            size, as ``{neighbourhood size: {stage: mean}}``.

        """
        values_by_group: dict[str, dict[str, dict[str, Tensor]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        fss_means_by_size: dict[int, dict[str, float]] = defaultdict(dict)
        for stage, metric_collection in metrics.items():
            for name, metric in metric_collection.items():
                if not metric.update_called:
                    continue
                values = metric.compute()
                if values.numel() <= 1:
                    continue
                group = next(
                    (
                        group
                        for metric_type, group in self.PLOT_GROUPS.items()
                        if isinstance(metric, metric_type)
                    ),
                    name,
                )
                values_by_group[group][name][stage] = values
                if isinstance(metric, FractionalSkillScorePerForecastDay):
                    fss_means_by_size[metric.neighbourhood_size][stage] = (
                        values.mean().item()
                    )
        return values_by_group, fss_means_by_size

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

        values_by_group, fss_means_by_size = self._collect_per_run_values(metrics)

        # One per-forecast-day plot for each group. Legend keys are the stage for a
        # single metric, otherwise the metric name (prefixed by stage if there are
        # several stages, with any redundant group prefix removed).
        plots: dict[str, Any] = {}
        for group, values_by_name in values_by_group.items():
            series: dict[str, list[float]] = {}
            for name, values_by_stage in values_by_name.items():
                for stage, values in values_by_stage.items():
                    if len(values_by_name) == 1:
                        key = stage
                    elif len(metrics) == 1:
                        key = name
                    else:
                        key = f"{stage}_{name.removeprefix(f'{group}_')}"
                    series[key] = values.tolist()
            plot_name = f"{group}_per_forecast_day"
            plots[plot_name] = wandb.plot.line_series(
                xs=list(range(1, len(next(iter(series.values()))) + 1)),
                ys=list(series.values()),
                keys=list(series),
                title=plot_name,
                xname="day",
            )

        # Mean FSS against neighbourhood size, with one line per stage
        if fss_means_by_size:
            sizes = sorted(fss_means_by_size)
            stages = list(fss_means_by_size[sizes[0]])
            plot_name = "fss_vs_neighbourhood_size"
            plots[plot_name] = wandb.plot.line_series(
                xs=sizes,
                ys=[
                    [fss_means_by_size[size][stage] for size in sizes]
                    for stage in stages
                ],
                keys=stages,
                title=plot_name,
                xname="neighbourhood_size",
            )

        if plots:
            run.log(plots)

    def _on_epoch_start(self, pl_module: LightningModule, stage: str) -> None:
        """Reset a stage's metrics collection, if present."""
        metrics = getattr(pl_module, f"{stage}_metrics", None)
        if isinstance(metrics, MetricCollection):
            metrics.reset()

    def _on_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Log a stage's per-epoch metrics, warning if the collection is missing."""
        metrics = getattr(pl_module, f"{stage}_metrics", None)
        if isinstance(metrics, MetricCollection):
            self.log_per_epoch_metrics(trainer, metrics, stage=stage)
        else:
            logger.warning("Could not load %s metrics!", stage)

    @override
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the start of a test epoch."""
        self._on_epoch_start(pl_module, "test")
        if self.climatology_metrics is not None:
            self.climatology_metrics.reset()

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
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

    @override
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a test epoch."""
        self._on_epoch_end(trainer, pl_module, "test")

    @override
    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Called at the start of a training epoch."""
        self._on_epoch_start(pl_module, "train")

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a training epoch."""
        self._on_epoch_end(trainer, pl_module, "train")

    @override
    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Called at the start of a validation epoch."""
        self._on_epoch_start(pl_module, "validation")

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called at the end of a validation epoch."""
        self._on_epoch_end(trainer, pl_module, "validation")

    @override
    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Called at the end of a run: log train/validation or test metrics."""
        metrics = {}

        for run_stage in self.STAGES_BY_TRAINER_FN.get(stage, []):
            collection = getattr(pl_module, f"{run_stage}_metrics", None)
            if isinstance(collection, MetricCollection):
                metrics[run_stage] = collection
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
