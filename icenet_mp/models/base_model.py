import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
    OptimizerConfig,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection
from typing_extensions import override

from icenet_mp.losses import build_loss
from icenet_mp.metrics import LandMaskMixin, SingleChannelMetricMixin
from icenet_mp.models.common import Mask
from icenet_mp.types import (
    DataSpace,
    Hemisphere,
    MaskType,
    ModelStepOutput,
    TensorNTCHW,
)

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys
    from torch.optim import Optimizer

log = logging.getLogger(__name__)


class BaseModel(LightningModule, ABC):
    """A base class for all models used in the IceNet-MP project."""

    # Parameters that should be excluded from hyperparameter logging
    ignored_hparams: ClassVar[frozenset[str]] = frozenset(
        ("latitudes_fn", "longitudes_fn", "mask_dir", "metrics")
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        channel_names: Sequence[str] | None = None,
        hemisphere: Hemisphere,
        input_spaces: Sequence[DictConfig],
        latitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        longitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        loss: DictConfig,
        mask_dir: str | Path | None = None,
        lr_scheduler: DictConfig,
        metrics: Sequence[Mapping[str, Any]],
        n_forecast_steps: int,
        n_history_steps: int,
        name: str,
        optimizer: DictConfig,
        output_space: DictConfig,
        scheduler: DictConfig,
        **kwargs: Any,
    ) -> None:
        """Initialise a BaseModel.

        Input spaces and the desired output space must be specified, as must the number
        of forecast and history steps.

        Optimizer configuration is also set here.

        ``mask_dir``, if given, is a directory holding `land_mask.npy` (generated for
        SSMIS datasets by `datasets create`). When present, the ``"diiee"``/``"fss_*"``
        metrics use it to exclude land/ice boundaries from ice-edge detection, so only
        ocean ice/no-ice transitions count as the sea-ice edge.

        ``metrics`` is the list of metrics to compute during training, validation, and
        testing. A TypeError is raised if an entry is not a mapping with string
        ``name`` and ``_target_`` keys, and a ValueError if a name is used more than
        once.
        """
        super().__init__(**kwargs)

        # Save model name, hemisphere, lat/lon information and channel names
        self.name = name
        self.hemisphere: Hemisphere = hemisphere
        self.latitudes_fn = latitudes_fn
        self.longitudes_fn = longitudes_fn
        self.channel_names = list(channel_names) if channel_names else []

        # Number of epochs in the checkpoint this model was loaded from, if any
        self.checkpoint_epoch: int | None = None

        # Save history and forecast steps
        if n_forecast_steps <= 0:
            msg = "Number of forecast steps must be greater than 0."
            raise ValueError(msg)
        self.n_forecast_steps = n_forecast_steps
        if n_history_steps <= 0:
            msg = "Number of history steps must be greater than 0."
            raise ValueError(msg)
        self.n_history_steps = n_history_steps

        # Construct the input and output spaces
        self.input_spaces = [DataSpace.from_dict(space) for space in input_spaces]
        self.output_space = DataSpace.from_dict(output_space)

        # Store the optimizer, scheduler and loss configs
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.lr_scheduler_cfg = lr_scheduler
        self.loss_cfg = loss

        # Validate and store the metric configs
        self.metric_cfgs: dict[str, dict[str, Any]] = {}
        for metric_cfg in metrics:
            if not (
                isinstance(metric_cfg, Mapping)
                and isinstance(metric_name := metric_cfg.get("name"), str)
                and isinstance(metric_cfg.get("_target_"), str)
            ):
                msg = (
                    f"Metric config {metric_cfg!r} must be a mapping with 'name' and "
                    "'_target_' keys, e.g. {'name': 'mae', '_target_': "
                    "'my_package.my_metric.MyMetricClass'}."
                )
                raise TypeError(msg)
            if metric_name in self.metric_cfgs:
                msg = f"Metric name {metric_name!r} is configured more than once."
                raise ValueError(msg)
            self.metric_cfgs[metric_name] = dict(metric_cfg)

        # Land mask for ice-edge metrics (excludes land/ice boundaries from FSS/DIIEE).
        try:
            land_mask = Mask(
                mask_type=MaskType.LAND,
                output_shape=self.output_space.shape,
                mask_dir=mask_dir,
            ).mask
        except FileNotFoundError:
            land_mask = None

        # Build test/train/validation metrics
        self.test_metrics = self.build_metrics(land_mask)
        self.train_metrics = self.build_metrics(land_mask)
        self.validation_metrics = self.build_metrics(land_mask)
        # Climatology baseline metrics, used if there is a climatology batch in testing
        self.climatology_metrics = self.build_metrics(land_mask)
        if skipped := [met for met in self.metric_cfgs if met not in self.test_metrics]:
            log.warning(
                "Disabling single-channel metrics for %s (%d output channels): %s.",
                type(self).__name__,
                self.output_space.channels,
                ", ".join(skipped),
            )

        # All arguments to the ultimate child class will be logged as hyperparameters,
        # and saved to W&B, unless explicitly ignored here.
        self.save_hyperparameters(ignore=[*self.ignored_hparams])

    @cached_property
    def latitudes(self) -> dict[str, list[float]]:
        return {} if not self.latitudes_fn else self.latitudes_fn()

    @cached_property
    def longitudes(self) -> dict[str, list[float]]:
        return {} if not self.longitudes_fn else self.longitudes_fn()

    @property
    def multistage_only(self) -> bool:
        return False

    def build_metrics(self, land_mask: torch.Tensor | None) -> MetricCollection:
        """Build a metric collection from the configured metrics.

        Each configured metric is a mapping with a ``name`` (its key in logs), a Hydra
        ``_target_`` and any other constructor arguments, e.g.
        ``{"name": "my_metric", "_target_": "my_package.MyMetric", "k": 3}``. Targets
        that subclass `LandMaskMixin` are also given the land mask.

        This should include only metrics that are compatible with the output space. We
        therefore filter out single-channel metrics from the metric collection if the
        model will predict multiple channels.

        Args:
            land_mask: Optional boolean tensor of shape (H, W), True for ocean cells and
                       False for land. This is passed to every `LandMaskMixin`
                       metric, and is used to exclude land cells from the metric.

        Returns:
            A MetricCollection containing the requested metrics.

        """
        requested: dict[str, Metric] = {}
        for name, metric_cfg in self.metric_cfgs.items():
            # Build fresh arguments, so the stored config is never modified
            kwargs = {key: value for key, value in metric_cfg.items() if key != "name"}
            target = hydra.utils.get_object(kwargs["_target_"])
            # Only metrics that use a land mask are given one
            if isinstance(target, type) and issubclass(target, LandMaskMixin):
                kwargs["land_mask"] = land_mask
            requested[name] = hydra.utils.instantiate(kwargs)
        # Disable compute groups to avoid erroneous automated groupings
        return MetricCollection(
            {
                name: metric
                for name, metric in requested.items()
                if self.output_space.channels == 1
                or not isinstance(metric, SingleChannelMetricMixin)
            },
            compute_groups=False,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct the optimizer and optional scheduler from the config."""
        # Create the optimizer
        optimizer: Optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=filter(lambda p: p.requires_grad, self.parameters()),
        )

        # If no scheduler config is provided, return just the optimizer
        if not self.scheduler_cfg:
            return OptimizerConfig(optimizer=optimizer)

        # Create the scheduler
        scheduler: LRSchedulerTypeUnion = hydra.utils.instantiate(
            self.scheduler_cfg, optimizer=optimizer
        )

        # Create the Lightning LRScheduler wrapper
        lr_scheduler = LRSchedulerConfigType(
            frequency=self.lr_scheduler_cfg.get("frequency", 1),
            interval=self.lr_scheduler_cfg.get("interval", "epoch"),
            monitor=self.lr_scheduler_cfg.get("monitor"),
            name=self.lr_scheduler_cfg.get("name"),
            reduce_on_plateau=self.lr_scheduler_cfg.get("reduce_on_plateau", False),
            scheduler=scheduler,
            strict=self.lr_scheduler_cfg.get("strict", True),
        )

        # Return the optimizer and scheduler
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    @abstractmethod
    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - start with multiple `NTCHW` inputs, one for each input dataset
        - return a single `NTCHW` output representing the predicted output

        Args:
            inputs: Dictionary of dataset name to TensorNTCHW with shape (batch, n_history_steps, C_input_k, H_input_k, W_input_k)

        Returns:
            Predicted TensorNTCHW with shape (batch, n_forecast_steps, C_output, H_output, W_output)

        """

    @override
    def load_state_dict(
        self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> "_IncompatibleKeys":
        """Load a state dict, ignoring any metric collections it contains."""
        metric_prefixes = tuple(
            f"{name}."
            for name, module in self.named_modules()
            if isinstance(module, MetricCollection)
        )
        return super().load_state_dict(
            {k: v for k, v in state_dict.items() if not k.startswith(metric_prefixes)},
            *args,
            **kwargs,
        )

    def loss(self, prediction: TensorNTCHW, target: TensorNTCHW) -> torch.Tensor:
        """Calculate the loss given a prediction and target."""
        return self.loss_fn(prediction, target)

    @property
    def loss_cfg(self) -> DictConfig:
        """Get the loss configuration."""
        return self._loss_cfg

    @loss_cfg.setter
    def loss_cfg(self, cfg: DictConfig) -> None:
        """Set the loss configuration and instantiate the loss function.

        If a `lead_time_exponent` key is present, then the loss function will be wrapped
        in a LeadTimeWeightedLoss with that exponent.
        """
        self._loss_cfg = cfg
        self.loss_fn: torch.nn.Module = build_loss(cfg)

    def process_batch(self, batch: dict[str, TensorNTCHW]) -> dict[str, TensorNTCHW]:
        """Process a batch before the forward pass and loss computation.

        Subclasses can override this to extract or transform inputs before the standard
        training/validation steps. The returned dict must include a ``"target"`` key.
        """
        return batch

    def test_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,  # noqa: PT019
    ) -> ModelStepOutput:
        """Run the test step, in PyTorch eval model (i.e. no gradients).

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Update the test metrics (and climatology metrics if appropriate)
        - Return the prediction, target and loss

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A ModelStepOutput containing the prediction, target and loss for the batch.

        """
        batch = self.process_batch(batch)
        target = batch.pop("target")
        prediction = self(batch)
        loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.test_metrics.update(prediction, target)
        if "climatology" in batch:
            self.climatology_metrics.update(batch["climatology"], target)

        return ModelStepOutput(prediction, target, loss)

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the training step.

        - Separate the batch into inputs and target
        - Run inputs and target through the model
        - Calculate the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A ModelStepOutput containing the prediction, target and loss for the batch.

        """
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()
        prediction = self(batch)
        loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)

    def validation_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the validation step.

        A batch contains one tensor for each input dataset and one for the target
        These are [NTCHW] tensors with (batch_size, n_history_steps, C, H, W)

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Calculate and log the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A ModelStepOutput containing the prediction, target and loss for the batch.

        """
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()
        prediction = self(batch)
        loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.validation_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
