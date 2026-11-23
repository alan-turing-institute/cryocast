from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_multi_avg_fn


class EMAWeightAveragingCallback(WeightAveraging):
    """A callback that updates an averaged model for Exponential Moving Average (EMA) after each training step."""

    def __init__(
        self,
        *,
        decay_rate: float,
        every_n_epochs: int | None = None,
        every_n_steps: int | None = None,
    ) -> None:
        """Summarise metrics during evaluation.

        Args:
            decay_rate: Parameter update decay rate.
            every_n_epochs: How many epochs to wait before updating.
            every_n_steps: How many steps to wait before updating.

        """
        super().__init__(
            multi_avg_fn=get_ema_multi_avg_fn(decay_rate), use_buffers=True
        )
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *args: Any, **kwargs: Any
    ) -> None:
        """Ignore the update if the module has no parameters."""
        if next(pl_module.parameters(), None) is not None:
            super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Ignore the update if the module has no parameters."""
        if next(pl_module.parameters(), None) is not None:
            super().on_train_epoch_end(trainer, pl_module)

    def should_update(
        self, step_idx: int | None = None, epoch_idx: int | None = None
    ) -> bool:
        """Update if we are at the requested number of steps or epochs."""
        if self.every_n_epochs and epoch_idx:
            return epoch_idx % self.every_n_epochs == 0

        if self.every_n_steps and step_idx:
            return step_idx % self.every_n_steps == 0

        return False
