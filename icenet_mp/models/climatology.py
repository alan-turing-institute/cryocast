from typing import Any

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from icenet_mp.types import TensorNTCHW

from .base_model import BaseModel


class Climatology(BaseModel):
    def __init__(
        self,
        target_variable_indices: list[int],
        **kwargs: Any,
    ) -> None:
        """Initialise a Climatology model.

        Args:
            target_variable_indices: The indices of the target variables.
            kwargs: Additional arguments passed to the base model.

        """
        super().__init__(**kwargs)
        self.automatic_optimization = False
        self.model = nn.Identity()
        self.variable_indices = target_variable_indices

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Climatology model does not need an optimizer."""
        return None

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        Return the climatology (calendar-day mean) field for each forecast step,
        supplied by the data loader under the ``climatology`` key with shape
        [batch, n_forecast_steps, C_target, H_target, W_target].
        """
        return inputs["climatology"]
