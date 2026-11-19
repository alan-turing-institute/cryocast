from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch

from cryocast.types import TensorNTCHW

if TYPE_CHECKING:
    from cryocast.models.encode_process_decode import EncodeProcessDecode


class TrainingObjective(ABC):
    """How an EncodeProcessDecode model calculates its training loss.

    The model owns the objective. The objective keeps no reference to the model: the
    model gives itself to each method. A subclass that must check or change the model
    does so in its constructor, so that no objective exists without these steps. The
    objective has no parameters, so it does not change the model state dict.
    """

    # Whether training batches must hold `future_<input>` keys
    needs_future_inputs: ClassVar[bool] = False

    def verify_single_stage_trainable(self) -> None:  # noqa: B027
        """Check that single-stage training can train the model with this objective.

        Raises:
            ValueError: If single-stage training cannot train the model.

        """

    @abstractmethod
    def training_step(
        self,
        model: "EncodeProcessDecode",
        batch: dict[str, TensorNTCHW],
        target: TensorNTCHW,
    ) -> tuple[TensorNTCHW, torch.Tensor]:
        """Calculate the training prediction and loss for one batch.

        Args:
            model: The model that owns this objective.
            batch: Processed batch, see `EncodeProcessDecode.training_step`.
            target: Detached TensorNTCHW target with shape (batch, n_forecast_steps,
                n_output_channels, H_output, W_output).

        Returns:
            The prediction, with the same shape as the target, and the scalar loss.

        """
