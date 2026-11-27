from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from cryocast.types import TensorNTCHW

from .training_objective import TrainingObjective

if TYPE_CHECKING:
    from cryocast.models.encode_process_decode import EncodeProcessDecode


class PhysicalObjective(TrainingObjective):
    """Compare the decoded forecast to the target in physical space.

    This is the standard objective. The training step runs the forward pass, so the
    loss trains the encoders, the processor and the decoder together.
    """

    @override
    def training_step(
        self,
        model: "EncodeProcessDecode",
        batch: dict[str, TensorNTCHW],
        target: TensorNTCHW,
    ) -> tuple[TensorNTCHW, torch.Tensor]:
        prediction = model(batch)
        return prediction, model.loss(prediction, target)
