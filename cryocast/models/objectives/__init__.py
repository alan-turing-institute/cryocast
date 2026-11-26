import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .latent_objective import LatentObjective
from .physical_objective import PhysicalObjective
from .training_objective import TrainingObjective

if TYPE_CHECKING:
    from cryocast.models.encode_process_decode import EncodeProcessDecode

log = logging.getLogger(__name__)


def build_training_objective(
    model: "EncodeProcessDecode", loss_weights: Mapping[str, float] | None = None
) -> TrainingObjective:
    """Choose the training objective for the processor of a model.

    Processors that set `computes_loss_in_latent_space=True` use a LatentObjective.
    Other processors use a PhysicalObjective, which ignores the loss weights.

    Args:
        model: The model that owns the objective. Its processor must exist.
        loss_weights: Weights for the auxiliary loss terms, see `LatentObjective`.

    Raises:
        ValueError: If the LatentObjective cannot train the model.

    """
    processor = model.processor
    if processor.computes_loss_in_latent_space:
        return LatentObjective(model, loss_weights)
    if loss_weights and any(loss_weights.values()):
        log.warning(
            "loss_weights has no effect, because processor %s does not compute its "
            "loss in latent space.",
            type(processor).__name__,
        )
    return PhysicalObjective()


__all__ = [
    "LatentObjective",
    "PhysicalObjective",
    "TrainingObjective",
    "build_training_objective",
]
