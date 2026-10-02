"""Lead-time weighted wrapper around any loss.

The wrapped base loss is evaluated independently at each forecast lead time
and the per-step values are combined with weights that grow with lead time:

    w_t = (t + 1) ** exponent,    t = 0, ..., T - 1

rescaled to have mean 1. The weights are computed in log space (as a softmax over
``exponent * log(t + 1)``) so that large positive or negative exponents cannot overflow
or underflow to a NaN loss. The rescaling keeps the overall loss magnitude (and hence
the learning rate) comparable with the unwrapped loss: weighting only redistributes
emphasis between lead times.

Wrapped losses that implement ``SupportsPerLeadTimeLoss`` are evaluated for every lead
time in one call; any other loss is called once per lead time.
"""

import math

import torch
from torch import nn

from icenet_mp.types import NDIM_NTCHW, SupportsPerLeadTimeLoss


class LeadTimeWeightedLoss(nn.Module):
    """Weight a base loss function so that later lead times contribute more."""

    def __init__(self, wrapped_loss: nn.Module, exponent: float = 1.0) -> None:
        """Initialise the LeadTimeWeightedLoss.

        Args:
            wrapped_loss: Loss function that takes an NCHW slice and returns a scalar.
            exponent: Power-law exponent for the weights. 0 gives uniform weighting,
                positive values emphasise later lead times, and negative values
                emphasise earlier lead times. Must be a finite number.

        """
        if not math.isfinite(exponent):
            msg = f"LeadTimeWeightedLoss exponent must be finite, but got {exponent}."
            raise ValueError(msg)
        super().__init__()
        self.wrapped_loss = wrapped_loss
        self.exponent = exponent

    def weights(self, n_steps: int, device: torch.device) -> torch.Tensor:
        """Return the per-lead-time weights for `n_steps` forecast steps."""
        # Numerically stable equivalent to steps**exponent / sum(steps**exponent)
        log_steps = torch.arange(
            1, n_steps + 1, device=device, dtype=torch.float32
        ).log()
        return torch.softmax(self.exponent * log_steps, dim=0) * n_steps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the lead-time weighted scalar loss for NTCHW prediction/target."""
        if prediction.ndim != NDIM_NTCHW:
            msg = (
                "LeadTimeWeightedLoss expects NTCHW inputs with a lead-time dimension, "
                f"but got a tensor of shape {tuple(prediction.shape)}."
            )
            raise ValueError(msg)
        if prediction.shape != target.shape:
            msg = (
                "LeadTimeWeightedLoss expects prediction and target to have the same "
                f"shape, but got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
            raise ValueError(msg)
        # Derive the number of steps from the input for increased flexibility
        n_steps = prediction.shape[1]
        if isinstance(self.wrapped_loss, SupportsPerLeadTimeLoss):
            per_step = self.wrapped_loss.per_lead_time_loss(prediction, target)
        else:
            per_step = torch.stack(
                [
                    self.wrapped_loss(prediction[:, t], target[:, t])
                    for t in range(n_steps)
                ]
            )
        return (self.weights(n_steps, per_step.device) * per_step).mean()
