import math
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional


class UncertaintyWeightedLoss(nn.Module):
    """Huber loss weighted by known per-pixel observational uncertainty.

    For valid uncertainty values ``sigma``, each point receives inverse-uncertainty
    weight ``sigma ** -power``. The weighted loss is normalised by the sum of weights.
    Away from configured clipping and validity thresholds, multiplying every
    uncertainty value by the same positive constant leaves the objective unchanged.

    Invalid, zero, negative, or sentinel-like uncertainty values are excluded. If a
    batch contains no valid uncertainty values, the loss falls back to ordinary Huber
    loss rather than returning NaN or suppressing the training signal.
    """

    requires_uncertainty: ClassVar[bool] = True

    def __init__(
        self,
        uncertainty_variable: str = "total_standard_uncertainty",
        delta: float = 0.5,
        min_uncertainty: float = 0.01,
        max_uncertainty: float = 1.0,
        power: float = 2.0,
    ) -> None:
        """Initialise uncertainty weighting parameters."""
        super().__init__()
        if not math.isfinite(delta) or delta <= 0:
            msg = "delta must be finite and greater than 0."
            raise ValueError(msg)
        if not math.isfinite(min_uncertainty) or min_uncertainty <= 0:
            msg = "min_uncertainty must be finite and greater than 0."
            raise ValueError(msg)
        if not math.isfinite(max_uncertainty) or max_uncertainty < min_uncertainty:
            msg = "max_uncertainty must be finite and greater than or equal to min_uncertainty."
            raise ValueError(msg)
        if not math.isfinite(power) or power <= 0:
            msg = "power must be finite and greater than 0."
            raise ValueError(msg)
        if not uncertainty_variable:
            msg = "uncertainty_variable must not be empty."
            raise ValueError(msg)

        self.delta = delta
        self.max_uncertainty = max_uncertainty
        self.min_uncertainty = min_uncertainty
        self.power = power
        self.uncertainty_variable = uncertainty_variable

    def forward(
        self, preds: Tensor, targets: Tensor, uncertainty: Tensor | None = None
    ) -> Tensor:
        """Compute inverse-uncertainty-weighted Huber loss."""
        if uncertainty is None:
            msg = (
                "UncertaintyWeightedLoss requires an uncertainty tensor. Ensure the "
                "configured target dataset contains the requested uncertainty variable."
            )
            raise ValueError(msg)
        if uncertainty.shape != targets.shape:
            msg = (
                f"Uncertainty shape {tuple(uncertainty.shape)} does not match target "
                f"shape {tuple(targets.shape)}."
            )
            raise ValueError(msg)

        if preds.shape != targets.shape:
            msg = (
                f"Prediction shape {tuple(preds.shape)} does not match target "
                f"shape {tuple(targets.shape)}."
            )
            raise ValueError(msg)

        # Accumulate in at least float32, retaining double precision when present.
        # Do not quantise physical uncertainty to a half-precision model's dtype.
        dtype = torch.promote_types(
            torch.promote_types(preds.dtype, targets.dtype), uncertainty.dtype
        )
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        preds = preds.to(dtype=dtype)
        targets = targets.to(dtype=dtype)
        uncertainty = uncertainty.to(device=preds.device, dtype=dtype)
        valid = (
            torch.isfinite(uncertainty)
            & (uncertainty > 0)
            & (uncertainty <= self.max_uncertainty)
        )
        if not torch.any(valid):
            return functional.huber_loss(preds, targets, delta=self.delta)

        # Select before computing Huber: multiplying an excluded NaN by zero
        # would otherwise contaminate both the scalar loss and its gradients.
        pointwise_loss = functional.huber_loss(
            preds[valid], targets[valid], reduction="none", delta=self.delta
        )
        sigma = uncertainty[valid].clamp_min(self.min_uncertainty)
        # Relative inverse powers preserve the normalised objective while bounding
        # every weight by one and keeping at least one weight equal to one.
        weights = (sigma.min() / sigma).pow(self.power)
        return (pointwise_loss * weights).sum() / weights.sum()
