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
        if delta <= 0:
            msg = "delta must be greater than 0."
            raise ValueError(msg)
        if min_uncertainty <= 0:
            msg = "min_uncertainty must be greater than 0."
            raise ValueError(msg)
        if max_uncertainty < min_uncertainty:
            msg = "max_uncertainty must be greater than or equal to min_uncertainty."
            raise ValueError(msg)
        if power <= 0:
            msg = "power must be greater than 0."
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
        pointwise_loss = functional.huber_loss(
            preds,
            targets,
            reduction="none",
            delta=self.delta,
        )
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

        uncertainty = uncertainty.to(device=preds.device, dtype=preds.dtype)
        valid = (
            torch.isfinite(uncertainty)
            & (uncertainty > 0)
            & (uncertainty <= self.max_uncertainty)
        )
        if not torch.any(valid):
            return pointwise_loss.mean()

        sigma = uncertainty.clamp_min(self.min_uncertainty)
        weights = torch.where(valid, sigma.pow(-self.power), torch.zeros_like(sigma))
        return (pointwise_loss * weights).sum() / weights.sum()
