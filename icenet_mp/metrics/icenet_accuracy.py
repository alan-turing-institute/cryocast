import torch
from torchmetrics import Metric

from icenet_mp.types import SEA_ICE_THRESHOLD

from .helpers import AccumulatorMixin, LandMaskMixin, SicOnlyMetricMixin


class IceNetAccuracyPerForecastDay(
    SicOnlyMetricMixin, LandMaskMixin, AccumulatorMixin, Metric
):
    """Binary accuracy metric for use at multiple leadtimes.

    Adapted from the IceNet implementation at:
    - https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

    """

    def __init__(self, land_mask: torch.Tensor | None = None) -> None:
        """Initialize the IceNetAccuracy metric.

        Parameters
        ----------
        land_mask : torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells are excluded from the accuracy calculation
            entirely, rather than counted as trivially-correct "no ice" agreements.

        """
        super().__init__()
        self._register_land_mask(land_mask)
        self.add_state(
            "weighted_score",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "possible_score",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update metric state with a new batch of predictions and targets."""
        self.ensure_single_channel(preds, target)
        preds = (preds > SEA_ICE_THRESHOLD).long()
        target = (target > SEA_ICE_THRESHOLD).long()
        sample_weight_ = (
            torch.ones_like(target) if sample_weight is None else sample_weight
        )
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            sample_weight_ = sample_weight_ * land_mask.to(dtype=sample_weight_.dtype)
        base_score = preds == target
        weighted_score = torch.sum(base_score * sample_weight_, dim=[0, 2, 3, 4])
        self._accumulate("weighted_score", weighted_score)
        possible_score = torch.sum(sample_weight_, dim=[0, 2, 3, 4])
        self._accumulate("possible_score", possible_score)

    def compute(self) -> torch.Tensor:
        """Compute the final accuracy metric as a percentage at each leadtime."""
        return self.weighted_score.float() / self.possible_score * 100.0  # type: ignore[arg-type, operator]
