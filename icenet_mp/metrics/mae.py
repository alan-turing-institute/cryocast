import torch

from .base_daily_metric import BaseDailyMetric
from .registry import metric_registry


@metric_registry.register("mae")
class MAEPerForecastDay(BaseDailyMetric):
    """Mean Absolute Error per forecast lead time."""

    def _compute_errors(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.abs(preds - targets)
