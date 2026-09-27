import torch

from .base_daily_metric import BaseDailyMetric


class MAEPerForecastDay(BaseDailyMetric):
    """Mean Absolute Error per forecast lead time."""

    def _compute_errors(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.abs(preds - targets)
