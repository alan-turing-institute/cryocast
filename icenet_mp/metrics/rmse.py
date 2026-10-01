import torch

from .base_daily_metric import BaseDailyMetric


class RMSEPerForecastDay(BaseDailyMetric):
    """Root Mean Squared Error per forecast lead time."""

    def _compute_errors(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return (preds - targets) ** 2

    def _finalize(self, mean_errors: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(mean_errors)
