import torch

from .base_daily_metric import BaseDailyMetric
from .registry import metric_registry


@metric_registry.register("spatial_mean_ground_truth")
class SpatialMeanGroundTruthPerForecastDay(BaseDailyMetric):
    """Land-masked spatial-mean ground-truth value per forecast lead time.

    Paired with `SpatialMeanPredictionPerForecastDay` to trace spatial-mean prediction
    versus ground truth across forecast steps, e.g. to spot a systematic bias or a
    collapse toward a constant value that per-pixel error metrics would not by
    themselves reveal.
    """

    def _compute_errors(
        self,
        preds: torch.Tensor,  # noqa: ARG002
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return targets


@metric_registry.register("spatial_mean_prediction")
class SpatialMeanPredictionPerForecastDay(BaseDailyMetric):
    """Land-masked spatial-mean prediction value per forecast lead time.

    Paired with `SpatialMeanGroundTruthPerForecastDay` to trace spatial-mean prediction
    versus ground truth across forecast steps.
    """

    def _compute_errors(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return preds
