import torch
from torchmetrics import Metric

from .helpers import AccumulatorMixin, LandMaskMixin, SicOnlyMetricMixin


class BaseIceAreaMetric(SicOnlyMetricMixin, LandMaskMixin, AccumulatorMixin, Metric):
    """Shared construction for threshold-based sea ice extent/edge metrics."""

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize shared state.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).
        land_mask: torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells are excluded from the metric entirely.

        """
        super().__init__()
        self.pixel_size = pixel_size
        self._register_land_mask(land_mask)

    def _masked_mismatch(
        self, preds_extent: torch.Tensor, target_extent: torch.Tensor
    ) -> torch.Tensor:
        """Land-masked disagreement map, same shape as the inputs (..., H, W)."""
        mismatch = (preds_extent != target_extent).float()
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            mismatch = mismatch * land_mask.to(dtype=mismatch.dtype)
        return mismatch


class MeanIceAreaMetric(BaseIceAreaMetric):
    """Shared accumulation for metrics reported as ``pixel_size**2 * mean(error)``.

    Subclasses supply the per-(batch, lead-time) error contribution via
    ``_batch_error()``; this class handles the running sum/count and final scaling.
    """

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize the metric state (see `_IceAreaMetricBase` for parameters)."""
        super().__init__(pixel_size=pixel_size, land_mask=land_mask)
        self.sum_errors: torch.Tensor
        self.sample_count: torch.Tensor

        # States initialized lazily on first update
        self.add_state("sum_errors", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("sample_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _batch_error(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-(batch, lead-time) error contribution, shape (B, T). Override in subclasses."""
        raise NotImplementedError

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the accumulators with a new batch.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground truth values of shape (B, T, C, H, W).

        """
        self.ensure_single_channel(preds, target)
        error = self._batch_error(preds, target)

        self._accumulate("sum_errors", error.sum(dim=0))  # Sum across batch dimension
        self.sample_count += error.shape[0]  # Increment count by batch size

    def compute(self) -> torch.Tensor:
        """Compute the final metric in km²."""
        if self.sum_errors.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        mean_error = self.sum_errors / self.sample_count
        return mean_error * self.pixel_size**2  # type: ignore[operator]
