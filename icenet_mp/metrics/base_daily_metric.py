import torch
from torchmetrics import Metric

from .helpers import AccumulatorMixin, LandMaskMixin


class BaseDailyMetric(LandMaskMixin, AccumulatorMixin, Metric):
    """Shared state management for per-timestep error metrics.

    Provides ``sum_errors`` and ``count`` buffers with distributed-reduction support,
    plus the common accumulation logic used by both daily error metrics and centroid
    distance metrics. The default ``_compute_batch_stats()`` computes an element-wise
    error (via ``_compute_errors()``, overridden per metric) with optional land
    masking; subclasses that need a different per-batch reduction (e.g. centroid
    distance) can override ``_compute_batch_stats()`` directly instead.
    """

    sum_errors: torch.Tensor
    count: torch.Tensor

    def __init__(self, land_mask: torch.Tensor | None = None) -> None:
        """Initialize the metric state.

        Parameters
        ----------
        land_mask : torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells are excluded from the metric entirely.

        """
        super().__init__()
        self._register_land_mask(land_mask)
        self.add_state(
            "sum_errors",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor([], dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """Update metric state with a new batch.

        Subclasses must implement ``_compute_batch_stats()`` to return
        ``(batch_sum_errors, batch_count)`` tensors of shape ``(T,)`` where T is the
        time dimension.
        """
        batch_sum_errors, batch_count = self._compute_batch_stats(preds, target)

        if (
            self.sum_errors.numel() != 0
            and self.sum_errors.shape[0] != batch_sum_errors.shape[0]
        ):
            msg = (
                f"Time dimension mismatch: expected {self.sum_errors.shape[0]}, "
                f"got {batch_sum_errors.shape[0]}"
            )
            raise ValueError(msg)
        self._accumulate("sum_errors", batch_sum_errors)
        self._accumulate("count", batch_count)

    def compute(self) -> torch.Tensor:
        """Compute metric per lead time from accumulated sufficient statistics."""
        if self.count.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.sum_errors.device)
        count = torch.clamp(self.count, min=1)
        mean_errors = self.sum_errors / count.float()
        return self._finalize(mean_errors)

    def _finalize(self, mean_errors: torch.Tensor) -> torch.Tensor:
        """Apply final transformation to mean errors. Override in subclasses."""
        return mean_errors

    def _compute_errors(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute element-wise errors. Override in subclasses."""
        raise NotImplementedError

    def _compute_batch_stats(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = preds.shape[0]
        n_channels = preds.shape[2]
        num_spatial = n_channels * preds.shape[3] * preds.shape[4]

        errors = self._compute_errors(preds, targets)
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            # `torch.where`, not `errors * land_mask`: multiplying can't zero out a
            # NaN (0 * NaN = NaN), which would otherwise poison the whole lead-time's
            # sum for one bad land pixel.
            errors = torch.where(land_mask, errors, torch.zeros_like(errors))
            active_spatial = n_channels * int(land_mask.sum())
        else:
            active_spatial = num_spatial

        errors_reshaped = errors.view(batch_size, -1, num_spatial)
        batch_sum_errors = errors_reshaped.sum(dim=(0, 2))
        batch_count = torch.full(
            (errors.shape[1],),
            batch_size * active_spatial,
            dtype=torch.long,
            device=errors.device,
        )
        return batch_sum_errors, batch_count
