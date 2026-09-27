import torch
import torch.nn.functional as F
from torchmetrics import Metric

from icenet_mp.types import SEA_ICE_THRESHOLD

from .helpers import (
    AccumulatorMixin,
    LandMaskMixin,
    SicOnlyMetricMixin,
    binary_ice_edge,
)


class FractionalSkillScorePerForecastDay(
    SicOnlyMetricMixin, LandMaskMixin, AccumulatorMixin, Metric
):
    """FractionalSkill Score (FSS) of the sea-ice edge, for use at multiple lead times.

    Computes the FSS of the sea-ice edge at a fixed neighbourhood size, following
    Roberts and Lean (2008) and its application to sea-ice edge position by
    Melsom et al. (2019, https://doi.org/10.5194/os-15-615-2019). Adapted from the
    `effectiveres_icenetv2_FSS` notebook's step-by-step computation.


    Each field is first reduced to a binary ice-edge map (cells that are ice but
    border a non-ice cell). The local fraction of edge cells is then computed within
    a fixed `neighbourhood_size` x `neighbourhood_size` window around every cell. FSS
    compares the mean squared error (MSE) between the predicted and true fraction
    fields to a reference (worst-case) MSE:

        FSS = 1 - MSE / MSE_ref

    FSS is 1 for a perfect match and 0 (or below) for no better than the worst-case
    reference. To assess effective resolution, instantiate this metric once per
    neighbourhood size of interest and compare where the resulting curve crosses 0.5.
    """

    def __init__(
        self, neighbourhood_size: int = 1, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize the FSS metric.

        Parameters
        ----------
        neighbourhood_size: int, optional
            Size (in pixels) of the square neighbourhood window used to compute local
            edge-cell fractions. Must be a positive odd integer (default is 1).
        land_mask: torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land/ice boundaries are excluded from the ice-edge detection,
            so only ocean ice/no-ice transitions count as the sea-ice edge.

        """
        super().__init__()
        if neighbourhood_size < 1 or neighbourhood_size % 2 == 0:
            msg = "neighbourhood_size must be a positive odd integer."
            raise ValueError(msg)
        self.neighbourhood_size = neighbourhood_size
        self._register_land_mask(land_mask)

        self.sum_mse: torch.Tensor
        self.sum_mse_ref: torch.Tensor
        self.count: torch.Tensor

        # States initialized lazily on first update
        self.add_state(
            "sum_mse",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_mse_ref",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count", default=torch.tensor([], dtype=torch.long), dist_reduce_fx="sum"
        )

    def _neighborhood_fraction(self, edge: torch.Tensor) -> torch.Tensor:
        """Local fraction of edge cells within an n x n window of each cell.

        Parameters
        ----------
        edge : torch.Tensor
            Boolean tensor of shape (N, H, W).

        """
        n = self.neighbourhood_size
        kernel = torch.ones((1, 1, n, n), dtype=torch.float32, device=edge.device)
        summed = F.conv2d(edge.float().unsqueeze(1), kernel, padding=n // 2)
        return summed.squeeze(1) / (n * n)

    def _config_stats(
        self, lambda_truth: torch.Tensor, lambda_model: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average MSE and reference MSE across all n x n non-overlapping sub-grids.

        Sampling every n-th cell at each of the n x n offsets removes the dependency
        of the MSE on a single, arbitrary alignment of the neighbourhood window.

        Parameters
        ----------
        lambda_truth : torch.Tensor
            Local edge-cell fraction field for the target, shape (N, H, W).
        lambda_model : torch.Tensor
            Local edge-cell fraction field for the prediction, shape (N, H, W).

        """
        n = self.neighbourhood_size
        mse_sum = None
        mse_ref_sum = None
        total_area = 0
        for dx in range(n):
            for dy in range(n):
                truth_slice = lambda_truth[:, dx::n, dy::n]
                model_slice = lambda_model[:, dx::n, dy::n]
                area = truth_slice.shape[-2] * truth_slice.shape[-1]
                if area == 0:
                    continue
                diff = model_slice - truth_slice
                mse_term = diff.pow(2).sum(dim=(-2, -1))
                term1 = truth_slice.pow(2).sum(dim=(-2, -1)) + model_slice.pow(2).sum(
                    dim=(-2, -1)
                )
                term2 = (1 - truth_slice).pow(2).sum(dim=(-2, -1)) + (
                    1 - model_slice
                ).pow(2).sum(dim=(-2, -1))
                mse_ref_term = torch.minimum(term1, term2)
                mse_sum = mse_term if mse_sum is None else mse_sum + mse_term
                mse_ref_sum = (
                    mse_ref_term if mse_ref_sum is None else mse_ref_sum + mse_ref_term
                )
                total_area += area
        if mse_sum is None or mse_ref_sum is None:
            zeros = torch.zeros(
                lambda_truth.shape[0], dtype=torch.float32, device=lambda_truth.device
            )
            return zeros, zeros.clone()
        return mse_sum / total_area, mse_ref_sum / total_area

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update the FSS accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground truth values of shape (B, T, C, H, W).

        """
        self.ensure_single_channel(preds, target)
        batch_size, n_steps, n_channels, height, width = preds.shape

        preds_mask = (preds > SEA_ICE_THRESHOLD).reshape(-1, height, width)
        target_mask = (target > SEA_ICE_THRESHOLD).reshape(-1, height, width)

        land_mask = getattr(self, "land_mask", None)
        lambda_model = self._neighborhood_fraction(
            binary_ice_edge(preds_mask, land_mask)
        )
        lambda_truth = self._neighborhood_fraction(
            binary_ice_edge(target_mask, land_mask)
        )

        mse, mse_ref = self._config_stats(lambda_truth, lambda_model)

        mse = mse.view(batch_size, n_steps, n_channels).sum(dim=(0, 2))
        mse_ref = mse_ref.view(batch_size, n_steps, n_channels).sum(dim=(0, 2))
        batch_count = torch.full(
            (n_steps,), batch_size * n_channels, dtype=torch.long, device=preds.device
        )

        self._accumulate("sum_mse", mse)
        self._accumulate("sum_mse_ref", mse_ref)
        self._accumulate("count", batch_count)

    def compute(self) -> torch.Tensor:
        """Compute the final FSS per lead time.

        Undefined (NaN) for lead times where neither field has any ice edge at all
        (mean_mse_ref == 0), consistent with
        `icenet_mp.metrics.extent_metrics.DistanceAveragedIceEdgeErrorPerForecastDay`.
        """
        if self.count.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.sum_mse.device)

        mean_mse = self.sum_mse / self.count
        mean_mse_ref = self.sum_mse_ref / self.count
        safe_mse_ref = mean_mse_ref.clamp(min=1e-8)
        fss = 1 - mean_mse / safe_mse_ref
        return torch.where(mean_mse_ref > 0, fss, torch.full_like(fss, float("nan")))
