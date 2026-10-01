import torch

from icenet_mp.types import SEA_ICE_THRESHOLD

from .base_ice_area_metric import BaseIceAreaMetric
from .helpers import binary_ice_edge


class DistanceAveragedIceEdgeErrorPerForecastDay(BaseIceAreaMetric):
    """Distance-averaged Integrated Ice Edge Error (DIIEE), in km, per lead time.

    The total misclassified-ice area (overestimation + underestimation, i.e. IIEE,
    see `IntegratedIceEdgeErrorPerForecastDay`) is normalised by the combined length
    of the predicted and true ice edges, giving an average displacement distance in
    km: roughly, how far the ice edge would need to move to reconcile the two fields.

        DIIEE = 2 * (over_area + under_area) / (pred_edge_length + true_edge_length)

    Edge length is approximated on the raster grid as
    ``(edge cell count) * pixel_size``; this is coarser than a true vector polygon
    perimeter, and needs no vector geometry, though an optional ``land_mask`` can be
    supplied to exclude land/ice boundaries from the edge count (see `__init__`).
    Lead times where both fields are entirely ice or entirely ice-free (combined edge
    length zero) are undefined and reported as NaN, rather than the ``-9999.99``
    sentinel used upstream, to compose correctly with tensor reductions (e.g.
    `nanmean`).
    """

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize the DIIEE metric (see `_IceAreaMetricBase` for parameters)."""
        super().__init__(pixel_size=pixel_size, land_mask=land_mask)
        self.sum_mismatch_area: torch.Tensor
        self.sum_edge_length: torch.Tensor

        # States initialized lazily on first update
        self.add_state(
            "sum_mismatch_area",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_edge_length",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update the DIIEE accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground-truth satellite SIC of shape (B, T, C, H, W).

        """
        self.ensure_single_channel(preds, target)
        batch_size, n_steps, n_channels, height, width = preds.shape

        preds_extent = (preds > SEA_ICE_THRESHOLD).reshape(-1, height, width)
        target_extent = (target > SEA_ICE_THRESHOLD).reshape(-1, height, width)
        land_mask = getattr(self, "land_mask", None)

        # Combined over- and under-estimation area (the IIEE integral)
        mismatch_map = self._masked_mismatch(preds_extent, target_extent)
        mismatch = mismatch_map.sum(dim=(-2, -1))
        mismatch_area = (
            mismatch.view(batch_size, n_steps, n_channels).sum(dim=(0, 2))
            * self.pixel_size**2
        )

        # Combined predicted + true ice-edge length
        pred_edge_cells = (
            binary_ice_edge(preds_extent, land_mask).sum(dim=(-2, -1)).float()
        )
        target_edge_cells = (
            binary_ice_edge(target_extent, land_mask).sum(dim=(-2, -1)).float()
        )
        edge_length = (pred_edge_cells + target_edge_cells).view(
            batch_size, n_steps, n_channels
        ).sum(dim=(0, 2)) * self.pixel_size

        self._accumulate("sum_mismatch_area", mismatch_area)
        self._accumulate("sum_edge_length", edge_length)

    def compute(self) -> torch.Tensor:
        """Compute the final DIIEE (average ice-edge displacement, in km) per lead time."""
        if self.sum_mismatch_area.numel() == 0:
            return torch.tensor(
                [], dtype=torch.float32, device=self.sum_mismatch_area.device
            )

        safe_edge_length = self.sum_edge_length.clamp(min=1e-8)
        diiee = 2.0 * self.sum_mismatch_area / safe_edge_length
        return torch.where(
            self.sum_edge_length > 0, diiee, torch.full_like(diiee, float("nan"))
        )
