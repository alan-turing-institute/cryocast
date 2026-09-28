import torch

from icenet_mp.types import SEA_ICE_THRESHOLD

from .base_ice_area_metric import MeanIceAreaMetric


class SeaIceExtentErrorPerForecastDay(MeanIceAreaMetric):
    """Sea Ice Extent error (SIEError) metric (in km^2) for use at multiple lead times.

    The SIE error is calculated as the signed difference between the predicted and
    true sea ice extent for each forecast day. Sea ice presence is defined by having
    a concentration greater than the threshold value.
    """

    def _batch_error(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds_extent = preds > SEA_ICE_THRESHOLD
        target_extent = target > SEA_ICE_THRESHOLD
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            preds_extent = preds_extent & land_mask
            target_extent = target_extent & land_mask

        # Calculate the SIE for each day of the forecast
        pred_sie = torch.sum(preds_extent, dim=(2, 3, 4))  # Shape: (B, T)
        true_sie = torch.sum(target_extent, dim=(2, 3, 4))  # Shape: (B, T)
        return (pred_sie - true_sie).float()
