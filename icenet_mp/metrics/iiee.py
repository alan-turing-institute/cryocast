import torch

from icenet_mp.types import SEA_ICE_THRESHOLD

from .base_ice_area_metric import MeanIceAreaMetric


class IntegratedIceEdgeErrorPerForecastDay(MeanIceAreaMetric):
    """Integrated Ice Edge Error (IIEE) metric (in km^2) for use at multiple lead times.

    IIEE is the area of the symmetric difference between the predicted and true ice
    extent: the total area where the two disagree on ice presence, following
    Goessling et al. (2016, https://doi.org/10.1002/2015GL067232). Sea ice presence
    is defined by having a probability greater than the threshold value.

    Unlike `SeaIceExtentErrorPerForecastDay`, which is a signed difference of extents
    and can cancel out over- and under-estimation, IIEE always accumulates
    disagreement and is therefore always >= |SIEError|.
    """

    def _batch_error(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds_extent = preds > SEA_ICE_THRESHOLD
        target_extent = target > SEA_ICE_THRESHOLD
        disagreement = self._masked_mismatch(preds_extent, target_extent)
        return torch.sum(disagreement, dim=(2, 3, 4))
