import torch
import torch.nn.functional as F


class AccumulatorMixin:
    """Mixin providing lazy first-batch accumulation for torchmetrics states.

    Several metrics in this package hold a running-sum state that starts as an
    empty tensor (so its dtype/shape isn't known until the first batch arrives)
    and must be initialised in place, rather than added to, on that first call.
    """

    def _accumulate(self, name: str, value: torch.Tensor) -> None:
        current = getattr(self, name)
        setattr(self, name, value if current.numel() == 0 else current + value)


class LandMaskMixin:
    """Mixin providing shared registration of an optional land-mask buffer."""

    def _register_land_mask(self, land_mask: torch.Tensor | None) -> None:
        if land_mask is not None:
            self.register_buffer("land_mask", land_mask.bool(), persistent=False)  # type: ignore[attr-defined]


class SicOnlyMetricMixin:
    """Mixin guarding metric inputs that are only defined for a single SIC channel."""

    def ensure_single_channel(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.shape[2] != 1 or targets.shape[2] != 1:
            msg = (
                f"{type(self).__name__} is only defined for a single "
                f"sea-ice-concentration channel, but got preds with "
                f"{preds.shape[2]} channel(s) and target with {targets.shape[2]} "
                f"channel(s). Multi-channel targets (e.g. an auxiliary "
                f"ice-thickness variable) are not supported by this metric."
            )
            raise ValueError(msg)


def binary_ice_edge(
    ice_mask: torch.Tensor, land_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Boolean ice-edge map: True for ice cells that border a non-ice ocean cell.

    Parameters
    ----------
    ice_mask : torch.Tensor
        Boolean tensor of shape (N, H, W).
    land_mask : torch.Tensor, optional
        Boolean tensor of shape (H, W), True for ocean cells and False for land.
        When given, land cells are excluded from the edge test — both as neighbors
        (an ice cell bordering only land is not counted as an edge cell, though it
        is still counted if it also borders true open water) and from the returned
        map itself, so a land cell is never reported as an edge cell even if its raw
        (unmasked) value happens to read as "ice".

    Cells beyond the grid boundary are treated as matching the cell they border (via
    replicate padding), rather than being manufactured as non-ice: the domain's own
    edge is not itself an ice/ocean transition, so it should not be able to invent a
    disagreement just because a real edge would need one more ring of pixels to
    resolve. A genuine ice edge that runs along the domain boundary is still detected
    normally, since it disagrees with its interior (in-grid) neighbors regardless.

    """
    comparison_mask = ice_mask
    if land_mask is not None:
        comparison_mask = ice_mask | ~land_mask.bool()
    padded = F.pad(comparison_mask.float(), (1, 1, 1, 1), mode="replicate").bool()
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    edge = ice_mask & (
        (ice_mask != up) | (ice_mask != down) | (ice_mask != left) | (ice_mask != right)
    )
    if land_mask is not None:
        edge = edge & land_mask.bool()
    return edge
