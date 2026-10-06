import pytest
import torch
from torchmetrics import Metric

from cryocast.metrics import IceNetAccuracyPerForecastDay, LandMaskMixin


class TestSingleChannelMetricMixin:
    def test_ensure_single_channel_raises_for_two_channel_inputs(self) -> None:
        metric = IceNetAccuracyPerForecastDay()
        two_channel = torch.rand(1, 1, 2, 4, 4)
        with pytest.raises(
            ValueError,
            match=r"IceNetAccuracyPerForecastDay is only defined for a single "
            r"sea-ice-concentration channel, but got preds with 2 channel\(s\)",
        ):
            metric.update(two_channel, two_channel)

    def test_ensure_single_channel_accepts_one_channel_inputs(self) -> None:
        metric = IceNetAccuracyPerForecastDay()
        one_channel = torch.rand(1, 1, 1, 4, 4)
        metric.update(one_channel, one_channel)
        assert metric.compute().item() == pytest.approx(100.0)


class TestLandMaskMixin:
    class MaskedMetric(LandMaskMixin, Metric):
        """A downstream-style metric that only passes the land mask on to super()."""

        def __init__(self, *, land_mask: torch.Tensor | None = None) -> None:
            """Initialise the metric, passing on the land mask."""
            super().__init__(land_mask=land_mask)

        def update(self) -> None:
            pass

        def compute(self) -> None:
            pass

    def test_land_mask_is_registered_as_boolean_buffer(self) -> None:
        metric = self.MaskedMetric(land_mask=torch.tensor([[1, 0], [0, 1]]))

        land_mask = metric.land_mask
        assert isinstance(land_mask, torch.Tensor)
        assert land_mask.dtype == torch.bool
        assert land_mask.tolist() == [[True, False], [False, True]]
        # Non-persistent, so the mask is not saved in checkpoints
        assert "land_mask" not in metric.state_dict()

    def test_no_land_mask_registers_no_buffer(self) -> None:
        metric = self.MaskedMetric()

        assert getattr(metric, "land_mask", None) is None
