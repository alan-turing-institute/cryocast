import pytest
import torch

from icenet_mp.metrics import IceNetAccuracyPerForecastDay


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
