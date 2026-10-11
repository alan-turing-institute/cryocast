"""Processor-level regression coverage for the SimVPv2 gSTA translator."""

import logging

import pytest
import torch
from torch import nn

from cryocast.models.common.gated_attention import GatedAttentionBlock
from cryocast.models.processors import GSTAProcessor
from cryocast.types import DataSpace, ProcessorOutput


def build_processor(
    *,
    channels: int = 2,
    n_history_steps: int = 2,
    n_forecast_steps: int = 3,
    n_blocks: int = 2,
    hid_channels: int = 8,
) -> GSTAProcessor:
    """Make a small but real model, with enough spatial cells for BatchNorm."""
    return GSTAProcessor(
        data_space=DataSpace(name="latent", channels=channels, shape=(6, 8)),
        n_history_steps=n_history_steps,
        n_forecast_steps=n_forecast_steps,
        n_blocks=n_blocks,
        hid_channels=hid_channels,
        kernel_size=5,
        dilation=1,
        drop_path_prob=0.0,
        mlp_drop_prob=0.0,
        mlp_ratio=2.0,
    )


@pytest.mark.parametrize(
    ("channels", "history", "forecast", "n_blocks"),
    [
        (1, 1, 2, 1),
        (2, 3, 1, 1),
        (3, 1, 4, 2),
        (2, 4, 3, 3),
        (1, 2, 5, 3),
    ],
    ids=[
        "one-channel-one-history",
        "multiple-history-short-forecast",
        "multiple-forecast",
        "four-history-three-forecast",
        "forecast-longer-than-history",
    ],
)
def test_gsta_forecast_shape_and_absent_latent_loss(
    channels: int, history: int, forecast: int, n_blocks: int
) -> None:
    """A single gSTA call must unfold to the requested NTCHW forecast."""
    processor = build_processor(
        channels=channels,
        n_history_steps=history,
        n_forecast_steps=forecast,
        n_blocks=n_blocks,
    )
    x = torch.randn(2, history, channels, 6, 8)
    processor.eval()
    with torch.no_grad():
        result = processor.rollout(x)
    assert isinstance(result, ProcessorOutput)
    assert result.prediction.shape == (2, forecast, channels, 6, 8)
    assert torch.isfinite(result.prediction).all()
    assert result.loss is None


@pytest.mark.parametrize("n_blocks", [0, -1, -10])
def test_gsta_rejects_empty_translator(n_blocks: int) -> None:
    """An empty or negative block count must never be silently accepted."""
    with pytest.raises(ValueError, match=r"n_blocks must be at least 1"):
        build_processor(n_blocks=n_blocks)


def test_gsta_single_block_warns_without_using_hidden_channels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single block directly connects history to forecast channels."""
    with caplog.at_level(logging.WARNING):
        processor = build_processor(
            channels=3,
            n_history_steps=2,
            n_forecast_steps=4,
            n_blocks=1,
            hid_channels=11,
        )
    assert "no hidden channels will be used" in caplog.text
    assert len(processor.translator) == 1
    [block] = processor.translator
    assert isinstance(block, GatedAttentionBlock)
    assert block.norm1.num_features == 6
    assert isinstance(block.projection, nn.Conv2d)
    assert block.projection.out_channels == 12


def test_gsta_block_channels_track_history_hidden_and_forecast_sizes() -> None:
    """The first/last blocks must use externally requested channel counts."""
    processor = build_processor(
        channels=3,
        n_history_steps=2,
        n_forecast_steps=4,
        n_blocks=3,
        hid_channels=10,
    )
    blocks = list(processor.translator)
    assert len(blocks) == 3
    assert [
        block.norm1.num_features
        for block in blocks
        if isinstance(block.norm1, nn.BatchNorm2d)
    ] == [6, 10, 10]
    projections = [block.projection for block in blocks]
    assert isinstance(projections[0], nn.Conv2d)
    assert projections[1] is None
    assert isinstance(projections[2], nn.Conv2d)
    assert [p.out_channels for p in projections if p is not None] == [10, 12]


def test_gsta_one_pass_preserves_oldest_to_newest_history_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Folding and unfolding both time dimensions must preserve their order."""
    processor = build_processor(
        channels=2, n_history_steps=2, n_forecast_steps=2, n_blocks=2
    )
    # Bypass convolution to isolate the exact tensor reshape contract.
    monkeypatch.setattr(processor, "translator", nn.Identity())
    x = torch.arange(1 * 2 * 2 * 6 * 8, dtype=torch.float32).reshape(1, 2, 2, 6, 8)
    result = processor.rollout(x)
    assert result.prediction.shape == x.shape
    torch.testing.assert_close(result.prediction, x, rtol=0, atol=0)


def test_gsta_translator_receives_full_history_flattened_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-shot translator must not drop any history steps or interleave them."""
    processor = build_processor(channels=2, n_history_steps=3, n_forecast_steps=2)

    class RecordAndSelect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen = x.detach().clone()
            return x[:, :4]

    recorder = RecordAndSelect()
    monkeypatch.setattr(processor, "translator", recorder)
    x = torch.arange(2 * 3 * 2 * 6 * 8, dtype=torch.float32).reshape(2, 3, 2, 6, 8)
    result = processor.rollout(x)

    assert recorder.seen is not None
    torch.testing.assert_close(recorder.seen, x.flatten(start_dim=1, end_dim=2))
    torch.testing.assert_close(result.prediction, x[:, :2])


def test_gsta_rollout_does_not_use_target_values_for_prediction() -> None:
    """Pure one-shot processors receive a target only for the outer loss."""
    torch.manual_seed(42)
    processor = build_processor(channels=2, n_history_steps=2, n_forecast_steps=3)
    processor.eval()
    x = torch.randn(2, 2, 2, 6, 8)
    target = torch.randn(2, 3, 2, 6, 8)
    with torch.no_grad():
        no_target = processor.rollout(x)
        with_target = processor.rollout(x, target)
    torch.testing.assert_close(
        no_target.prediction, with_target.prediction, rtol=0, atol=0
    )
    assert with_target.loss is None


@pytest.mark.parametrize("n_blocks", [1, 2, 3])
def test_gsta_backpropagates_to_history_and_all_trainable_weights(
    n_blocks: int,
) -> None:
    """Training gradients must reach history and every translator block."""
    torch.manual_seed(19 + n_blocks)
    processor = build_processor(
        channels=3, n_history_steps=2, n_forecast_steps=3, n_blocks=n_blocks
    )
    x = torch.randn(2, 2, 3, 6, 8, requires_grad=True)
    output = processor.rollout(x).prediction
    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, weight in processor.named_parameters():
        assert weight.grad is not None, name
        assert torch.isfinite(weight.grad).all(), name


def test_gsta_one_step_optimizer_changes_prediction() -> None:
    """A real optimizer update should change output without a mocked trainer."""
    torch.manual_seed(23)
    processor = build_processor(channels=2, n_history_steps=2, n_forecast_steps=2)
    optimizer = torch.optim.AdamW(processor.parameters(), lr=0.01)
    x = torch.randn(2, 2, 2, 6, 8)
    target = torch.randn(2, 2, 2, 6, 8)
    processor.eval()
    with torch.no_grad():
        initial = processor.rollout(x).prediction.detach().clone()
    processor.train()
    optimizer.zero_grad(set_to_none=True)
    (processor.rollout(x).prediction - target).square().mean().backward()
    optimizer.step()
    processor.eval()
    with torch.no_grad():
        updated = processor.rollout(x).prediction
    assert torch.isfinite(updated).all()
    assert not torch.equal(initial, updated)
