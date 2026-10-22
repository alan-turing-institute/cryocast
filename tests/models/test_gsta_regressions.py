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


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 2, 6, 8),
        (2, 1, 6, 6, 8),
        (2, 6, 1, 6, 8),
        (2, 2, 2, 6, 8),
        (2, 6, 6, 8),
        (2, 2, 3, 1, 6, 8),
    ],
    ids=[
        "swapped-time-channel",
        "one-history-six-channels",
        "six-history-one-channel",
        "wrong-channels",
        "rank-four",
        "rank-six",
    ],
)
def test_gsta_rejects_mislabelled_history_before_mutating_batchnorm(
    shape: tuple[int, ...],
) -> None:
    """Equal folded channel counts must not silently change forecast semantics."""
    processor = build_processor(
        channels=3, n_history_steps=2, n_forecast_steps=4
    ).train()
    buffers = {name: value.clone() for name, value in processor.named_buffers()}
    with pytest.raises(ValueError, match="Expected history"):
        processor.rollout(torch.zeros(shape))
    for name, value in processor.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


@pytest.mark.parametrize("n_blocks", [1, 3])
def test_gsta_noncontiguous_history_matches_contiguous_values_and_gradients(
    n_blocks: int,
) -> None:
    """Layout changes preserve the forecast, gradients and independence from targets."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(624)
        processor = (
            build_processor(
                channels=3, n_history_steps=2, n_forecast_steps=4, n_blocks=n_blocks
            )
            .double()
            .eval()
        )
        values = (
            torch.randn(2, 2, 3, 8, 6, dtype=torch.float64)
            .transpose(-1, -2)
            .requires_grad_()
        )
    assert not values.is_contiguous()
    reference = values.detach().contiguous().requires_grad_()
    snapshot = values.detach().clone()
    target = torch.full(
        (2, 4, 3, 6, 8), float("nan"), dtype=torch.float64, requires_grad=True
    )
    actual = processor.rollout(values, target)
    expected = processor.rollout(reference)
    assert actual.prediction.shape == (2, 4, 3, 6, 8)
    assert actual.loss is None
    torch.testing.assert_close(
        actual.prediction, expected.prediction, rtol=1e-12, atol=1e-12
    )
    params = tuple(processor.parameters())
    gradients = torch.autograd.grad(
        actual.prediction.square().mean(), (values, target, *params), allow_unused=True
    )
    reference_gradients = torch.autograd.grad(
        expected.prediction.square().mean(), (reference, *params)
    )
    assert gradients[1] is None
    for observed, wanted in zip(
        (gradients[0], *gradients[2:]), reference_gradients, strict=True
    ):
        assert observed is not None
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, wanted, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(values, snapshot, rtol=0, atol=0)


def test_gsta_history_and_forecast_channel_order_has_exact_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-shot reorder must send each forecast gradient to the correct history cell."""
    processor = build_processor(channels=2, n_history_steps=3, n_forecast_steps=2)

    class SelectHistory(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, [4, 5, 0, 1]]

    monkeypatch.setattr(processor, "translator", SelectHistory())
    values = (
        torch.arange(2 * 3 * 2 * 8 * 6, dtype=torch.float64)
        .reshape(2, 3, 2, 8, 6)
        .transpose(-1, -2)
        .requires_grad_()
    )
    output = processor.rollout(values).prediction
    torch.testing.assert_close(output, values[:, [2, 0]], rtol=0, atol=0)
    weights = torch.arange(output.numel(), dtype=torch.float64).reshape_as(output)
    (output * weights).sum().backward()
    expected = torch.zeros_like(values)
    expected[:, 2] = weights[:, 0]
    expected[:, 0] = weights[:, 1]
    torch.testing.assert_close(values.grad, expected, rtol=0, atol=0)
