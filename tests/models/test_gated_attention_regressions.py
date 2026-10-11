"""Numerical and gradient regression tests for SimVPv2 gated attention."""

import pytest
import torch
from torch import nn

from cryocast.models.common.gated_attention import GatedAttention, GatedAttentionBlock


def build_block(
    in_channels: int = 3,
    out_channels: int = 3,
    *,
    drop_path_prob: float = 0.0,
    mlp_drop_prob: float = 0.0,
) -> GatedAttentionBlock:
    """Construct a compact block to test actual convolution and residual paths."""
    return GatedAttentionBlock(
        in_channels,
        out_channels,
        kernel_size=5,
        dilation=1,
        mlp_ratio=2.0,
        drop_path_prob=drop_path_prob,
        mlp_drop_prob=mlp_drop_prob,
    )


@pytest.mark.parametrize(
    ("channels", "height", "width"),
    [(1, 5, 7), (3, 6, 6), (8, 7, 5)],
    ids=["single-channel", "three-channels", "eight-channels"],
)
def test_gated_attention_preserves_spatial_and_channel_shape(
    channels: int, height: int, width: int
) -> None:
    """The gating mechanism must preserve spatial and channel dimensions."""
    layer = GatedAttention(channels, kernel_size=7, dilation=2)
    x = torch.randn(2, channels, height, width)
    actual = layer(x)
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()


def test_gated_attention_zero_output_projection_is_exact_residual() -> None:
    """An empty update branch should reduce to the identity transform."""
    layer = GatedAttention(3, kernel_size=5, dilation=1)
    assert layer.proj_out.bias is not None
    with torch.no_grad():
        layer.proj_out.weight.zero_()
        layer.proj_out.bias.zero_()
    x = torch.randn(2, 3, 5, 5)
    torch.testing.assert_close(layer(x), x, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("gate_bias", "value_bias"),
    [(0.0, 3.0), (8.0, 2.0), (-8.0, -2.0)],
    ids=["half-open", "open", "closed"],
)
def test_gate_matches_explicit_sigmoid_reference(
    gate_bias: float, value_bias: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projection must multiply the value by sigmoid(gate), not raw gate."""
    layer = GatedAttention(1, kernel_size=5, dilation=1)
    monkeypatch.setattr(layer, "attend", nn.Identity())
    assert layer.expand.bias is not None
    assert layer.proj_out.bias is not None
    with torch.no_grad():
        layer.expand.weight.zero_()
        layer.expand.bias.copy_(torch.tensor([value_bias, gate_bias]))
        layer.proj_out.weight.fill_(1)
        layer.proj_out.bias.zero_()
    x = torch.full((2, 1, 4, 3), 0.7)
    expected = x + value_bias * torch.sigmoid(torch.tensor(gate_bias))
    torch.testing.assert_close(layer(x), expected.expand_as(x))


def test_attention_learns_through_depthwise_convolutions_and_gates() -> None:
    """Gradients should reach input, attention convolutions and projection."""
    torch.manual_seed(19)
    layer = GatedAttention(3, kernel_size=7, dilation=2)
    x = torch.randn(2, 3, 7, 5, requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(
    ("input_channels", "output_channels"),
    [(2, 2), (2, 4), (5, 3)],
    ids=["same", "grow", "shrink"],
)
def test_gated_block_channel_projection_and_gradient(
    input_channels: int, output_channels: int
) -> None:
    """Every channel configuration should project and backpropagate."""
    torch.manual_seed(21)
    block = build_block(input_channels, output_channels)
    x = torch.randn(2, input_channels, 5, 7, requires_grad=True)
    result = block(x)
    assert result.shape == (2, output_channels, 5, 7)
    result.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.layer_scale_1.grad is not None
    assert block.layer_scale_2.grad is not None
    if input_channels == output_channels:
        assert block.projection is None
    else:
        assert isinstance(block.projection, nn.Conv2d)
        assert block.projection.weight.grad is not None


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_stochastic_depth_endpoints(
    probability: float,
) -> None:
    """The exact endpoint probabilities need no stochastic RNG draws."""
    block = build_block(drop_path_prob=probability)
    block.train()
    residual = torch.randn(6, 3, 4, 5)
    dropped = block._drop_path(residual)
    if probability == 0:
        assert dropped is residual
    else:
        torch.testing.assert_close(dropped, torch.zeros_like(residual))


def test_stochastic_depth_drops_or_scales_entire_samples() -> None:
    """A mask must act on each sample, not independently on grid pixels."""
    torch.manual_seed(53)
    block = build_block(drop_path_prob=0.5)
    block.train()
    x = torch.ones(32, 3, 4, 5)
    result = block._drop_path(x)
    per_sample = result[:, 0, 0, 0]
    assert set(per_sample.tolist()) == {0.0, 2.0}
    torch.testing.assert_close(
        result, per_sample.view(-1, 1, 1, 1).expand_as(result), rtol=0, atol=0
    )


def test_stochastic_depth_is_disabled_during_evaluation() -> None:
    """Eval mode must never drop residuals even when drop probability is 1."""
    block = build_block(drop_path_prob=1.0)
    block.eval()
    x = torch.randn(2, 3, 5, 5)
    assert block._drop_path(x) is x


def test_full_block_with_dropped_residuals_is_exact_identity() -> None:
    """For matching channels, both fully dropped branches leave x unchanged."""
    block = build_block(3, 3, drop_path_prob=1.0)
    block.train()
    x = torch.randn(2, 3, 5, 5, requires_grad=True)
    result = block(x)
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_full_block_with_dropped_residuals_keeps_channel_projection() -> None:
    """A drop-path probability of 1 still applies the required projection."""
    block = build_block(3, 5, drop_path_prob=1.0)
    block.train()
    x = torch.randn(2, 3, 6, 6, requires_grad=True)
    assert block.projection is not None
    result = block(x)
    expected = block.projection(x)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.projection.weight.grad is not None


def test_full_block_evaluation_uses_residual_branches_at_drop_path_one() -> None:
    """Setting eval() must reactivate both residual branches."""
    torch.manual_seed(32)
    block = build_block(drop_path_prob=1.0)
    block.eval()
    x = torch.randn(2, 3, 5, 5)
    result = block(x)
    assert torch.isfinite(result).all()
    assert not torch.equal(result, x)
    torch.testing.assert_close(result, block(x), rtol=0, atol=0)


def test_mlp_dropout_configuration_is_applied_to_both_dropout_layers() -> None:
    """Both Conv-MLP dropout sites must use the requested probability."""
    block = build_block(3, 3, mlp_drop_prob=0.35)
    dropouts = [layer for layer in block.mlp if isinstance(layer, nn.Dropout)]
    assert len(dropouts) == 2
    assert all(layer.p == pytest.approx(0.35) for layer in dropouts)
