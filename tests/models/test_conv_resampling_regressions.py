"""Regression tests for convolutional encoder and decoder resampling blocks."""

import pytest
import torch
from torch import nn

from cryocast.models.common import ConvBlockDownsample, ConvBlockUpsample
from cryocast.models.common.weighted_upsample import WeightedUpsample


@pytest.mark.parametrize("n_subblocks", [1, 2, 3])
@pytest.mark.parametrize("scale_factor", [1, 2, 3])
@pytest.mark.parametrize("out_channels", [None, 5], ids=["default", "explicit"])
def test_conv_downsample_shapes_and_gradients(
    n_subblocks: int,
    scale_factor: int,
    out_channels: int | None,
) -> None:
    """Every block depth and stride must preserve expected output geometry."""
    input_channels = 4
    layer = ConvBlockDownsample(
        input_channels,
        out_channels=out_channels,
        scale_factor=scale_factor,
        n_subblocks=n_subblocks,
        kernel_size=3,
        norm_type="none",
    )
    x = torch.randn(2, input_channels, 12, 18, requires_grad=True)
    result = layer(x)
    expected_channels = out_channels or input_channels * scale_factor
    assert result.shape == (
        2,
        expected_channels,
        12 // scale_factor,
        18 // scale_factor,
    )
    assert torch.isfinite(result).all()

    result.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("upsample_mode", ["bilinear", "shuffle"])
@pytest.mark.parametrize("n_subblocks", [1, 3])
@pytest.mark.parametrize("scale_factor", [1, 2, 3])
@pytest.mark.parametrize("out_channels", [None, 5], ids=["default", "explicit"])
def test_conv_upsample_shapes_and_gradients(
    upsample_mode: str,
    n_subblocks: int,
    scale_factor: int,
    out_channels: int | None,
) -> None:
    """Both modes must produce the expected spatial size and propagate gradients."""
    input_channels = 6
    layer = ConvBlockUpsample(
        input_channels,
        out_channels=out_channels,
        scale_factor=scale_factor,
        n_subblocks=n_subblocks,
        upsample_mode=upsample_mode,
        kernel_size=3,
        norm_type="none",
    )
    x = torch.randn(2, input_channels, 4, 6, requires_grad=True)
    result = layer(x)
    expected_channels = (
        out_channels if out_channels is not None else input_channels // scale_factor
    )
    assert result.shape == (2, expected_channels, 4 * scale_factor, 6 * scale_factor)
    assert torch.isfinite(result).all()

    result.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("n_subblocks", [0, -2])
def test_conv_downsample_rejects_nonpositive_subblock_count(
    n_subblocks: int,
) -> None:
    """Reject invalid encoder depths rather than constructing a broken block."""
    with pytest.raises(ValueError, match="n_subblocks must be at least 1"):
        ConvBlockDownsample(6, n_subblocks=n_subblocks)


@pytest.mark.parametrize("mode", ["bilinear", "shuffle"])
@pytest.mark.parametrize("n_subblocks", [0, -2])
def test_conv_upsample_rejects_nonpositive_subblock_count(
    mode: str,
    n_subblocks: int,
) -> None:
    """Reject invalid decoder depths in both upsampling modes."""
    with pytest.raises(ValueError, match="n_subblocks must be at least 1"):
        ConvBlockUpsample(6, n_subblocks=n_subblocks, upsample_mode=mode)


@pytest.mark.parametrize("mode", ["nearest", "pixelshuffle", "", "cubic"])
def test_conv_upsample_rejects_unsupported_mode(mode: str) -> None:
    """Supported mode names must be explicit, not accidentally accepted."""
    with pytest.raises(ValueError, match="Unsupported upsample_mode"):
        ConvBlockUpsample(4, upsample_mode=mode)


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [
        ("bilinear", nn.Upsample),
        ("shuffle", WeightedUpsample),
    ],
)
def test_conv_upsample_selects_the_requested_resampler(
    mode: str, expected_type: type[nn.Module]
) -> None:
    """The selected mode should determine the first functional block."""
    layer = ConvBlockUpsample(
        4, out_channels=2, scale_factor=2, upsample_mode=mode, norm_type="none"
    )
    assert isinstance(layer.model[0], expected_type)


def test_bilinear_resampling_uses_non_aligned_corners() -> None:
    """The spatial upsampling kernel must agree with a PyTorch reference."""
    layer = ConvBlockUpsample(
        4, out_channels=2, scale_factor=2, upsample_mode="bilinear", norm_type="none"
    )
    x = torch.arange(2 * 4 * 3 * 4, dtype=torch.float32).reshape(2, 4, 3, 4)
    expected = nn.functional.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(layer.model[0](x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("factor", [2, 3])
def test_pixel_shuffle_initialisation_uses_identical_subpixel_kernel_groups(
    factor: int,
) -> None:
    """ICNR initialization should repeat each subpixel filter in shared groups."""
    torch.manual_seed(123)
    layer = WeightedUpsample(3, out_channels=4, upsample_factor=factor, kernel_size=1)
    conv = layer.block[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.weight.shape[0] == 4 * factor**2
    reshaped = conv.weight.detach().reshape(4, factor**2, 3, 1, 1)
    torch.testing.assert_close(
        reshaped, reshaped[:, :1].expand_as(reshaped), rtol=0, atol=0
    )


def test_pixel_shuffle_upsampling_preserves_constant_subpixel_cells() -> None:
    """With a one-pixel kernel, ICNR should not introduce checkerboard values."""
    torch.manual_seed(151)
    layer = WeightedUpsample(3, out_channels=2, upsample_factor=2, kernel_size=1)
    conv = layer.block[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.bias is not None
    with torch.no_grad():
        conv.bias.zero_()
    x = torch.randn(2, 3, 3, 5)
    result = layer(x)
    reduced = result[:, :, ::2, ::2]
    expected = reduced.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
