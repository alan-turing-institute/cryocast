"""Regression tests for learned and parameter-free DC-AE resampling paths."""

import pytest
import torch
from torch import nn

from cryocast.models.common import ResidualDownsample, ResidualUpsample


@pytest.mark.parametrize("pixel_shuffle", [True, False], ids=["pixel", "convolution"])
@pytest.mark.parametrize("shortcut", [True, False], ids=["shortcut", "learned-only"])
@pytest.mark.parametrize("factor", [1, 2, 3], ids=["factor1", "factor2", "factor3"])
def test_residual_downsample_all_paths_preserve_expected_shape_and_gradients(
    pixel_shuffle: bool,  # noqa: FBT001
    shortcut: bool,  # noqa: FBT001
    factor: int,
) -> None:
    """Every downsampling path must be differentiable at multiple factors."""
    layer = ResidualDownsample(
        in_channels=3,
        out_channels=5,
        factor=factor,
        pixel_shuffle=pixel_shuffle,
        shortcut=shortcut,
        kernel_size=1,
    )
    x = torch.randn(2, 3, 12, 18, requires_grad=True)
    y = layer(x)

    assert y.shape == (2, 5, 12 // factor, 18 // factor)
    assert torch.isfinite(y).all()
    assert (layer.shortcut is not None) is shortcut

    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in layer.parametric.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("pixel_shuffle", [True, False], ids=["pixel", "nearest"])
@pytest.mark.parametrize("shortcut", [True, False], ids=["shortcut", "learned-only"])
@pytest.mark.parametrize("factor", [1, 2, 3], ids=["factor1", "factor2", "factor3"])
def test_residual_upsample_all_paths_preserve_expected_shape_and_gradients(
    pixel_shuffle: bool,  # noqa: FBT001
    shortcut: bool,  # noqa: FBT001
    factor: int,
) -> None:
    """Every upsampling path must be differentiable and return the right size."""
    layer = ResidualUpsample(
        in_channels=5,
        out_channels=3,
        factor=factor,
        pixel_shuffle=pixel_shuffle,
        shortcut=shortcut,
        kernel_size=1,
    )
    x = torch.randn(2, 5, 4, 6, requires_grad=True)
    y = layer(x)

    assert y.shape == (2, 3, 4 * factor, 6 * factor)
    assert torch.isfinite(y).all()
    assert (layer.shortcut is not None) is shortcut

    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in layer.parametric.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("pixel_shuffle", [True, False], ids=["pixel", "convolution"])
def test_downsample_shortcut_is_exact_when_learned_branch_is_zero(
    pixel_shuffle: bool,  # noqa: FBT001
) -> None:
    """An unequal channel count must still respect the parameter-free shortcut."""
    layer = ResidualDownsample(
        in_channels=3,
        out_channels=7,
        factor=2,
        pixel_shuffle=pixel_shuffle,
        kernel_size=1,
    )
    assert layer.shortcut is not None
    assert len(list(layer.shortcut.parameters())) == 0
    with torch.no_grad():
        for parameter in layer.parametric.parameters():
            parameter.zero_()

    x = torch.arange(3 * 6 * 8, dtype=torch.float32).reshape(1, 3, 6, 8)
    expected = layer.shortcut(x)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("pixel_shuffle", [True, False], ids=["pixel", "nearest"])
def test_upsample_shortcut_is_exact_when_learned_branch_is_zero(
    pixel_shuffle: bool,  # noqa: FBT001
) -> None:
    """Channel adaptation and pixel rearrangement remain exact and non-trainable."""
    layer = ResidualUpsample(
        in_channels=5,
        out_channels=3,
        factor=2,
        pixel_shuffle=pixel_shuffle,
        kernel_size=1,
    )
    assert layer.shortcut is not None
    assert len(list(layer.shortcut.parameters())) == 0
    with torch.no_grad():
        for parameter in layer.parametric.parameters():
            parameter.zero_()

    x = torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(1, 5, 3, 4)
    expected = layer.shortcut(x)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("block_type", "in_channels", "out_channels", "height", "width"),
    [
        (ResidualDownsample, 3, 7, 8, 6),
        (ResidualUpsample, 7, 3, 4, 3),
    ],
    ids=["downsample", "upsample"],
)
@pytest.mark.parametrize("pixel_shuffle", [True, False], ids=["pixel", "standard"])
def test_disabling_shortcut_removes_all_parameter_free_output(
    block_type: type[ResidualDownsample | ResidualUpsample],
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    pixel_shuffle: bool,  # noqa: FBT001
) -> None:
    """A zeroed learned path must yield zero, not a hidden residual."""
    layer = block_type(
        in_channels=in_channels,
        out_channels=out_channels,
        factor=2,
        pixel_shuffle=pixel_shuffle,
        shortcut=False,
        kernel_size=1,
    )
    with torch.no_grad():
        for parameter in layer.parametric.parameters():
            parameter.zero_()
    assert layer.shortcut is None

    x = torch.randn(2, in_channels, height, width)
    result = layer(x)
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)


def test_strided_downsample_matches_explicit_subsampling_with_unit_kernel() -> None:
    """A stride-two convolution must sample the same locations as x[:, :, ::2]."""
    layer = ResidualDownsample(
        in_channels=1,
        out_channels=1,
        factor=2,
        pixel_shuffle=False,
        shortcut=False,
        kernel_size=1,
    )
    assert isinstance(layer.parametric, nn.Conv2d)
    assert layer.parametric.bias is not None
    with torch.no_grad():
        layer.parametric.weight.fill_(1)
        layer.parametric.bias.zero_()
    x = torch.arange(6 * 8, dtype=torch.float32).reshape(1, 1, 6, 8)
    expected = x[:, :, ::2, ::2]
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


def test_pixel_downsample_matches_explicit_pixel_unshuffle() -> None:
    """The 1x1 convolution and unshuffle must respect pixel ordering."""
    layer = ResidualDownsample(
        in_channels=1,
        out_channels=4,
        factor=2,
        pixel_shuffle=True,
        shortcut=False,
        kernel_size=1,
    )
    assert isinstance(layer.parametric, nn.Sequential)
    assert isinstance(layer.parametric[0], nn.Conv2d)
    assert layer.parametric[0].bias is not None
    with torch.no_grad():
        layer.parametric[0].weight.fill_(1)
        layer.parametric[0].bias.zero_()
    x = torch.arange(4 * 6, dtype=torch.float32).reshape(1, 1, 4, 6)
    expected = nn.PixelUnshuffle(2)(x)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("factor", [0, -1])
@pytest.mark.parametrize(
    ("block_type", "pixel_shuffle"),
    [
        (ResidualDownsample, True),
        (ResidualDownsample, False),
        (ResidualUpsample, True),
        (ResidualUpsample, False),
    ],
    ids=["down-pixel", "down-strided", "up-pixel", "up-nearest"],
)
def test_nonpositive_resampling_factors_are_rejected_at_initialization(
    block_type: type[ResidualDownsample | ResidualUpsample],
    pixel_shuffle: bool,  # noqa: FBT001
    factor: int,
) -> None:
    """Every resampling mode must reject invalid factors consistently."""
    with pytest.raises(ValueError, match="factor must be >= 1"):
        block_type(
            in_channels=3,
            out_channels=6,
            factor=factor,
            pixel_shuffle=pixel_shuffle,
            kernel_size=1,
        )


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("pixel_shuffle", [False, True])
@pytest.mark.parametrize("direction", ["down", "up"])
def test_shortcut_pixel_order_and_gradients_against_manual_indexing(
    factor: int,
    direction: str,
    *,
    pixel_shuffle: bool,
) -> None:
    """Check noncontiguous inputs against explicit phase indexing, not the shortcut."""
    channels = 2
    in_channels = channels if direction == "down" else channels * factor**2
    out_channels = channels * factor**2 if direction == "down" else channels
    block = ResidualDownsample if direction == "down" else ResidualUpsample
    layer = block(
        in_channels=in_channels,
        out_channels=out_channels,
        factor=factor,
        pixel_shuffle=pixel_shuffle,
        kernel_size=1,
    ).double()
    with torch.no_grad():
        for parameter in layer.parametric.parameters():
            parameter.zero_()
    assert layer.shortcut is not None
    assert not list(layer.shortcut.parameters())
    height, width = (2 * factor, 3 * factor) if direction == "down" else (2, 3)
    x = (
        torch.arange(in_channels * height * width, dtype=torch.float64)
        .reshape(1, in_channels, height, width)
        .transpose(-1, -2)
        .detach()
        .requires_grad_()
    )
    snapshot = x.detach().clone()
    if direction == "down":
        expected = torch.cat(
            [
                x[:, channel : channel + 1, row::factor, col::factor]
                for channel in range(channels)
                for row in range(factor)
                for col in range(factor)
            ],
            dim=1,
        )
    else:
        expected = x.new_empty(1, channels, x.shape[-2] * factor, x.shape[-1] * factor)
        for channel in range(channels):
            for row in range(factor):
                for col in range(factor):
                    expected[:, channel, row::factor, col::factor] = x[
                        :, channel * factor**2 + row * factor + col
                    ]
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    coefficient = torch.linspace(-1, 1, actual.numel(), dtype=torch.float64).reshape_as(
        actual
    )
    (observed_gradient,) = torch.autograd.grad(
        (actual * coefficient).sum(), x, retain_graph=True
    )
    (reference_gradient,) = torch.autograd.grad((expected * coefficient).sum(), x)
    torch.testing.assert_close(observed_gradient, reference_gradient, rtol=0, atol=0)
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0)
