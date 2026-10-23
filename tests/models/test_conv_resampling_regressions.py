"""Regression tests for convolutional encoder and decoder resampling blocks."""

from io import BytesIO

import pytest
import torch
from torch import nn
from torch.func import functional_call

from cryocast.models.common import (
    ConvBlockDownsample,
    ConvBlockUpsample,
    ResidualDownsample,
    ResidualUpsample,
)
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
    x = torch.randn(2, 3, 3, 5)
    result = layer(x)
    reduced = result[:, :, ::2, ::2]
    expected = reduced.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


_RESAMPLERS = (
    "weighted",
    "conv-down",
    "conv-up-bilinear",
    "conv-up-shuffle",
    "residual-down-pixel",
    "residual-down-strided",
    "residual-up-pixel",
    "residual-up-nearest",
)


def _resampler(kind: str, factor: int = 2, out_channels: int | None = 3) -> nn.Module:
    """Construct small real resamplers with smooth activations for numerical checks."""
    if kind == "weighted":
        return WeightedUpsample(2, out_channels=out_channels, upsample_factor=factor)
    if kind == "conv-down":
        return ConvBlockDownsample(
            2,
            out_channels=out_channels,
            scale_factor=factor,
            kernel_size=1,
            n_subblocks=1,
            norm_type="none",
            activation="GELU",
        )
    if kind.startswith("conv-up-"):
        return ConvBlockUpsample(
            2,
            out_channels=out_channels,
            scale_factor=factor,
            kernel_size=1,
            n_subblocks=1,
            norm_type="none",
            activation="GELU",
            upsample_mode=kind.removeprefix("conv-up-"),
        )
    block = (
        ResidualDownsample if kind.startswith("residual-down-") else ResidualUpsample
    )
    assert out_channels is not None
    return block(
        in_channels=2,
        out_channels=out_channels,
        factor=factor,
        pixel_shuffle=kind.endswith("pixel"),
        kernel_size=1,
    )


@pytest.mark.parametrize("kind", _RESAMPLERS[:4])
@pytest.mark.parametrize("factor", [0, -1])
@pytest.mark.parametrize("out_channels", [None, 3], ids=["implicit", "explicit"])
def test_invalid_conv_factors_fail_before_allocation_or_rng_use(
    kind: str,
    factor: int,
    out_channels: int | None,
) -> None:
    """Invalid factors fail at construction, without consuming initialization draws."""
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="factor must be >= 1"):
        _resampler(kind, factor, out_channels)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("kernel_size", [1, 3])
def test_icnr_untouched_initialization_matches_nearest_convolution(
    factor: int,
    kernel_size: int,
) -> None:
    """The initialized weights AND biases must agree across subpixel positions."""
    torch.manual_seed(625)
    layer = WeightedUpsample(
        2,
        out_channels=3,
        upsample_factor=factor,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    ).double()
    conv = layer.block[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.bias is not None
    bias_groups = conv.bias.reshape(3, factor**2)
    torch.testing.assert_close(
        bias_groups,
        bias_groups[:, :1].expand_as(bias_groups),
        rtol=0,
        atol=0,
    )
    for x in (
        torch.zeros(1, 2, 3, 5, dtype=torch.float64),
        torch.linspace(-1, 1, 30, dtype=torch.float64).reshape(1, 2, 3, 5),
    ):
        low_resolution = nn.functional.conv2d(
            x,
            conv.weight[:: factor**2],
            conv.bias[:: factor**2],
            padding=conv.padding,
        )
        expected = low_resolution.repeat_interleave(factor, -2).repeat_interleave(
            factor, -1
        )
        torch.testing.assert_close(layer(x), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("factor", [2, 3])
def test_weighted_shuffle_matches_explicit_subpixel_indices_and_gradients(
    factor: int,
) -> None:
    """Use distinct phase kernels and manual indexing, without a shuffle reference."""
    layer = WeightedUpsample(2, out_channels=3, upsample_factor=factor).double()
    conv = layer.block[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.bias is not None
    with torch.no_grad():
        conv.weight.copy_(
            torch.arange(conv.weight.numel(), dtype=torch.float64).reshape_as(
                conv.weight
            )
            / 100
        )
        conv.bias.copy_(torch.arange(conv.bias.numel(), dtype=torch.float64) / 10)
    x = (
        torch.linspace(-1, 1, 12, dtype=torch.float64)
        .reshape(1, 2, 2, 3)
        .transpose(-1, -2)
        .requires_grad_()
    )
    snapshot = x.detach().clone()
    packed = nn.functional.conv2d(x, conv.weight, conv.bias)
    expected = x.new_empty(1, 3, x.shape[-2] * factor, x.shape[-1] * factor)
    for channel in range(3):
        for row in range(factor):
            for col in range(factor):
                expected[:, channel, row::factor, col::factor] = packed[
                    :, channel * factor**2 + row * factor + col
                ]
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    coefficient = torch.linspace(
        -0.5, 0.5, actual.numel(), dtype=torch.float64
    ).reshape_as(actual)
    parameters = (x, conv.weight, conv.bias)
    actual_grad = torch.autograd.grad(
        (actual * coefficient).sum(), parameters, retain_graph=True
    )
    expected_grad = torch.autograd.grad((expected * coefficient).sum(), parameters)
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("kind", _RESAMPLERS)
def test_all_resamplers_pass_input_and_parameter_finite_difference_checks(
    kind: str,
) -> None:
    """Check actual backward values for the input and every trainable parameter."""
    torch.manual_seed(625)
    layer = _resampler(kind).double().eval()
    x = (
        torch.linspace(-0.4, 0.6, 48, dtype=torch.float64)
        .reshape(1, 2, 4, 6)
        .transpose(-1, -2)
        .detach()
        .requires_grad_()
    )
    snapshot = x.detach().clone()
    names, weights = zip(*layer.named_parameters(), strict=True)

    def evaluate(values: torch.Tensor, *parameters: torch.Tensor) -> torch.Tensor:
        state = dict(layer.named_buffers())
        state.update(zip(names, parameters, strict=True))
        return functional_call(layer, state, (values,), strict=True)

    assert torch.autograd.gradcheck(evaluate, (x, *weights), fast_mode=True)
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["weighted", "residual-up-pixel", "conv-up-shuffle"])
def test_resampler_checkpoint_preserves_next_adam_update(kind: str) -> None:
    """Safe checkpoint loading preserves forecasts, parameters and optimizer moments."""
    torch.manual_seed(625)
    layer = _resampler(kind).double().eval()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
    x = torch.linspace(-1, 1, 48, dtype=torch.float64).reshape(1, 2, 4, 6)

    def step(model: nn.Module, opt: torch.optim.Optimizer) -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        output = model(x)
        output.square().mean().backward()
        opt.step()
        return output.detach()

    step(layer, optimizer)
    stream = BytesIO()
    torch.save(
        {"model": layer.state_dict(), "optimizer": optimizer.state_dict()}, stream
    )
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = _resampler(kind).double().eval()
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(layer(x), restored(x), rtol=0, atol=0)
    torch.testing.assert_close(
        step(layer, optimizer), step(restored, restored_optimizer), rtol=0, atol=0
    )
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    first_state, second_state = optimizer.state_dict(), restored_optimizer.state_dict()
    assert first_state["param_groups"] == second_state["param_groups"]
    for index, moments in first_state["state"].items():
        for name, value in moments.items():
            torch.testing.assert_close(
                value, second_state["state"][index][name], rtol=0, atol=0
            )
