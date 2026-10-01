from pathlib import Path

import numpy as np
import pytest
import torch

from icenet_mp.models.common import (
    ChannelAdaptor,
    ConvBlockUpsample,
    ConvNormActUpsample,
    Mask,
    NormalisedFold,
    ResBlock,
    ResidualDownsample,
    ResidualUpsample,
    RestrictRange,
)
from icenet_mp.models.common.gated_attention import GatedAttention, GatedAttentionBlock
from icenet_mp.types import RangeRestriction


class TestChannelAdapt:
    def test_identity_when_equal(self) -> None:
        x = torch.randn(2, 5, 3, 3)
        assert torch.equal(ChannelAdaptor(5, 5)(x), x)

    def test_shrink_exact_multiple_averages_contiguous_groups(self) -> None:
        # channel c holds the constant value c; groups of 4 average to 1.5 and 5.5
        x = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1).expand(1, 8, 2, 2)
        y = ChannelAdaptor(8, 2)(x)
        assert y.shape == (1, 2, 2, 2)
        assert torch.allclose(y[:, 0], torch.full((1, 2, 2), 1.5))
        assert torch.allclose(y[:, 1], torch.full((1, 2, 2), 5.5))

    def test_grow_exact_multiple_duplicates_contiguous_groups(self) -> None:
        x = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1).expand(1, 2, 2, 2)
        y = ChannelAdaptor(2, 8)(x)
        assert y.shape == (1, 8, 2, 2)
        assert torch.allclose(y[:, :4], torch.full((1, 4, 2, 2), 10.0))
        assert torch.allclose(y[:, 4:], torch.full((1, 4, 2, 2), 20.0))

    @pytest.mark.parametrize(
        ("in_channels", "out_channels"),
        [(7, 3), (3, 7)],
        ids=["shrink-7to3", "grow-3to7"],
    )
    def test_non_exact_ratio_produces_correct_shape(
        self, in_channels: int, out_channels: int
    ) -> None:
        # Non-divisible ratios (e.g. a multi-dataset combined latent) must not raise.
        x = torch.randn(2, in_channels, 4, 4)
        y = ChannelAdaptor(in_channels, out_channels)(x)
        assert y.shape == (2, out_channels, 4, 4)


class TestConvBlockUpsample:
    @pytest.mark.parametrize("kernel_size", [2, 3, 4, 5], ids=lambda k: f"k{k}")
    @pytest.mark.parametrize("in_channels", [4, 16], ids=lambda c: f"in{c}")
    @pytest.mark.parametrize(
        ("height", "width"), [(8, 8), (12, 20)], ids=["8x8", "12x20"]
    )
    def test_output_shape(
        self, kernel_size: int, in_channels: int, height: int, width: int
    ) -> None:
        layer = ConvBlockUpsample(in_channels, kernel_size=kernel_size)
        x = torch.zeros(1, in_channels, height, width)
        y = layer(x)
        assert y.shape == (1, in_channels // 2, height * 2, width * 2)


class TestConvNormActUpsample:
    @pytest.mark.parametrize("kernel_size", [2, 3, 4, 5], ids=lambda k: f"k{k}")
    @pytest.mark.parametrize("in_channels", [4, 16], ids=lambda c: f"in{c}")
    @pytest.mark.parametrize("out_channels", [5, 13], ids=lambda c: f"out{c}")
    @pytest.mark.parametrize(
        ("height", "width"), [(8, 8), (12, 20)], ids=["8x8", "12x20"]
    )
    def test_output_shape(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
    ) -> None:
        layer = ConvNormActUpsample(in_channels, out_channels, kernel_size=kernel_size)
        x = torch.zeros(1, in_channels, height, width)
        y = layer(x)
        assert y.shape == (1, out_channels, height * 2, width * 2)


class TestGatedAttention:
    @pytest.mark.parametrize("dilation", [0, -1], ids=["zero", "negative"])
    def test_dilation_below_one_raises(self, dilation: int) -> None:
        with pytest.raises(ValueError, match=r"dilation\(.*\) must be at least 1."):
            GatedAttention(channels=4, kernel_size=5, dilation=dilation)

    @pytest.mark.parametrize(
        ("kernel_size", "dilation"), [(1, 2), (2, 3)], ids=["k1-d2", "k2-d3"]
    )
    def test_kernel_size_below_dilation_raises(
        self, kernel_size: int, dilation: int
    ) -> None:
        with pytest.raises(
            ValueError, match=r"kernel_size \(.*\) must be >= dilation \(.*\)."
        ):
            GatedAttention(channels=4, kernel_size=kernel_size, dilation=dilation)


class TestGatedAttentionBlock:
    @pytest.mark.parametrize("value", [-0.1, 1.1], ids=["below_zero", "above_one"])
    @pytest.mark.parametrize(
        "kwarg", ["drop_path_prob", "mlp_drop_prob"], ids=lambda k: k
    )
    def test_probability_outside_unit_interval_raises(
        self, kwarg: str, value: float
    ) -> None:
        probabilities = {"drop_path_prob": 0.5, "mlp_drop_prob": 0.0, kwarg: value}
        with pytest.raises(
            ValueError, match=rf"{kwarg}\(.*\) must be between 0 and 1."
        ):
            GatedAttentionBlock(
                4,
                4,
                kernel_size=5,
                dilation=1,
                mlp_ratio=2.0,
                **probabilities,
            )


class TestNormalisedFold:
    @pytest.mark.parametrize(
        "input_chw", [(4, 57, 67), (1, 60, 50)], ids=["4x57x67", "1x60x50"]
    )
    @pytest.mark.parametrize(
        "latent_hw", [(32, 32), (20, 10)], ids=["latent32x32", "latent20x10"]
    )
    def test_overlap_handling(
        self, input_chw: tuple[int, int, int], latent_hw: tuple[int, int]
    ) -> None:
        input_ones = torch.ones(1, *input_chw)
        input_hw = input_chw[1:]
        unfold = torch.nn.Unfold(
            kernel_size=latent_hw,
            stride=latent_hw,
            padding=latent_hw,
        )
        fold = NormalisedFold(
            output_size=input_hw,
            kernel_size=latent_hw,
            stride=latent_hw,
            padding=latent_hw,
        )
        output = fold(unfold(input_ones))
        assert torch.allclose(output, input_ones)


class TestMask:
    def test_none_returns_input_unchanged(self) -> None:
        mask = Mask(mask_type=None, output_shape=(2, 2))
        values = torch.randn(1, 1, 2, 2)

        assert torch.equal(mask(values), values)

    def test_loaded_mask_is_applied(self, tmp_path: Path) -> None:
        np.save(
            tmp_path / "land_mask.npy",
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        mask = Mask(mask_type="land", output_shape=(2, 2), mask_dir=tmp_path)
        values = torch.ones(1, 1, 2, 2)

        result = mask(values)

        expected = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        assert torch.equal(result, expected)


class TestResBlock:
    @pytest.mark.parametrize(
        "attention_heads", [None, 2], ids=["no_attention", "attention"]
    )
    def test_shape_and_gradient(self, attention_heads: int | None) -> None:
        channels = 8
        block = ResBlock(
            channels, attention_heads=attention_heads, kernel_size=3, padding=1
        )
        assert (block.attn is None) == (attention_heads is None)
        assert (block.attn_norm is None) == (attention_heads is None)
        x = torch.randn(2, channels, 6, 6, requires_grad=True)

        y = block(x)

        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert not torch.equal(x.grad, torch.zeros_like(x.grad))
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} did not receive a gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"


class TestResidualResample:
    @pytest.mark.parametrize(
        ("block_cls", "in_channels", "out_channels", "in_size", "out_size"),
        [(ResidualDownsample, 4, 8, 8, 4), (ResidualUpsample, 8, 4, 4, 8)],
        ids=["downsample", "upsample"],
    )
    @pytest.mark.parametrize(
        "pixel_shuffle", [True, False], ids=["pixel_shuffle", "no_pixel_shuffle"]
    )
    def test_zeroed_parametric_leaves_only_the_shortcut(
        self,
        *,
        block_cls: type[ResidualDownsample | ResidualUpsample],
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
        pixel_shuffle: bool,
    ) -> None:
        block = block_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=2,
            pixel_shuffle=pixel_shuffle,
            kernel_size=1,
        )
        for p in block.parametric.parameters():
            p.data.zero_()

        x = torch.randn(2, in_channels, in_size, in_size)
        assert block.shortcut is not None
        y = block(x)
        assert torch.allclose(y, block.shortcut(x))
        assert y.shape == (2, out_channels, out_size, out_size)


class TestRestrictRange:
    def test_clamp_bounds_values(self) -> None:
        restrict = RestrictRange(RangeRestriction.CLAMP, min_val=0.0, max_val=1.0)
        values = torch.tensor([[[[-1.0, 0.25], [0.75, 2.0]]]])

        result = restrict(values)

        expected = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        "method",
        [RangeRestriction.SIGMOID, RangeRestriction.TANH],
        ids=["sigmoid", "tanh"],
    )
    def test_smooth_restrictions_stay_in_range(
        self, *, method: RangeRestriction
    ) -> None:
        restrict = RestrictRange(method, min_val=-2.0, max_val=3.0)
        values = torch.tensor([[[[-100.0, 0.0, 100.0]]]])

        result = restrict(values)

        assert torch.all(result >= -2.0)
        assert torch.all(result <= 3.0)
