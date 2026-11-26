import re
from typing import Any

import pytest
import torch
from torch import nn

from cryocast.models.decoders import (
    BaseDecoder,
    CNNDecoder,
    DeepCompressionDecoder,
    NaiveLinearDecoder,
    PiecewiseDecoder,
)
from cryocast.types import DataSpace


class TestDecoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=["batch1", "batch2"])
    @pytest.mark.parametrize(
        ("test_decoder_cls", "test_decoder_kwargs"),
        [
            (CNNDecoder, {"n_layers": 1}),
            (NaiveLinearDecoder, {}),
            (PiecewiseDecoder, {}),
        ],
        ids=["CNNDecoder", "NaiveLinearDecoder", "PiecewiseDecoder"],
    )
    @pytest.mark.parametrize(
        "test_latent_chw",
        [(128, 32, 32), (2, 200, 100)],
        ids=["latent128x32x32", "latent2x200x100"],
    )
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 5], ids=["steps1", "steps5"])
    @pytest.mark.parametrize(
        "test_output_chw",
        [(4, 64, 64), (1, 20, 20)],
        ids=["output4x64x64", "output1x20x20"],
    )
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_decoder_cls: type[BaseDecoder],
        test_decoder_kwargs: dict[str, Any],
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_output_chw: tuple[int, int, int],
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        output_space = DataSpace(
            name="output", channels=test_output_chw[0], shape=test_output_chw[1:]
        )
        decoder = test_decoder_cls(
            data_space_in=latent_space,
            data_space_out=output_space,
            **test_decoder_kwargs,
        )
        result: torch.Tensor = decoder.rollout(
            torch.randn(
                test_batch_size,
                test_n_forecast_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            output_space.channels,
            *output_space.shape,
        )


class TestCNNDecoder:
    @pytest.mark.parametrize(
        "test_latent_chw",
        [(3, 32, 32), (5, 200, 100)],
        ids=["latent3x32x32", "latent5x200x100"],
    )
    @pytest.mark.parametrize(
        "test_n_layers", [1, 2, 5], ids=["layers1", "layers2", "layers5"]
    )
    def test_latent_shape_errors(
        self, test_latent_chw: tuple[int, int, int], test_n_layers: int
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        output_space = DataSpace(name="output", shape=(256, 256), channels=4)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The number of input channels {test_latent_chw[0]} must be divisible by {2**test_n_layers}. Without this, it is not possible to apply {test_n_layers} convolutions."
            ),
        ):
            CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_layers=test_n_layers,
            )


class TestDeepCompressionDecoder:
    @pytest.mark.parametrize(
        "pixel_shuffle", [True, False], ids=["pixel_shuffle", "no_pixel_shuffle"]
    )
    @pytest.mark.parametrize(
        ("patch_size", "stride", "hid_channels"),
        [
            (1, 2, (16, 8, 4)),
            (2, 2, (16, 8, 4)),
            (1, 3, (8, 4)),
            (2, 1, (4,)),
        ],
        ids=[
            "patch1-stride2-hid16x8x4",
            "patch2-stride2-hid16x8x4",
            "patch1-stride3-hid8x4",
            "patch2-stride1-hid4",
        ],
    )
    @pytest.mark.parametrize(
        "latent_hw",
        [(4, 4), (2, 6), (5, 3)],
        ids=["latent4x4", "latent2x6", "latent5x3"],
    )
    def test_forward_shape(
        self,
        *,
        pixel_shuffle: bool,
        patch_size: int,
        stride: int,
        hid_channels: tuple[int, ...],
        latent_hw: tuple[int, int],
    ) -> None:
        hid_blocks = [1] * len(hid_channels)
        spatial_factor = patch_size * stride ** (len(hid_channels) - 1)
        latent_space = DataSpace(name="latent", channels=16, shape=latent_hw)
        output_space = DataSpace(
            name="output",
            channels=2,
            shape=(spatial_factor * latent_hw[0], spatial_factor * latent_hw[1]),
        )

        decoder = DeepCompressionDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            patch_size=patch_size,
            pixel_shuffle=pixel_shuffle,
            stride=stride,
        )
        result = decoder.rollout(torch.randn(2, 3, *latent_space.chw))
        assert result.shape == (2, 3, *output_space.chw)

    def test_shape_mismatch_raises(self) -> None:
        stride = 2
        hid_channels = [16, 8, 4]
        hid_blocks = [1, 1, 1]
        spatial_factor = stride ** (len(hid_channels) - 1)
        latent_space = DataSpace(name="latent", channels=16, shape=(4, 4))
        output_hw = spatial_factor * 4
        output_space = DataSpace(
            name="output", channels=2, shape=(output_hw, output_hw + stride)
        )
        with pytest.raises(ValueError, match="will decode latents of shape"):
            DeepCompressionDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                hid_channels=hid_channels,
                hid_blocks=hid_blocks,
                stride=stride,
            )


class TestPiecewiseDecoder:
    @pytest.mark.parametrize(
        "test_patch_size",
        [(2, 2), (3, 3), (7, 3)],
        ids=["patch2x2", "patch3x3", "patch7x3"],
    )
    @pytest.mark.parametrize(
        "test_output_chw",
        [(4, 37, 53), (1, 256, 256)],
        ids=["output4x37x53", "output1x256x256"],
    )
    def test_decoding_gives_same_range_as_input(
        self,
        test_patch_size: tuple[int, int],
        test_output_chw: tuple[int, int, int],
    ) -> None:
        # Generate input and output spaces
        output_space = DataSpace(
            name="output", channels=test_output_chw[0], shape=test_output_chw[1:]
        )
        stride = [max(1, p // 2) for p in test_patch_size]
        n_patches = (
            (output_space.shape[0] + 2 * stride[0] - (test_patch_size[0] - 1) - 1)
            // stride[0]
            + 1
        ) * (
            (output_space.shape[1] + 2 * stride[1] - (test_patch_size[1] - 1) - 1)
            // stride[1]
            + 1
        )
        input_space = DataSpace(
            name="input", channels=test_output_chw[0] * n_patches, shape=test_patch_size
        )

        # Initialise decoder
        decoder = PiecewiseDecoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=input_space,
            data_space_out=output_space,
            use_final_normalisation=False,
            use_hann_window=False,
        )

        # Generate a sequentially increasing input tensor
        input_ntchw = torch.arange(
            1,
            input_space.channels * input_space.shape[0] * input_space.shape[1] + 1,
            dtype=torch.float32,
        ).reshape(1, 1, input_space.channels, *input_space.shape)
        input_min_val = input_ntchw.min().item()
        input_max_val = input_ntchw.max().item()

        # Rollout the decoder and check that the output values are in the same range as the input values
        latent_ntchw = decoder.rollout(input_ntchw)
        assert latent_ntchw.shape == (1, 1, *output_space.chw)
        assert torch.all(input_min_val < latent_ntchw)
        assert torch.all(latent_ntchw < input_max_val)

    def test_projects_channels_without_spatial_convolution(self) -> None:
        decoder = PiecewiseDecoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=DataSpace(name="input", channels=90, shape=(4, 4)),
            data_space_out=DataSpace(name="output", channels=1, shape=(8, 8)),
            use_final_normalisation=False,
        )

        output = decoder(torch.randn(2, 90, 4, 4))

        assert output.shape == (2, 1, 8, 8)
        convolutions = [m for m in decoder.modules() if isinstance(m, nn.Conv2d)]
        assert len(convolutions) == 1
        assert convolutions[0].kernel_size == (1, 1)

    def test_no_projection_when_channels_match(self) -> None:
        # An (8, 8) output from (4, 4) patches at stride 2 needs 25 patches
        decoder = PiecewiseDecoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=DataSpace(name="input", channels=25, shape=(4, 4)),
            data_space_out=DataSpace(name="output", channels=1, shape=(8, 8)),
            use_final_normalisation=False,
        )

        assert not any(isinstance(m, nn.Conv2d) for m in decoder.modules())


class TestDecoderCheckpointing:
    latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
    output_space = DataSpace(name="output", channels=1, shape=(16, 16))

    def _decoder(
        self, decode_chunk_size: int | None, norm_type: str = "groupnorm"
    ) -> CNNDecoder:
        torch.manual_seed(0)
        return CNNDecoder(
            decode_chunk_size=decode_chunk_size,
            data_space_in=self.latent_space,
            data_space_out=self.output_space,
            n_layers=1,
            norm_type=norm_type,
        )

    @pytest.mark.parametrize(
        "test_chunk_size", [1, 3, 10], ids=["chunk1", "chunk3", "chunk10"]
    )
    @pytest.mark.parametrize("test_grad", [True, False], ids=["grad", "no_grad"])
    def test_matches_unchunked_rollout(
        self, test_chunk_size: int, *, test_grad: bool
    ) -> None:
        reference = self._decoder(None)
        chunked = self._decoder(test_chunk_size)
        x = torch.randn(2, 3, self.latent_space.channels, *self.latent_space.shape)

        with torch.set_grad_enabled(test_grad):
            expected = reference.rollout(x)
            result = chunked.rollout(x)
        torch.testing.assert_close(result, expected)

        if test_grad:
            expected.square().sum().backward()
            result.square().sum().backward()
            for p_ref, p_chunked in zip(
                reference.parameters(), chunked.parameters(), strict=True
            ):
                # Checkpointed chunks add per-step gradients in a different order,
                # and conv biases before GroupNorm have zero true gradient (noise).
                torch.testing.assert_close(
                    p_chunked.grad, p_ref.grad, rtol=1e-4, atol=1e-4
                )

    def test_non_positive_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="decode_chunk_size must be positive"):
            self._decoder(0)

    def test_batchnorm_raises(self) -> None:
        decoder = self._decoder(2, norm_type="batchnorm")
        x = torch.randn(1, 3, self.latent_space.channels, *self.latent_space.shape)

        with pytest.raises(ValueError, match="batch normalisation"):
            decoder.rollout(x)
