from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from icenet_mp.models.decoders import (
    BaseDecoder,
    CNNDecoder,
    DeepCompressionDecoder,
    NaiveLinearDecoder,
    PiecewiseDecoder,
)
from icenet_mp.types import DataSpace


class TestDecoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2])
    @pytest.mark.parametrize(
        "test_decoder_cls", ["CNNDecoder", "NaiveLinearDecoder", "PiecewiseDecoder"]
    )
    @pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (2, 200, 100)])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 5])
    @pytest.mark.parametrize("test_output_chw", [(4, 64, 64), (1, 20, 20)])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_decoder_cls: str,
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
        decoder: BaseDecoder = {
            "CNNDecoder": CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_layers=1,
            ),
            "NaiveLinearDecoder": NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
            ),
            "PiecewiseDecoder": PiecewiseDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
            ),
        }[test_decoder_cls]
        result: torch.Tensor = decoder.rollout(
            torch.randn(
                test_batch_size,
                test_n_forecast_steps,
                latent_space.channels,
                *latent_space.shape,
            ),
            None,
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            output_space.channels,
            *output_space.shape,
        )


class TestCNNDecoder:
    @pytest.mark.parametrize("test_latent_chw", [(3, 32, 32), (5, 200, 100)])
    @pytest.mark.parametrize("test_n_layers", [1, 2, 5])
    def test_latent_shape_errors(
        self, test_latent_chw: tuple[int, int, int], test_n_layers: int
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        output_space = DataSpace(name="output", shape=(256, 256), channels=4)
        with pytest.raises(
            ValueError,
            match=f"The number of input channels {test_latent_chw[0]} must be divisible by {2**test_n_layers}. Without this, it is not possible to apply {test_n_layers} convolutions.",
        ):
            CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_layers=test_n_layers,
            )


class TestDecoderBounded:
    @pytest.mark.parametrize(
        "test_decoder_cls", ["CNNDecoder", "NaiveLinearDecoder", "PiecewiseDecoder"]
    )
    def test_bounded_fixes_values_between_0_and_1(self, test_decoder_cls: str) -> None:
        test_n_forecast_steps = 1
        # latent channels must be divisible by 2 for CNNDecoder with n_layers=1
        latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
        output_space = DataSpace(name="output", channels=1, shape=(16, 16))

        decoder = {
            "CNNDecoder": CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_layers=1,
                restrict_range="sigmoid",
            ),
            "NaiveLinearDecoder": NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                restrict_range="sigmoid",
            ),
            "PiecewiseDecoder": PiecewiseDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                restrict_range="tanh",
            ),
        }[test_decoder_cls]

        extreme_input = torch.full(
            (1, test_n_forecast_steps, latent_space.channels, *latent_space.shape),
            1e10,
            dtype=torch.float32,
        )
        with torch.no_grad():
            out = decoder.rollout(extreme_input, None)

        assert torch.all(out >= 0.0).item()
        assert torch.all(out <= 1.0).item()


class TestDecoderMask:
    """The active grid cell mask, applied via BaseDecoder.finalise()."""

    @staticmethod
    def _spaces() -> tuple[DataSpace, DataSpace]:
        latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
        output_space = DataSpace(name="output", channels=1, shape=(16, 16))
        return latent_space, output_space

    @pytest.mark.parametrize(
        ("mask_type", "mask_filename"),
        [("active", "active_mask.npy"), ("land", "land_mask.npy")],
    )
    def test_mask_loads_and_zeros_masked_cells(
        self, tmp_path: Path, mask_type: str, mask_filename: str
    ) -> None:
        latent_space, output_space = self._spaces()
        # Mask the top half (0), keep the bottom half (1 = active/sea).
        mask = np.ones(output_space.shape, dtype=np.uint8)
        mask[:8, :] = 0
        np.save(tmp_path / mask_filename, mask)

        decoder = NaiveLinearDecoder(
            mask_dir=str(tmp_path),
            data_space_in=latent_space,
            data_space_out=output_space,
            mask_type=mask_type,
        )

        # An identical decoder without a mask gives the unmasked reference output
        unmasked_decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
        )
        unmasked_decoder.load_state_dict(decoder.state_dict())

        assert decoder.mask.mask.shape == output_space.shape
        latent = torch.randn(2, 1, latent_space.channels, *latent_space.shape)
        out = decoder.rollout(latent, None)
        reference = unmasked_decoder.rollout(latent, None)
        # Every masked cell must be exactly zero; unmasked cells are untouched.
        assert torch.all(out[..., :8, :] == 0).item()
        assert torch.equal(out[..., 8:, :], reference[..., 8:, :])

    def test_use_mask_without_file_raises(self, tmp_path) -> None:  # noqa: ANN001
        latent_space, output_space = self._spaces()
        with pytest.raises(FileNotFoundError, match="mask is requested"):
            NaiveLinearDecoder(
                mask_dir=str(tmp_path / "does_not_exist"),
                data_space_in=latent_space,
                data_space_out=output_space,
                mask_type="active",
            )

    def test_mask_type_none_creates_no_buffer(self) -> None:
        latent_space, output_space = self._spaces()
        decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
            mask_type=None,
        )
        assert not hasattr(decoder.mask, "mask")

    def test_unknown_mask_type_raises(self) -> None:
        """A typo in mask_type fails loudly rather than silently disabling masking."""
        latent_space, output_space = self._spaces()
        with pytest.raises(ValueError, match="not a valid MaskType"):
            NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                mask_type="activ",
            )

    def test_use_mask_with_bounded_keeps_masked_cells_exactly_zero(
        self, tmp_path: Path
    ) -> None:
        """Masking is applied AFTER sigmoid, so masked cells are 0, not sigmoid(0)=0.5."""
        latent_space, output_space = self._spaces()
        mask = np.ones(output_space.shape, dtype=np.uint8)
        mask[:8, :] = 0  # top half inactive
        np.save(tmp_path / "active_mask.npy", mask)

        decoder = NaiveLinearDecoder(
            mask_dir=str(tmp_path),
            data_space_in=latent_space,
            data_space_out=output_space,
            mask_type="active",
            restrict_range="sigmoid",
        )
        # An identical decoder without mask or bounding gives the raw reference output
        raw_decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
        )
        raw_decoder.load_state_dict(decoder.state_dict())

        latent = torch.randn(2, 1, latent_space.channels, *latent_space.shape)
        out = decoder.rollout(latent, None)
        reference = torch.sigmoid(raw_decoder.rollout(latent, None))
        # Masked cells must be exactly 0 even with bounding on...
        assert torch.all(out[..., :8, :] == 0).item()
        # ...and active cells are the sigmoid output, strictly inside (0, 1).
        active = out[..., 8:, :]
        assert torch.equal(active, reference[..., 8:, :])
        assert torch.all((active > 0) & (active < 1)).item()

    def test_finalise_is_identity_when_mask_skip_and_bound_off(self) -> None:
        """Finalise makes no changes when masking/skip connection/bounding are off."""
        latent_space, output_space = self._spaces()
        decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
        )
        x = torch.randn(2, output_space.channels, *output_space.shape)
        assert torch.equal(decoder.finalise(x, None), x)


class _SkipConnectionBase:
    """Shared NaiveLinearDecoder fixtures for skip-connection / range-restriction tests.

    Builds a decoder whose main conv has zeroed weights and a constant bias, so its
    pre-skip-connection contribution is a known, spatially-uniform value.
    """

    _LATENT_SPACE = DataSpace(name="latent", channels=4, shape=(8, 8))
    _OUTPUT_SPACE = DataSpace(name="output", channels=1, shape=(8, 8))

    @classmethod
    def _decoder(
        cls,
        *,
        method: str = "none",
        restrict_range: str = "sigmoid",
        restrict_range_min: float | None = None,
        restrict_range_max: float | None = None,
        bias: float = -20.0,
    ) -> NaiveLinearDecoder:
        decoder = NaiveLinearDecoder(
            data_space_in=cls._LATENT_SPACE,
            data_space_out=cls._OUTPUT_SPACE,
            restrict_range=restrict_range,
            restrict_range_min=restrict_range_min,
            restrict_range_max=restrict_range_max,
            skip_connection={"method": method},
        )
        conv = cast("nn.Conv2d", decoder.model[0])
        assert conv.bias is not None
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.fill_(bias)
        return decoder

    @classmethod
    def _latent(cls, n_forecast_steps: int = 1) -> torch.Tensor:
        return torch.randn(
            1, n_forecast_steps, cls._LATENT_SPACE.channels, *cls._LATENT_SPACE.shape
        )

    @classmethod
    def _persistence(cls) -> torch.Tensor:
        """A spatially-varying persistence field in (0, 1), never exactly 0 or 1."""
        space = cls._OUTPUT_SPACE
        numel = space.channels * space.shape[0] * space.shape[1]
        return torch.linspace(0.05, 0.95, numel).reshape(
            1, 1, space.channels, *space.shape
        )


class TestDecoderSkipConnection(_SkipConnectionBase):
    """Ensure the skip-connection order in BaseDecoder.rollout() is correct."""

    @staticmethod
    def _zero(conv_module: nn.Module) -> None:
        conv = cast("nn.Conv2d", conv_module)
        assert conv.bias is not None
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()

    def test_additive_skip_recovers_persistence(self) -> None:
        decoder = self._decoder(method="additive", bias=0.0)
        n_forecast_steps = 2
        persistence = self._persistence()

        out = decoder.rollout(self._latent(n_forecast_steps), persistence)

        assert torch.allclose(
            out, persistence.expand(-1, n_forecast_steps, -1, -1, -1), atol=1e-4
        )

    def test_additive_skip_can_decrease_below_persistence(self) -> None:
        # A negative residual must be reachable, or the decoder could only ever push
        # the prediction above persistence, never below it.
        decoder = self._decoder(method="additive", bias=-0.5)
        persistence = self._persistence()

        out = decoder.rollout(self._latent(), persistence)

        assert torch.all(out < persistence)

    def test_gated_skip_recovers_half_persistence(self) -> None:
        decoder = self._decoder(method="gated")
        # Zero the gate's conv so its pre-activation is 0 everywhere
        # gate = sigmoid(0) = 0.5, so output will be:
        # 0.5 * bounded_main + 0.5 * persistence ~= 0.5 * persistence.
        assert decoder.skip_connection is not None
        self._zero(decoder.skip_connection.gate.block[0])
        persistence = self._persistence()

        out = decoder.rollout(self._latent(), persistence)

        assert torch.allclose(out, 0.5 * persistence, atol=1e-4)

    def test_convolutional_skip_is_clamped_to_valid_range(self) -> None:
        decoder = self._decoder(method="convolutional")
        # Force the fusion's final conv to output a large constant, well outside [0, 1].
        assert decoder.skip_connection is not None
        final_conv = cast("nn.Conv2d", decoder.skip_connection.fusion[-1])
        assert final_conv.bias is not None
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(10.0)

        out = decoder.rollout(self._latent(), self._persistence())

        assert torch.allclose(out, torch.ones_like(out))

    def test_skip_connection_without_persistence_raises(self) -> None:
        decoder = self._decoder(method="additive")
        with pytest.raises(ValueError, match="no persistence input was provided"):
            decoder.rollout(self._latent(), None)


class TestDecoderCustomOutputRange(_SkipConnectionBase):
    """Check that restrict_range_min/max are configured correctly."""

    def test_defaults_to_unit_range(self) -> None:
        decoder = self._decoder()
        assert decoder.range_min == 0
        assert decoder.range_max == 1

    def test_custom_range_is_respected_without_skip_connection(self) -> None:
        decoder = self._decoder(
            restrict_range="tanh",
            restrict_range_min=-5.0,
            restrict_range_max=5.0,
            bias=100.0,
        )
        assert decoder.range_min == -5.0
        assert decoder.range_max == 5.0

        out = decoder.rollout(self._latent(), None)

        # tanh saturates towards +1, rescaled into [-5, 5] -> +5 everywhere.
        assert torch.allclose(out, 5.0 * torch.ones_like(out), atol=1e-3)

    def test_final_clamp_uses_custom_range_with_skip_connection(self) -> None:
        decoder = self._decoder(
            method="convolutional",
            restrict_range="tanh",
            restrict_range_min=-5.0,
            restrict_range_max=5.0,
        )
        assert decoder.skip_connection is not None
        final_conv = cast("nn.Conv2d", decoder.skip_connection.fusion[-1])
        assert final_conv.bias is not None
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(100.0)  # Far outside [-5, 5]

        persistence = torch.zeros(1, 1, *self._OUTPUT_SPACE.chw)
        out = decoder.rollout(self._latent(), persistence)

        assert torch.allclose(out, 5.0 * torch.ones_like(out))

    def test_additive_residual_bound_covers_asymmetric_configured_range(self) -> None:
        # An asymmetric configured range [0, 2] should still give a symmetric [-2, 2]
        # residual bound (max of the absolute endpoints), not [0, 2].
        decoder = self._decoder(
            method="additive",
            restrict_range="tanh",
            restrict_range_min=0.0,
            restrict_range_max=2.0,
            bias=-100.0,
        )
        persistence = torch.full((1, 1, *self._OUTPUT_SPACE.chw), 1.0)

        out = decoder.rollout(self._latent(), persistence)

        # tanh(-100) ~= -1, rescaled by the residual bound of 2 -> residual ~= -2.
        # persistence (1.0) + residual (-2) clamped to the configured [0, 2] -> 0.
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)


class TestDeepCompressionDecoder:
    @pytest.mark.parametrize("pixel_shuffle", [True, False])
    @pytest.mark.parametrize(
        ("patch_size", "stride", "hid_channels"),
        [
            (1, 2, (16, 8, 4)),
            (2, 2, (16, 8, 4)),
            (1, 3, (8, 4)),
            (2, 1, (4,)),
        ],
    )
    @pytest.mark.parametrize("latent_hw", [(4, 4), (2, 6), (5, 3)])
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
        result = decoder.rollout(torch.randn(2, 3, *latent_space.chw), None)
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
    @pytest.mark.parametrize("test_patch_size", [(2, 2), (3, 3), (7, 3)])
    @pytest.mark.parametrize("test_output_chw", [(4, 37, 53), (1, 256, 256)])
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
            restrict_range="none",
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
        latent_ntchw = decoder.rollout(input_ntchw, None)
        assert latent_ntchw.shape == (1, 1, *output_space.chw)
        assert torch.all(input_min_val < latent_ntchw)
        assert torch.all(latent_ntchw < input_max_val)

    def test_clamp_restricts_output_to_unit_range(self) -> None:
        output_space = DataSpace(name="output", channels=1, shape=(4, 4))
        patch_size = (2, 2)
        stride = [max(1, p // 2) for p in patch_size]
        n_patches = (
            (output_space.shape[0] + 2 * stride[0] - (patch_size[0] - 1) - 1)
            // stride[0]
            + 1
        ) * (
            (output_space.shape[1] + 2 * stride[1] - (patch_size[1] - 1) - 1)
            // stride[1]
            + 1
        )
        input_space = DataSpace(name="input", channels=n_patches, shape=patch_size)
        decoder = PiecewiseDecoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=input_space,
            data_space_out=output_space,
            restrict_range="clamp",
            use_final_normalisation=False,
            use_hann_window=False,
        )
        x = torch.full(
            (1, 1, input_space.channels, *input_space.shape), 1e10, dtype=torch.float32
        )
        output = decoder.rollout(x, None)
        assert torch.all(output >= 0.0).item()
        assert torch.all(output <= 1.0).item()
