import pytest
import torch

from icenet_mp.models.decoders import PiecewiseDecoder
from icenet_mp.models.encoders import PiecewiseEncoder
from icenet_mp.types import DataSpace


class TestPiecewiseEncodeDecode:
    @pytest.mark.parametrize("test_batch_size", [1, 2, 3])
    @pytest.mark.parametrize(
        "test_input_chw", [(1, 60, 60), (4, 20, 60)], ids=lambda chw: f"chw={chw}"
    )
    @pytest.mark.parametrize(
        "test_patch_size", [(2, 2), (4, 2), (5, 3)], ids=lambda patch: f"patch={patch}"
    )
    @pytest.mark.parametrize("test_timesteps", [1, 2, 3])
    def test_forward(
        self,
        test_batch_size: int,
        test_input_chw: tuple[int, int, int],
        test_patch_size: tuple[int, int],
        test_timesteps: int,
    ) -> None:
        # In order to exactly reproduce the input, we need:
        # - timesteps to be the same in the encoder and decoder
        n_history_steps = test_timesteps
        # - patch size to divide the input size (true for every parametrized case)
        assert test_input_chw[1] % test_patch_size[0] == 0
        assert test_input_chw[2] % test_patch_size[1] == 0
        # - no convolutional blocks, to avoid changing the values
        n_conv_blocks = 0
        input_ntchw = (
            torch.tensor(
                range(
                    1,
                    test_batch_size
                    * n_history_steps
                    * test_input_chw[0]
                    * test_input_chw[1]
                    * test_input_chw[2]
                    + 1,
                )
            )
            .reshape(
                test_batch_size,
                n_history_steps,
                test_input_chw[0],
                test_input_chw[1],
                test_input_chw[2],
            )
            .to(dtype=torch.float)
        )
        input_space = DataSpace(
            name="input", channels=input_ntchw.shape[2], shape=input_ntchw.shape[3:]
        )
        encoder = PiecewiseEncoder(
            conv_subblocks_initial=n_conv_blocks,
            conv_subblocks_final=n_conv_blocks,
            data_space_in=input_space,
            latent_space=test_patch_size,
        )
        latent_ntchw = encoder.rollout(input_ntchw)
        decoder = PiecewiseDecoder(
            conv_subblocks_initial=n_conv_blocks,
            conv_subblocks_final=n_conv_blocks,
            data_space_in=encoder.data_space_out,
            data_space_out=input_space,
            restrict_range="none",
            use_final_normalisation=False,
            use_hann_window=False,
        )
        output_ntchw = decoder.rollout(latent_ntchw, None)
        assert torch.equal(input_ntchw, output_ntchw)

    def test_stacked_conv_blocks_have_connected_gradients(self) -> None:
        """Smoke-test gradient connectivity through stacked encoder/decoder blocks."""
        torch.manual_seed(0)
        n_history_steps = 2
        input_ntchw = torch.randn(2, n_history_steps, 2, 16, 16, requires_grad=True)
        input_space = DataSpace(name="input", channels=2, shape=(16, 16))

        encoder = PiecewiseEncoder(
            conv_activation="SiLU",
            conv_subblocks_initial=3,
            conv_subblocks_final=3,
            data_space_in=input_space,
            latent_space=(4, 4),
        )
        decoder = PiecewiseDecoder(
            conv_activation="SiLU",
            conv_subblocks_initial=3,
            conv_subblocks_final=3,
            data_space_in=encoder.data_space_out,
            data_space_out=input_space,
            restrict_range="none",
            use_final_normalisation=False,
            use_hann_window=False,
        )

        latent_ntchw = encoder.rollout(input_ntchw)
        output_ntchw = decoder.rollout(latent_ntchw, None)
        output_ntchw.square().mean().backward()

        assert input_ntchw.grad is not None
        assert torch.isfinite(input_ntchw.grad).all()
        assert torch.count_nonzero(input_ntchw.grad) > 0

        for component in (encoder, decoder):
            conv_gradients = [
                module.weight.grad
                for module in component.modules()
                if isinstance(module, torch.nn.Conv2d)
            ]
            assert conv_gradients
            for gradient in conv_gradients:
                assert gradient is not None
                assert torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient) > 0

    @pytest.mark.parametrize(
        "test_use_hann_window", [False, True], ids=["flat", "hann"]
    )
    def test_multigroup_round_trip_selects_target_variable(
        self,
        test_use_hann_window: bool,  # noqa: FBT001
    ) -> None:
        n_patches = 25
        n_prefix_channels = n_patches  # a preceding one-channel piecewise encoder
        n_target_channels = 3
        selected_variable = 1

        # Encode a multi-variable target group and place it after the prefix group
        encoder = PiecewiseEncoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=DataSpace(
                name="input", channels=n_target_channels, shape=(8, 8)
            ),
            latent_space=(4, 4),
        )
        source = torch.arange(n_target_channels * 8 * 8, dtype=torch.float32).reshape(
            1, n_target_channels, 8, 8
        )
        combined_latent = torch.cat(
            (torch.zeros(1, n_prefix_channels, 4, 4), encoder(source)), dim=1
        )
        decoder = PiecewiseDecoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=DataSpace(
                name="latent", channels=combined_latent.shape[1], shape=(4, 4)
            ),
            data_space_out=DataSpace(name="output", channels=1, shape=(8, 8)),
            use_final_normalisation=False,
            use_hann_window=test_use_hann_window,
        )

        # Set the 1x1 projection to select one target variable from each patch
        projection = decoder.model[0]
        assert isinstance(projection, torch.nn.Conv2d)
        assert projection.bias is not None
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.zero_()
            for patch_idx in range(n_patches):
                channel_idx = (
                    n_prefix_channels
                    + patch_idx * n_target_channels
                    + selected_variable
                )
                projection.weight[patch_idx, channel_idx] = 1.0

        output = decoder(combined_latent)

        torch.testing.assert_close(
            output, source[:, selected_variable : selected_variable + 1]
        )
