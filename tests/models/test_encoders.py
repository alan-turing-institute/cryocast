import pytest
import torch

from cryocast.models.encoders import (
    BaseEncoder,
    CNNEncoder,
    DeepCompressionEncoder,
    NaiveLinearEncoder,
    PiecewiseEncoder,
)
from cryocast.types import DataSpace


class TestEncoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=["batch1", "batch2"])
    @pytest.mark.parametrize(
        "test_encoder_cls",
        [CNNEncoder, NaiveLinearEncoder, PiecewiseEncoder],
        ids=["CNNEncoder", "NaiveLinearEncoder", "PiecewiseEncoder"],
    )
    @pytest.mark.parametrize(
        "test_input_chw",
        [(4, 64, 64), (1, 20, 200)],
        ids=["input4x64x64", "input1x20x200"],
    )
    @pytest.mark.parametrize(
        "test_latent_hw", [(32, 32), (40, 73)], ids=["latent32x32", "latent40x73"]
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 5], ids=["history1", "history5"]
    )
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_encoder_cls: type[CNNEncoder | NaiveLinearEncoder | PiecewiseEncoder],
        test_input_chw: tuple[int, int, int],
        test_latent_hw: tuple[int, int],
        test_n_history_steps: int,
    ) -> None:
        input_space = DataSpace(
            name="input", channels=test_input_chw[0], shape=test_input_chw[1:]
        )
        encoder: BaseEncoder = test_encoder_cls(
            data_space_in=input_space,
            latent_space=test_latent_hw,
        )
        encoder.verify_output_channels()
        result: torch.Tensor = encoder.rollout(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                input_space.channels,
                *input_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            test_n_history_steps,
            encoder.data_space_out.channels,
            *test_latent_hw,
        )


class TestDeepCompressionEncoder:
    @pytest.mark.parametrize(
        "pixel_shuffle", [True, False], ids=["pixel_shuffle", "no_pixel_shuffle"]
    )
    @pytest.mark.parametrize(
        ("patch_size", "stride", "hid_channels"),
        [
            (1, 2, (4, 8, 16)),
            (2, 2, (4, 8, 16)),
            (1, 3, (4, 8)),
            (2, 1, (4,)),
        ],
        ids=[
            "patch1-stride2-hid4x8x16",
            "patch2-stride2-hid4x8x16",
            "patch1-stride3-hid4x8",
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
        input_space = DataSpace(
            name="input",
            channels=3,
            shape=(spatial_factor * latent_hw[0], spatial_factor * latent_hw[1]),
        )

        encoder = DeepCompressionEncoder(
            data_space_in=input_space,
            latent_space=latent_hw,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            patch_size=patch_size,
            pixel_shuffle=pixel_shuffle,
            stride=stride,
        )
        encoder.verify_output_channels()
        result = encoder.rollout(torch.randn(2, 3, *input_space.chw))
        assert result.shape == (2, 3, encoder.data_space_out.channels, *latent_hw)

    def test_shape_mismatch_raises(self) -> None:
        stride = 2
        hid_channels = [4, 8, 16]
        hid_blocks = [1, 1, 1]
        spatial_factor = stride ** (len(hid_channels) - 1)
        latent_hw = (4, 4)
        input_space = DataSpace(
            name="input",
            channels=3,
            shape=(
                spatial_factor * latent_hw[0],
                spatial_factor * (latent_hw[1] + 1),
            ),
        )
        with pytest.raises(ValueError, match="will encode inputs of shape"):
            DeepCompressionEncoder(
                data_space_in=input_space,
                latent_space=latent_hw,
                hid_channels=hid_channels,
                hid_blocks=hid_blocks,
                stride=stride,
            )


class TestPiecewiseEncoder:
    @pytest.mark.parametrize(
        "test_input_chw",
        [(4, 64, 64), (1, 20, 200)],
        ids=["input4x64x64", "input1x20x200"],
    )
    @pytest.mark.parametrize(
        "test_latent_hw", [(32, 32), (40, 73)], ids=["latent32x32", "latent40x73"]
    )
    def test_ones_are_encoded_to_zero_or_one(
        self,
        test_input_chw: tuple[int, int, int],
        test_latent_hw: tuple[int, int],
    ) -> None:
        input_space = DataSpace(
            name="input", channels=test_input_chw[0], shape=test_input_chw[1:]
        )
        encoder = PiecewiseEncoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=input_space,
            latent_space=test_latent_hw,
        )

        # Encode an input of ones
        input_ntchw = torch.ones((1, 1, *input_space.chw))
        latent_ntchw = encoder.rollout(input_ntchw)

        # Assert that the output only contains 0s and 1s, and that there are some of each
        assert torch.all(torch.isin(latent_ntchw, torch.tensor([0, 1])))
        assert torch.count_nonzero(1 - latent_ntchw) > 0  # count 0s
        assert torch.count_nonzero(latent_ntchw) > 0  # count 1s

    @pytest.mark.parametrize(
        "test_patches",
        [
            (
                (2, 2),
                [(1, 2, 6, 7), (2, 3, 7, 8), (7, 8, 12, 13)],
            ),
            (
                (3, 3),
                [(1, 2, 3, 6, 7, 8, 11, 12, 13), (7, 8, 9, 12, 13, 14, 17, 18, 19)],
            ),
        ],
        ids=["patch2x2", "patch3x3"],
    )
    def test_patches_are_extracted(
        self, test_patches: tuple[tuple[int, int], list[tuple[int, int, int, int]]]
    ) -> None:
        # Input is a 5x5 grid of values from 1 to 25
        input_ntchw = (
            torch.tensor(range(1, 26)).reshape(1, 1, 1, 5, 5).to(dtype=torch.float)
        )
        input_space = DataSpace(
            name="input", channels=input_ntchw.shape[2], shape=input_ntchw.shape[3:]
        )
        patch_shape: tuple[int, int] = test_patches[0]
        encoder = PiecewiseEncoder(
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            data_space_in=input_space,
            latent_space=patch_shape,
        )
        latent_ntchw = encoder.rollout(input_ntchw)

        # Extract the p x p patches from the latent space and check that they match the expected patches
        latent_ntchw_patches = [
            latent_ntchw[0, 0, idx] for idx in range(latent_ntchw.shape[2])
        ]
        for expected_values in test_patches[1]:
            expected_patch = (
                torch.tensor(expected_values).reshape(patch_shape).to(dtype=torch.float)
            )
            # One of the latent patches should be the expected patch
            assert any(
                torch.allclose(expected_patch, latent_ntchw_patch)
                for latent_ntchw_patch in latent_ntchw_patches
            )


class TestVerifyOutputChannels:
    def _make_encoder(
        self,
        *,
        declared_channels: int,
        actual_channels: int,
        name: str = "input",
    ) -> NaiveLinearEncoder:
        input_space = DataSpace(name=name, channels=actual_channels, shape=(8, 8))
        encoder = NaiveLinearEncoder(data_space_in=input_space, latent_space=(4, 4))
        # Override the declared output channels to simulate a mismatch with reality
        encoder.data_space_out = DataSpace(
            channels=declared_channels,
            name=encoder.data_space_out.name,
            shape=encoder.data_space_out.shape,
        )
        return encoder

    def test_raises_with_channel_counts_and_encoder_name(self) -> None:
        encoder = self._make_encoder(
            declared_channels=5, actual_channels=3, name="my-dataset"
        )
        with pytest.raises(ValueError, match="declared 5") as excinfo:
            encoder.verify_output_channels()
        assert "declared 5 output channel(s) but forward() actually produced 3" in str(
            excinfo.value
        )
        assert "NaiveLinearEncoder ('my-dataset')" in str(excinfo.value)

    @pytest.mark.parametrize(
        "initial_training_mode", [True, False], ids=["train", "eval"]
    )
    def test_restores_training_mode(self, *, initial_training_mode: bool) -> None:
        encoder = self._make_encoder(declared_channels=5, actual_channels=5)
        encoder.train(initial_training_mode)
        encoder.verify_output_channels()
        assert encoder.training is initial_training_mode

    @pytest.mark.parametrize(
        "initial_training_mode", [True, False], ids=["train", "eval"]
    )
    def test_restores_training_mode_even_when_it_raises(
        self, *, initial_training_mode: bool
    ) -> None:
        encoder = self._make_encoder(declared_channels=5, actual_channels=3)
        encoder.train(initial_training_mode)
        with pytest.raises(ValueError, match="declared 5"):
            encoder.verify_output_channels()
        assert encoder.training is initial_training_mode

    @staticmethod
    def _running_stats(
        batch_norm: torch.nn.BatchNorm2d,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch_norm.running_mean is not None
        assert batch_norm.running_var is not None
        return batch_norm.running_mean.clone(), batch_norm.running_var.clone()

    def test_does_not_mutate_batchnorm_running_stats(self) -> None:
        # verify_output_channels feeds a zero probe through the encoder; it must not
        # leak that probe into BatchNorm running statistics used by real training.
        input_space = DataSpace(name="input", channels=3, shape=(8, 8))
        encoder = NaiveLinearEncoder(data_space_in=input_space, latent_space=(4, 4))
        batch_norm = next(
            module
            for module in encoder.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        )
        running_mean_before, running_var_before = self._running_stats(batch_norm)

        encoder.verify_output_channels()

        running_mean_after, running_var_after = self._running_stats(batch_norm)
        assert torch.equal(running_mean_after, running_mean_before)
        assert torch.equal(running_var_after, running_var_before)
