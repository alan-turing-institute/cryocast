import pytest
import torch

from icenet_mp.models.decoders import PiecewiseDecoder
from icenet_mp.models.encoders import PiecewiseEncoder
from icenet_mp.types import DataSpace


def _make_decoder(in_channels: int) -> PiecewiseDecoder:
    return PiecewiseDecoder(
        data_space_in=DataSpace(name="combined", channels=in_channels, shape=(4, 4)),
        data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
        conv_subblocks_initial=0,
        conv_subblocks_final=0,
        use_final_normalisation=False,
    )


def test_naive_piecewise_decoder_projects_channels_without_spatial_convolution() -> (
    None
):
    """Reduce extra latent channels with a single 1x1 projection."""
    decoder = _make_decoder(in_channels=90)
    x = torch.randn(2, 90, 4, 4)

    output = decoder(x)

    assert output.shape == (2, 1, 8, 8)
    convolutions = [m for m in decoder.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convolutions) == 1
    assert convolutions[0].kernel_size == (1, 1)


def test_naive_piecewise_decoder_has_no_convolution_when_channels_match() -> None:
    """Skip the projection when the latent already has one channel per patch."""
    decoder = _make_decoder(in_channels=25)

    assert not any(isinstance(m, torch.nn.Conv2d) for m in decoder.modules())


@pytest.mark.parametrize("use_hann_window", [False, True], ids=["flat", "hann"])
def test_naive_piecewise_round_trip_selects_target_variable(
    use_hann_window: bool,  # noqa: FBT001
) -> None:
    """Recover a target variable from a combined latent with a selecting projection."""
    target_space = DataSpace(name="sic", channels=3, shape=(8, 8))
    encoder = PiecewiseEncoder(
        data_space_in=target_space,
        latent_space=(4, 4),
        conv_subblocks_initial=0,
        conv_subblocks_final=0,
    )
    source = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    target_latent = encoder(source)

    # Represent a preceding one-channel piecewise encoder: 25 patches precede the
    # target group's 75 channels in the combined latent.
    prefix = torch.zeros(1, 25, 4, 4)
    combined_latent = torch.cat((prefix, target_latent), dim=1)
    decoder = PiecewiseDecoder(
        data_space_in=DataSpace(
            name="combined", channels=combined_latent.shape[1], shape=(4, 4)
        ),
        data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
        conv_subblocks_initial=0,
        conv_subblocks_final=0,
        use_final_normalisation=False,
        use_hann_window=use_hann_window,
    )

    # Set the projection to select variable 1 of the target group from each patch
    projection = decoder.model[0]
    assert isinstance(projection, torch.nn.Conv2d)
    assert projection.bias is not None
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()
        for patch_idx in range(25):
            projection.weight[patch_idx, 25 + patch_idx * 3 + 1] = 1.0

    output = decoder(combined_latent)

    torch.testing.assert_close(output, source[:, 1:2])
