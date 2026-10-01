import numpy as np
import pytest
import torch

from icenet_mp.models.encoders import ReprojectionEncoder
from icenet_mp.types import DataSpace

INPUT_NAME = "source"
OUTPUT_NAME = "target"


def _latlon_grid(shape: tuple[int, int]) -> tuple[list[float], list[float]]:
    """Flat lat/lon lists for a regular geographic grid."""
    h, w = shape
    lats = np.repeat(np.linspace(0, 60, h), w).tolist()
    lons = np.tile(np.linspace(0, 120, w), h).tolist()
    return lats, lons


class TestReprojectionEncoder:
    def test_raises_when_latlon_not_set(self) -> None:
        with pytest.raises(KeyError):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(4, 4)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
            )

    def test_raises_when_input_latlon_missing(self) -> None:
        lats_out, lons_out = _latlon_grid((2, 2))
        with pytest.raises(KeyError):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(2, 2)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {OUTPUT_NAME: lats_out},
                longitudes_fn=lambda: {OUTPUT_NAME: lons_out},
            )

    def test_raises_when_output_latlon_missing(self) -> None:
        lats_in, lons_in = _latlon_grid((2, 2))
        with pytest.raises(ValueError, match="Cannot reproject"):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(2, 2)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {INPUT_NAME: lats_in},
                longitudes_fn=lambda: {INPUT_NAME: lons_in},
            )

    def test_raises_when_input_lat_wrong_size(self) -> None:
        lats_out, lons_out = _latlon_grid((2, 2))
        with pytest.raises(ValueError, match="Input dataset"):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(4, 4)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {INPUT_NAME: [0.0, 1.0], OUTPUT_NAME: lats_out},
                longitudes_fn=lambda: {INPUT_NAME: [0.0, 1.0], OUTPUT_NAME: lons_out},
            )

    def test_raises_when_output_lat_wrong_size(self) -> None:
        lats_in, lons_in = _latlon_grid((2, 2))
        with pytest.raises(ValueError, match="Output dataset"):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(2, 2)),
                latent_space=(3, 3),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {INPUT_NAME: lats_in, OUTPUT_NAME: [0.0, 1.0]},
                longitudes_fn=lambda: {INPUT_NAME: lons_in, OUTPUT_NAME: [0.0, 1.0]},
            )

    @pytest.mark.parametrize("input_shape", [(3, 3), (4, 4)])
    @pytest.mark.parametrize("latent_shape", [(2, 2), (3, 4)])
    def test_returns_tensors_of_correct_shape(
        self, input_shape: tuple[int, int], latent_shape: tuple[int, int]
    ) -> None:
        lats_in, lons_in = _latlon_grid(input_shape)
        lats_out, lons_out = _latlon_grid(latent_shape)
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=input_shape),
            latent_space=latent_shape,
            project_to=OUTPUT_NAME,
            latitudes_fn=lambda: {INPUT_NAME: lats_in, OUTPUT_NAME: lats_out},
            longitudes_fn=lambda: {INPUT_NAME: lons_in, OUTPUT_NAME: lons_out},
        )
        nn_h, nn_w = encoder.nearest_neighbours(torch.device("cpu"))
        assert nn_h.shape == latent_shape
        assert nn_w.shape == latent_shape

    def test_index_values_are_in_input_range(self) -> None:
        input_shape = (4, 5)
        latent_shape = (2, 3)
        lats_in, lons_in = _latlon_grid(input_shape)
        lats_out, lons_out = _latlon_grid(latent_shape)
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=input_shape),
            latent_space=latent_shape,
            project_to=OUTPUT_NAME,
            latitudes_fn=lambda: {INPUT_NAME: lats_in, OUTPUT_NAME: lats_out},
            longitudes_fn=lambda: {INPUT_NAME: lons_in, OUTPUT_NAME: lons_out},
        )
        nn_h, nn_w = encoder.nearest_neighbours(torch.device("cpu"))
        assert torch.all((nn_h >= 0) & (nn_h < input_shape[0]))
        assert torch.all((nn_w >= 0) & (nn_w < input_shape[1]))

    def test_identity_reprojection_maps_to_self(self) -> None:
        # When input and output grids are identical, each output cell should map to
        # the corresponding input cell
        shape = (3, 3)
        lats = np.repeat([0.0, 30.0, 60.0], 3).tolist()
        lons = np.tile([0.0, 60.0, 120.0], 3).tolist()
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=shape),
            latent_space=shape,
            project_to=OUTPUT_NAME,
            latitudes_fn=lambda: {INPUT_NAME: lats, OUTPUT_NAME: lats},
            longitudes_fn=lambda: {INPUT_NAME: lons, OUTPUT_NAME: lons},
        )
        nn_h, nn_w = encoder.nearest_neighbours(torch.device("cpu"))
        expected_h = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        expected_w = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        assert torch.equal(nn_h, expected_h)
        assert torch.equal(nn_w, expected_w)

    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    @pytest.mark.parametrize("test_input_chw", [(2, 4, 4), (3, 6, 6)])
    @pytest.mark.parametrize("test_latent_hw", [(2, 2), (3, 4)])
    @pytest.mark.parametrize("test_n_history_steps", [1, 3])
    def test_rollout_output_shape(
        self,
        test_batch_size: int,
        test_input_chw: tuple[int, int, int],
        test_latent_hw: tuple[int, int],
        test_n_history_steps: int,
    ) -> None:
        channels, h, w = test_input_chw
        input_shape = (h, w)
        lats_in, lons_in = _latlon_grid(input_shape)
        lats_out, lons_out = _latlon_grid(test_latent_hw)
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(
                name=INPUT_NAME, channels=channels, shape=input_shape
            ),
            latent_space=test_latent_hw,
            project_to=OUTPUT_NAME,
            latitudes_fn=lambda: {INPUT_NAME: lats_in, OUTPUT_NAME: lats_out},
            longitudes_fn=lambda: {INPUT_NAME: lons_in, OUTPUT_NAME: lons_out},
        )
        x = torch.randn(test_batch_size, test_n_history_steps, channels, h, w)
        out = encoder.rollout(x)
        assert out.shape == (
            test_batch_size,
            test_n_history_steps,
            channels,
            *test_latent_hw,
        )

    def test_forward_applies_nearest_neighbour_indexing(self) -> None:
        # Use an identity reprojection on a 3x3 grid and verify values are sampled correctly
        shape = (3, 3)
        channels = 1
        lats = np.repeat([0.0, 30.0, 60.0], 3).tolist()
        lons = np.tile([0.0, 60.0, 120.0], 3).tolist()
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(name=INPUT_NAME, channels=channels, shape=shape),
            latent_space=shape,
            project_to=OUTPUT_NAME,
            latitudes_fn=lambda: {INPUT_NAME: lats, OUTPUT_NAME: lats},
            longitudes_fn=lambda: {INPUT_NAME: lons, OUTPUT_NAME: lons},
        )
        encoder.eval()
        x_nchw = torch.randn(2, channels, *shape)
        with torch.no_grad():
            assert torch.equal(encoder(x_nchw), encoder.norm(x_nchw))

    def test_forward_channels_preserved(self) -> None:
        input_shape = (4, 4)
        latent_shape = (2, 2)
        channels = 5
        lats_in, lons_in = _latlon_grid(input_shape)
        lats_out, lons_out = _latlon_grid(latent_shape)
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(
                name=INPUT_NAME, channels=channels, shape=input_shape
            ),
            latent_space=latent_shape,
            project_to=OUTPUT_NAME,
            latitudes_fn=lambda: {INPUT_NAME: lats_in, OUTPUT_NAME: lats_out},
            longitudes_fn=lambda: {INPUT_NAME: lons_in, OUTPUT_NAME: lons_out},
        )
        x = torch.randn(3, channels, *input_shape)
        out = encoder(x)
        assert out.shape[1] == channels
