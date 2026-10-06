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
    @pytest.mark.parametrize(
        "set_output_latlons", [False, True], ids=["no_latlons", "output_latlons_only"]
    )
    def test_raises_when_input_latlon_missing(
        self, *, set_output_latlons: bool
    ) -> None:
        lats_out, lons_out = _latlon_grid((2, 2))
        with pytest.raises(
            ValueError,
            match="Input dataset 'source' has 0 latitudes and 0 longitudes but 4",
        ):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(2, 2)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
                latitudes_fn=(lambda: {OUTPUT_NAME: lats_out})
                if set_output_latlons
                else None,
                longitudes_fn=(lambda: {OUTPUT_NAME: lons_out})
                if set_output_latlons
                else None,
            )

    def test_raises_when_output_latlon_missing(self) -> None:
        lats_in, lons_in = _latlon_grid((2, 2))
        with pytest.raises(
            ValueError,
            match="Output dataset 'target' has 0 latitudes and 0 longitudes but 4",
        ):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(2, 2)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {INPUT_NAME: lats_in},
                longitudes_fn=lambda: {INPUT_NAME: lons_in},
            )

    @pytest.mark.parametrize(
        ("n_lats", "n_lons"),
        [(2, 16), (16, 2), (2, 2)],
        ids=["lats-wrong", "lons-wrong", "both-wrong"],
    )
    def test_raises_when_input_latlon_wrong_size(
        self, n_lats: int, n_lons: int
    ) -> None:
        lats_in, lons_in = _latlon_grid((4, 4))
        lats_out, lons_out = _latlon_grid((2, 2))
        with pytest.raises(
            ValueError,
            match=(
                f"Input dataset '{INPUT_NAME}' has {n_lats} latitudes and {n_lons} "
                "longitudes but 16 of each are needed"
            ),
        ):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(4, 4)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {
                    INPUT_NAME: lats_in[:n_lats],
                    OUTPUT_NAME: lats_out,
                },
                longitudes_fn=lambda: {
                    INPUT_NAME: lons_in[:n_lons],
                    OUTPUT_NAME: lons_out,
                },
            )

    @pytest.mark.parametrize(
        ("n_lats", "n_lons"),
        [(2, 9), (9, 2), (2, 2)],
        ids=["lats-wrong", "lons-wrong", "both-wrong"],
    )
    def test_raises_when_output_latlon_wrong_size(
        self, n_lats: int, n_lons: int
    ) -> None:
        lats_in, lons_in = _latlon_grid((2, 2))
        lats_out, lons_out = _latlon_grid((3, 3))
        with pytest.raises(
            ValueError,
            match=(
                f"Output dataset '{OUTPUT_NAME}' has {n_lats} latitudes and {n_lons} "
                "longitudes but 9 of each are needed"
            ),
        ):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(2, 2)),
                latent_space=(3, 3),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {
                    INPUT_NAME: lats_in,
                    OUTPUT_NAME: lats_out[:n_lats],
                },
                longitudes_fn=lambda: {
                    INPUT_NAME: lons_in,
                    OUTPUT_NAME: lons_out[:n_lons],
                },
            )

    def test_raises_when_output_longitudes_missing(self) -> None:
        lats, lons = _latlon_grid((2, 2))
        with pytest.raises(
            ValueError,
            match="Output dataset 'target' has 4 latitudes and 0 longitudes but 4",
        ):
            ReprojectionEncoder(
                data_space_in=DataSpace(name=INPUT_NAME, channels=2, shape=(2, 2)),
                latent_space=(2, 2),
                project_to=OUTPUT_NAME,
                latitudes_fn=lambda: {INPUT_NAME: lats, OUTPUT_NAME: lats},
                longitudes_fn=lambda: {INPUT_NAME: lons},
            )

    @pytest.mark.parametrize(
        ("input_shape", "latent_shape"),
        [
            ((3, 3), (2, 2)),
            ((3, 3), (3, 4)),
            ((4, 4), (2, 2)),
            ((4, 4), (3, 4)),
            ((4, 5), (2, 3)),
        ],
        ids=["3x3->2x2", "3x3->3x4", "4x4->2x2", "4x4->3x4", "4x5->2x3"],
    )
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
        assert torch.all((nn_h >= 0) & (nn_h < input_shape[0]))
        assert torch.all((nn_w >= 0) & (nn_w < input_shape[1]))

    def test_identity_reprojection_maps_to_self(self) -> None:
        # When input and output grids are identical, each output cell should map to
        # the corresponding input cell, so forward reduces to the normalisation alone
        shape = (3, 3)
        channels = 2
        lats = np.repeat([0.0, 30.0, 60.0], 3).tolist()
        lons = np.tile([0.0, 60.0, 120.0], 3).tolist()
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(name=INPUT_NAME, channels=channels, shape=shape),
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

        encoder.eval()
        x_nchw = torch.randn(2, channels, *shape)
        with torch.no_grad():
            assert torch.equal(encoder(x_nchw), encoder.norm(x_nchw))

    def test_forward_applies_nearest_neighbour_indexing(self) -> None:
        # Reproject a 3x3 grid onto a 2x2 grid lying on its corners, so that each output
        # cell takes its value from exactly one known input cell
        channels = 2
        lats_in = np.repeat([0.0, 30.0, 60.0], 3).tolist()
        lons_in = np.tile([0.0, 60.0, 120.0], 3).tolist()
        lats_out = np.repeat([0.0, 60.0], 2).tolist()
        lons_out = np.tile([0.0, 120.0], 2).tolist()
        encoder = ReprojectionEncoder(
            data_space_in=DataSpace(name=INPUT_NAME, channels=channels, shape=(3, 3)),
            latent_space=(2, 2),
            project_to=OUTPUT_NAME,
            latitudes_fn=lambda: {INPUT_NAME: lats_in, OUTPUT_NAME: lats_out},
            longitudes_fn=lambda: {INPUT_NAME: lons_in, OUTPUT_NAME: lons_out},
        )
        # In eval mode a freshly-initialised BatchNorm is (almost) the identity
        encoder.eval()
        x_nchw = torch.randn(2, channels, 3, 3)

        out = encoder(x_nchw)

        expected = torch.stack(
            [
                torch.stack([x_nchw[:, :, 0, 0], x_nchw[:, :, 0, 2]], dim=-1),
                torch.stack([x_nchw[:, :, 2, 0], x_nchw[:, :, 2, 2]], dim=-1),
            ],
            dim=-2,
        )
        assert torch.allclose(out, expected, atol=1e-4)

    @pytest.mark.parametrize(
        "test_batch_size", [1, 2, 5], ids=["batch1", "batch2", "batch5"]
    )
    @pytest.mark.parametrize(
        "test_input_chw", [(2, 4, 4), (3, 6, 6)], ids=["2x4x4", "3x6x6"]
    )
    @pytest.mark.parametrize(
        "test_latent_hw", [(2, 2), (3, 4)], ids=["latent2x2", "latent3x4"]
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 3], ids=["history1", "history3"]
    )
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
