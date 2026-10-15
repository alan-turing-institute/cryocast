import numpy as np
import torch

from cryocast.research.argo_setconv import SparseSetConvConfig, SparseSetConvEncoder
from cryocast.research.argo_sparse import SparseObservations
from cryocast.research.argo_torch import (
    latlon_to_unit_xyz,
    torch_sparse_sequence_from_observations,
)


def _observations(
    latitudes: list[float],
    longitudes: list[float],
    measurements: list[list[float]],
) -> SparseObservations:
    """Create a sparse TEMP/PSAL sample for SetConv tests."""
    return SparseObservations(
        latitudes=np.asarray(latitudes, dtype=np.float64),
        longitudes=np.asarray(longitudes, dtype=np.float64),
        measurements=np.asarray(measurements, dtype=np.float64).reshape((-1, 2)),
        variable_names=("TEMP", "PSAL"),
    )


def _latent_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return a small deterministic latent query grid."""
    return (
        np.array([[70.0, 70.0, 70.0], [72.0, 72.0, 72.0]], dtype=np.float64),
        np.array([[-20.0, -18.0, -16.0], [-20.0, -18.0, -16.0]], dtype=np.float64),
    )


def _encoder(*, length_scale_km: float = 500.0) -> SparseSetConvEncoder:
    """Create the small SetConv encoder used by unit tests."""
    latitudes, longitudes = _latent_grid()
    return SparseSetConvEncoder(
        SparseSetConvConfig(
            latent_channels=4,
            hidden_channels=8,
            length_scale_km=length_scale_km,
            query_chunk_size=2,
        ),
        latitudes,
        longitudes,
    )


def test_setconv_handles_ragged_sequences_and_returns_expected_shape() -> None:
    """Pad different observation counts and produce a finite BTCHW latent tensor."""
    first = _observations(
        [70.0, 71.0],
        [-20.0, -18.0],
        [[-1.0, 33.0], [0.0, 34.0]],
    )
    second = _observations([72.0], [-17.0], [[1.0, 35.0]])
    batch = torch_sparse_sequence_from_observations([[first, second], [second, first]])
    output = _encoder()(batch)
    assert output.shape == (2, 2, 4, 2, 3)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(batch.lengths, torch.tensor([[2, 1], [1, 2]]))


def test_setconv_uses_moving_coordinates_at_each_timestep() -> None:
    """Change point positions across time without changing sensor values."""
    first = _observations([70.0], [-20.0], [[1.0, 34.0]])
    moved = _observations([78.0], [40.0], [[1.0, 34.0]])
    batch = torch_sparse_sequence_from_observations([[first, moved]])
    output = _encoder()(batch)
    assert not torch.allclose(output[:, 0], output[:, 1])


def test_setconv_projection_receives_gradients() -> None:
    """Backpropagate through the learnable projection after sparse aggregation."""
    observations = _observations(
        [70.0, 71.0],
        [-20.0, -18.0],
        [[-1.0, 33.0], [0.0, 34.0]],
    )
    encoder = _encoder()
    batch = torch_sparse_sequence_from_observations([[observations]])
    encoder(batch).square().mean().backward()
    gradients = [parameter.grad for parameter in encoder.parameters()]
    assert gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_setconv_no_observations_returns_zero_latent_tensor() -> None:
    """Keep no data distinct from an observed physical zero."""
    empty = _observations([], [], [])
    batch = torch_sparse_sequence_from_observations([[empty]])
    output = _encoder()(batch)
    assert torch.count_nonzero(output) == 0


def test_setconv_is_invariant_to_sparse_observation_order() -> None:
    """Reordering a point set must not change its latent encoding."""
    observations = _observations(
        [70.0, 71.0, 72.0],
        [-20.0, -18.0, -16.0],
        [[-1.0, 33.0], [0.0, 34.0], [1.0, 35.0]],
    )
    permuted = observations.take(np.array([2, 0, 1], dtype=np.int64))
    encoder = _encoder().eval()
    original = encoder(torch_sparse_sequence_from_observations([[observations]]))
    reordered = encoder(torch_sparse_sequence_from_observations([[permuted]]))
    torch.testing.assert_close(original, reordered, atol=1e-5, rtol=1e-5)


def test_setconv_treats_dateline_neighbours_as_geographically_close() -> None:
    """Aggregate geographically adjacent observations across the dateline."""
    observations = _observations(
        [70.0, 70.0],
        [-179.9, 0.0],
        [[10.0, 30.0], [0.0, 40.0]],
    )
    encoder = _encoder(length_scale_km=100.0)
    batch = torch_sparse_sequence_from_observations([[observations]])
    query_xyz = latlon_to_unit_xyz(
        torch.tensor([70.0]),
        torch.tensor([179.9]),
    )
    features, _ = encoder._aggregate_to_queries(batch, query_xyz)
    assert features[0, 0, 0, 0] > 9.0
    assert features[0, 0, 0, 1] < 31.0


def test_setconv_encode_queries_uses_the_learnable_latent_path() -> None:
    """Encode arbitrary held-out coordinates with the same learnable query path."""
    observations = _observations(
        [70.0, 71.0, 72.0],
        [-20.0, -18.0, -16.0],
        [[-1.0, 33.0], [0.0, 34.0], [1.0, 35.0]],
    )
    encoder = _encoder()
    batch = torch_sparse_sequence_from_observations([[observations]])
    output = encoder.encode_queries(
        batch,
        torch.tensor([70.5, 71.5]),
        torch.tensor([-19.0, -17.0]),
    )
    assert output.shape == (1, 1, 2, 4)
    assert torch.isfinite(output).all()
