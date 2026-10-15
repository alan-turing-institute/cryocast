import numpy as np
import torch

from cryocast.research.argo_cross_attention import (
    SparseCrossAttentionConfig,
    SparseCrossAttentionEncoder,
)
from cryocast.research.argo_sparse import SparseObservations
from cryocast.research.argo_torch import (
    spherical_fourier_features,
    torch_sparse_sequence_from_observations,
)


def _observations(
    latitudes: list[float],
    longitudes: list[float],
    measurements: list[list[float]],
) -> SparseObservations:
    """Create a sparse TEMP/PSAL sample for cross-attention tests."""
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


def _encoder() -> SparseCrossAttentionEncoder:
    """Create the small cross-attention encoder used by unit tests."""
    latitudes, longitudes = _latent_grid()
    torch.manual_seed(1)
    return SparseCrossAttentionEncoder(
        SparseCrossAttentionConfig(
            latent_channels=4,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=24,
            query_chunk_size=2,
            dropout=0.0,
        ),
        latitudes,
        longitudes,
    )


def test_cross_attention_handles_ragged_sequences_and_expected_shape() -> None:
    """Handle ragged sequences and return the expected finite latent shape."""
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


def test_cross_attention_uses_moving_coordinates_each_timestep() -> None:
    """Use new observation coordinates independently at every timestep."""
    first = _observations([70.0], [-20.0], [[1.0, 34.0]])
    moved = _observations([78.0], [40.0], [[1.0, 34.0]])
    batch = torch_sparse_sequence_from_observations([[first, moved]])
    output = _encoder()(batch)
    assert not torch.allclose(output[:, 0], output[:, 1])


def test_cross_attention_is_invariant_to_observation_order() -> None:
    """Keep the encoding invariant to sparse observation ordering."""
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


def test_cross_attention_padding_does_not_change_short_sample() -> None:
    """Ignore padded observations through the key-padding mask."""
    short = _observations([70.0], [-20.0], [[1.0, 34.0]])
    long = _observations(
        [70.0, 71.0, 72.0],
        [-20.0, -18.0, -16.0],
        [[-1.0, 33.0], [0.0, 34.0], [1.0, 35.0]],
    )
    encoder = _encoder().eval()
    alone = encoder(torch_sparse_sequence_from_observations([[short]]))[0, 0]
    padded = encoder(torch_sparse_sequence_from_observations([[short], [long]]))[0, 0]
    torch.testing.assert_close(alone, padded, atol=1e-5, rtol=1e-5)


def test_cross_attention_latent_path_receives_gradients() -> None:
    """Backpropagate through every parameter used by the latent encoding path."""
    observations = _observations(
        [70.0, 71.0],
        [-20.0, -18.0],
        [[-1.0, 33.0], [0.0, 34.0]],
    )
    encoder = _encoder()
    encoder(
        torch_sparse_sequence_from_observations([[observations]])
    ).square().mean().backward()
    gradients = [parameter.grad for parameter in encoder.parameters()]
    assert gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_cross_attention_empty_sample_returns_finite_zero_latent() -> None:
    """Represent an empty sparse sample with a finite zero latent tensor."""
    empty = _observations([], [], [])
    output = _encoder()(torch_sparse_sequence_from_observations([[empty]]))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_cross_attention_position_features_are_continuous_at_dateline() -> None:
    """Keep nearby positions continuous when longitude crosses the dateline."""
    east = spherical_fourier_features(
        torch.tensor([70.0]),
        torch.tensor([179.9]),
        3,
    )
    west = spherical_fourier_features(
        torch.tensor([70.0]),
        torch.tensor([-179.9]),
        3,
    )
    assert torch.max(torch.abs(east - west)) < 0.02


def test_cross_attention_encode_queries_uses_shared_latent_path() -> None:
    """Encode held-out coordinates through the same output projection as the grid."""
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
