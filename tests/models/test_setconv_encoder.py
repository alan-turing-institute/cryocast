import pytest
import torch

from icenet_mp.models.encoders import SetConvEncoder
from icenet_mp.types import DataSpace


def _coords() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    latitudes = {
        "sensors": [80.0, 80.0, 82.0],
        "target-grid": [80.0, 80.0, 82.0, 82.0],
    }
    longitudes = {
        "sensors": [-10.0, 10.0, 0.0],
        "target-grid": [-10.0, 10.0, -10.0, 10.0],
    }
    return latitudes, longitudes


def _make_encoder(*, output_channels: int = 3) -> SetConvEncoder:
    latitudes, longitudes = _coords()
    return SetConvEncoder(
        data_space_in=DataSpace(name="sensors", channels=2, shape=(1, 3)),
        latent_space=(2, 2),
        project_to="target-grid",
        output_channels=output_channels,
        length_scale_degrees=3.0,
        latitudes_fn=lambda: latitudes,
        longitudes_fn=lambda: longitudes,
    )


def test_setconv_rollout_shape_and_gradients() -> None:
    encoder = _make_encoder(output_channels=4)
    values = torch.randn(2, 3, 2, 1, 3, requires_grad=True)

    output = encoder.rollout(values)

    assert output.shape == (2, 3, 4, 2, 2)
    output.sum().backward()
    assert values.grad is not None
    assert encoder.log_length_scale_degrees.grad is not None


def test_setconv_ignores_missing_observations() -> None:
    encoder = _make_encoder()
    values = torch.tensor([[[[1.0, float("nan"), 3.0]], [[2.0, 4.0, float("nan")]]]])

    output = encoder(values)

    assert output.shape == (1, 3, 2, 2)
    assert torch.isfinite(output).all()


def test_setconv_recovers_values_at_matching_coordinates() -> None:
    latitudes = {"sensors": [80.0, 80.0], "target-grid": [80.0, 80.0]}
    longitudes = {"sensors": [-20.0, 20.0], "target-grid": [-20.0, 20.0]}
    encoder = SetConvEncoder(
        data_space_in=DataSpace(name="sensors", channels=1, shape=(1, 2)),
        latent_space=(1, 2),
        project_to="target-grid",
        output_channels=1,
        length_scale_degrees=0.1,
        learnable_length_scale=False,
        latitudes_fn=lambda: latitudes,
        longitudes_fn=lambda: longitudes,
    )
    with torch.no_grad():
        encoder.feature_projection.weight.zero_()
        encoder.feature_projection.weight[0, 0, 0, 0] = 1.0
        encoder.feature_projection.bias.zero_()

    output = encoder(torch.tensor([[[[1.0, 3.0]]]]))

    torch.testing.assert_close(output, torch.tensor([[[[1.0, 3.0]]]]), atol=1e-4, rtol=1e-4)


def test_setconv_rejects_missing_or_mismatched_coordinates() -> None:
    input_space = DataSpace(name="sensors", channels=1, shape=(1, 2))

    with pytest.raises(ValueError, match="Missing coordinates"):
        SetConvEncoder(
            data_space_in=input_space,
            latent_space=(1, 1),
            project_to="target-grid",
            latitudes_fn=lambda: {"sensors": [80.0, 81.0]},
            longitudes_fn=lambda: {"sensors": [0.0, 1.0]},
        )

    with pytest.raises(ValueError, match="incompatible coordinate count"):
        SetConvEncoder(
            data_space_in=input_space,
            latent_space=(1, 1),
            project_to="target-grid",
            latitudes_fn=lambda: {
                "sensors": [80.0],
                "target-grid": [80.0],
            },
            longitudes_fn=lambda: {
                "sensors": [0.0],
                "target-grid": [0.0],
            },
        )


def test_setconv_rejects_non_positive_length_scale() -> None:
    latitudes, longitudes = _coords()
    with pytest.raises(ValueError, match="length_scale_degrees must be positive"):
        SetConvEncoder(
            data_space_in=DataSpace(name="sensors", channels=1, shape=(1, 3)),
            latent_space=(2, 2),
            project_to="target-grid",
            length_scale_degrees=0.0,
            latitudes_fn=lambda: latitudes,
            longitudes_fn=lambda: longitudes,
        )
