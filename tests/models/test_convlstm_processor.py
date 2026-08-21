import pytest
import torch

from icenet_mp.models.processors import ConvLSTMCell, ConvLSTMProcessor
from icenet_mp.types import DataSpace, ProcessorOutput


def test_convlstm_cell_preserves_spatial_shape_and_backpropagates() -> None:
    cell = ConvLSTMCell(in_channels=3, hidden_channels=5, kernel_size=3)
    x = torch.randn(2, 3, 8, 10, requires_grad=True)
    hidden = torch.zeros(2, 5, 8, 10)
    state = torch.zeros(2, 5, 8, 10)

    next_hidden, next_state = cell(x, (hidden, state))

    assert next_hidden.shape == (2, 5, 8, 10)
    assert next_state.shape == (2, 5, 8, 10)
    next_hidden.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("n_history_steps", [1, 3])
@pytest.mark.parametrize("n_forecast_steps", [1, 4])
def test_convlstm_processor_rollout_shape(
    n_history_steps: int, n_forecast_steps: int
) -> None:
    latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
    processor = ConvLSTMProcessor(
        data_space=latent_space,
        n_history_steps=n_history_steps,
        n_forecast_steps=n_forecast_steps,
        hidden_channels=6,
        kernel_size=3,
        n_layers=2,
        dropout=0.0,
    )
    x = torch.randn(2, n_history_steps, 4, 8, 8)

    result = processor.rollout(x)

    assert isinstance(result, ProcessorOutput)
    assert result.loss is None
    assert result.prediction.shape == (2, n_forecast_steps, 4, 8, 8)


def test_convlstm_processor_backpropagates_through_history() -> None:
    latent_space = DataSpace(name="latent", channels=2, shape=(6, 6))
    processor = ConvLSTMProcessor(
        data_space=latent_space,
        n_history_steps=3,
        n_forecast_steps=2,
        hidden_channels=4,
        kernel_size=3,
        n_layers=2,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 2, 6, 6, requires_grad=True)

    processor.rollout(x).prediction.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in processor.named_parameters():
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"


def test_zero_residual_head_reduces_to_persistence() -> None:
    latent_space = DataSpace(name="latent", channels=2, shape=(5, 5))
    processor = ConvLSTMProcessor(
        data_space=latent_space,
        n_history_steps=3,
        n_forecast_steps=4,
        hidden_channels=4,
        kernel_size=3,
        n_layers=1,
        residual=True,
    )
    torch.nn.init.zeros_(processor.output_projection.weight)
    torch.nn.init.zeros_(processor.output_projection.bias)
    x = torch.randn(1, 3, 2, 5, 5)

    prediction = processor.rollout(x).prediction
    expected = x[:, -1:].expand_as(prediction)

    torch.testing.assert_close(prediction, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_channels": 0}, "hidden_channels"),
        ({"kernel_size": 2}, "kernel_size"),
        ({"n_layers": 0}, "n_layers"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_convlstm_processor_rejects_invalid_configuration(
    kwargs: dict[str, int | float], message: str
) -> None:
    latent_space = DataSpace(name="latent", channels=2, shape=(4, 4))
    with pytest.raises(ValueError, match=message):
        ConvLSTMProcessor(
            data_space=latent_space,
            n_history_steps=2,
            n_forecast_steps=2,
            **kwargs,
        )


def test_convlstm_processor_validates_input_shape() -> None:
    latent_space = DataSpace(name="latent", channels=2, shape=(4, 4))
    processor = ConvLSTMProcessor(
        data_space=latent_space,
        n_history_steps=2,
        n_forecast_steps=2,
        hidden_channels=4,
    )

    with pytest.raises(ValueError, match="history steps"):
        processor.rollout(torch.randn(1, 3, 2, 4, 4))
    with pytest.raises(ValueError, match="latent channels"):
        processor.rollout(torch.randn(1, 2, 3, 4, 4))
