"""Regression tests for irregular-position SetConv gridding and CNN features."""

import math

import pytest
import torch

from cryocast.models.common import SetConv, SetConvCNN


def _grid(height: int = 4, width: int = 5) -> torch.Tensor:
    row, col = torch.meshgrid(
        torch.linspace(0, 1, height),
        torch.linspace(0, 1, width),
        indexing="ij",
    )
    return torch.stack((col, row), dim=-1)


class TestSetConv:
    @pytest.mark.parametrize("time_steps", [None, 3], ids=["static", "moving"])
    def test_output_shape(self, time_steps: int | None) -> None:
        leading_shape = (2,) if time_steps is None else (2, time_steps)
        values = torch.randn(*leading_shape, 7, 3)
        positions = torch.rand(*leading_shape, 7, 2)
        result = SetConv(lengthscale=0.2)(values, positions, _grid())
        assert result.shape == (*leading_shape, 4, 4, 5)
        assert torch.isfinite(result).all()
        assert torch.all(result[..., 0, :, :] >= 0)

    def test_single_observation_matches_normalized_value(self) -> None:
        values = torch.tensor([[[2.0, -3.0]]])
        positions = torch.tensor([[[0.0, 0.0]]])
        grid = torch.tensor([[[0.0, 0.0]]])
        actual = SetConv(lengthscale=0.5)(values, positions, grid)
        torch.testing.assert_close(actual, torch.tensor([[[[1.0]], [[2.0]], [[-3.0]]]]))

    def test_symmetric_observations_average_without_density_bias(self) -> None:
        values = torch.tensor([[[2.0], [8.0]]])
        positions = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
        grid = torch.tensor([[[0.0, 0.0]]])
        actual = SetConv(lengthscale=1.0)(values, positions, grid)
        assert actual[0, 0, 0, 0].item() == pytest.approx(2 * math.exp(-0.5))
        assert actual[0, 1, 0, 0].item() == pytest.approx(5.0)

    def test_masked_nan_padding_is_ignored(self) -> None:
        values = torch.tensor([[[7.0], [float("nan")]]])
        positions = torch.tensor([[[0.5, 0.5], [float("nan"), float("nan")]]])
        mask = torch.tensor([[True, False]])
        layer = SetConv(lengthscale=0.15, learnable=False)
        actual = layer(values, positions, _grid(), mask)
        expected = layer(values[:, :1], positions[:, :1], _grid())
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()

    def test_empty_point_set_has_zero_density_and_signal(self) -> None:
        layer = SetConv(lengthscale=0.2)
        actual = layer(
            torch.empty(2, 0, 2),
            torch.empty(2, 0, 2),
            _grid(),
        )
        assert actual.shape == (2, 3, 4, 5)
        assert torch.count_nonzero(actual) == 0

    def test_all_masked_is_finite_and_zero(self) -> None:
        layer = SetConv(lengthscale=0.1)
        values = torch.full((2, 3, 1), float("nan"))
        positions = torch.full((2, 3, 2), float("nan"))
        actual = layer(values, positions, _grid(), torch.zeros(2, 3, dtype=torch.bool))
        assert torch.count_nonzero(actual) == 0
        assert torch.isfinite(actual).all()

    def test_permutation_invariance(self) -> None:
        generator = torch.Generator().manual_seed(10)
        values = torch.randn(2, 7, 3, generator=generator)
        positions = torch.rand(2, 7, 2, generator=generator)
        mask = torch.tensor([[True] * 6 + [False]] * 2)
        permutation = torch.tensor([6, 2, 5, 0, 3, 1, 4])
        layer = SetConv(lengthscale=0.32)
        original = layer(values, positions, _grid(), mask)
        permuted = layer(
            values[:, permutation],
            positions[:, permutation],
            _grid(),
            mask[:, permutation],
        )
        torch.testing.assert_close(original, permuted, rtol=1e-6, atol=1e-6)

    def test_moving_points_recompute_weights_for_every_time_step(self) -> None:
        values = torch.tensor([[[[1.0]], [[2.0]]]])
        positions = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]]]])
        grid = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        result = SetConv(lengthscale=0.1)(values, positions, grid)
        assert result.shape == (1, 2, 2, 1, 2)
        # The observation density follows the moving sensor.
        assert result[0, 0, 0, 0, 0] > result[0, 0, 0, 0, 1]
        assert result[0, 1, 0, 0, 1] > result[0, 1, 0, 0, 0]
        assert result[0, 0, 1, 0, 0].item() == pytest.approx(1.0)
        assert result[0, 1, 1, 0, 1].item() == pytest.approx(2.0)

    def test_chunk_size_does_not_change_results(self) -> None:
        values = torch.randn(2, 2, 11, 3)
        positions = torch.rand(2, 2, 11, 2)
        mask = torch.tensor([[[True] * 9 + [False] * 2] * 2] * 2)
        small = SetConv(lengthscale=0.22, chunk_size=1, learnable=False)
        large = SetConv(lengthscale=0.22, chunk_size=1024, learnable=False)
        torch.testing.assert_close(
            small(values, positions, _grid(), mask),
            large(values, positions, _grid(), mask),
        )

    def test_gradient_flows_through_values_positions_and_lengthscale(self) -> None:
        values = torch.tensor([[[1.0], [3.0]]], requires_grad=True)
        positions = torch.tensor([[[0.2, 0.3], [0.6, 0.9]]], requires_grad=True)
        layer = SetConv(lengthscale=0.3)
        output = layer(values, positions, _grid())
        output.square().mean().backward()
        for parameter in (values, positions, layer.log_lengthscale):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0

    def test_lengthscale_is_positive_and_can_be_frozen(self) -> None:
        layer = SetConv(lengthscale=0.2, learnable=False)
        assert layer.lengthscale.item() == pytest.approx(0.2)
        assert not list(layer.parameters())
        assert "log_lengthscale" in layer.state_dict()

    def test_far_queries_have_no_signal_when_density_underflows(self) -> None:
        values = torch.tensor([[[8.0]]])
        positions = torch.tensor([[[0.0, 0.0]]])
        grid = torch.tensor([[[100.0, 100.0]]])
        output = SetConv(lengthscale=0.1)(values, positions, grid)
        assert torch.count_nonzero(output) == 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lengthscale": 0.0}, "lengthscale"),
            ({"lengthscale": -1.0}, "lengthscale"),
            ({"lengthscale": float("nan")}, "lengthscale"),
            ({"lengthscale": float("inf")}, "lengthscale"),
            ({"lengthscale": 0.1, "chunk_size": 0}, "chunk_size"),
            ({"lengthscale": 0.1, "eps": 0.0}, "eps"),
        ],
    )
    def test_rejects_invalid_parameters(
        self, kwargs: dict[str, float | int], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            SetConv(**kwargs)

    @pytest.mark.parametrize(
        ("values", "positions", "grid", "match"),
        [
            (torch.ones(2, 3), torch.ones(2, 3, 2), _grid(), "values"),
            (torch.ones(2, 3, 1), torch.ones(2, 4, 2), _grid(), "positions"),
            (
                torch.ones(2, 3, 1),
                torch.ones(2, 3, 2),
                torch.ones(4, 2),
                "grid_positions",
            ),
        ],
    )
    def test_rejects_incompatible_shapes(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        grid: torch.Tensor,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            SetConv(0.2)(values, positions, grid)

    def test_rejects_incompatible_dtypes(self) -> None:
        layer = SetConv(0.2)
        values = torch.ones(1, 2, 1)
        positions = torch.ones(1, 2, 2)
        with pytest.raises(ValueError, match="same dtype"):
            layer(values.double(), positions, _grid())
        with pytest.raises(TypeError, match="floating-point"):
            layer(values, positions.int(), _grid())

    @pytest.mark.parametrize("shape", [(0, 2, 1), (1, 2, 0)])
    def test_rejects_empty_batch_or_channels(self, shape: tuple[int, ...]) -> None:
        values = torch.ones(shape)
        positions = torch.ones(*shape[:-1], 2)
        with pytest.raises(ValueError, match="positive"):
            SetConv(0.2)(values, positions, _grid())

    def test_rejects_wrong_mask_shape_and_dtype(self) -> None:
        values = torch.ones(2, 3, 1)
        positions = torch.ones(2, 3, 2)
        layer = SetConv(0.2)
        with pytest.raises(ValueError, match="valid_mask"):
            layer(values, positions, _grid(), torch.ones(2, 4, dtype=torch.bool))
        with pytest.raises(TypeError, match="boolean"):
            layer(values, positions, _grid(), torch.ones(2, 3))

    @pytest.mark.parametrize("bad_input", ["values", "positions", "grid"])
    def test_rejects_non_finite_unmasked_data(self, bad_input: str) -> None:
        values = torch.ones(1, 1, 1)
        positions = torch.ones(1, 1, 2)
        grid = _grid()
        if bad_input == "values":
            values[0, 0, 0] = float("nan")
        elif bad_input == "positions":
            positions[0, 0, 0] = float("inf")
        else:
            grid[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            SetConv(0.2)(values, positions, grid)


class TestSetConvCNN:
    @pytest.mark.parametrize("time_steps", [None, 3], ids=["static", "moving"])
    def test_output_shape_and_gradients(self, time_steps: int | None) -> None:
        leading_shape = (2,) if time_steps is None else (2, time_steps)
        values = torch.randn(*leading_shape, 5, 2, requires_grad=True)
        positions = torch.rand(*leading_shape, 5, 2)
        model = SetConvCNN(
            in_channels=2,
            out_channels=7,
            hidden_channels=8,
            lengthscale=0.2,
            chunk_size=7,
        )
        output = model(values, positions, _grid())
        assert output.shape == (*leading_shape, 7, 4, 5)
        output.square().mean().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert model.cnn[0].weight.grad is not None
        assert model.setconv.log_lengthscale.grad is not None

    def test_wrong_input_channels(self) -> None:
        model = SetConvCNN(in_channels=3, out_channels=2, lengthscale=0.1)
        with pytest.raises(ValueError, match="Expected 3"):
            model(torch.ones(1, 4, 2), torch.ones(1, 4, 2), _grid())

    @pytest.mark.parametrize(
        ("in_channels", "out_channels", "hidden_channels"),
        [(0, 4, 2), (4, 0, 2), (4, 3, 0)],
    )
    def test_invalid_channels(
        self, in_channels: int, out_channels: int, hidden_channels: int
    ) -> None:
        with pytest.raises(ValueError, match="positive"):
            SetConvCNN(
                in_channels,
                out_channels,
                hidden_channels=hidden_channels,
                lengthscale=0.2,
            )
