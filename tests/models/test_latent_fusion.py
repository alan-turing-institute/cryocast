import pytest
import torch

from cryocast.models.common import LatentFusion


def _inputs() -> list[torch.Tensor]:
    return [
        torch.randn(2, 3, 4, 6, 6),
        torch.randn(2, 3, 2, 6, 6),
        torch.randn(2, 3, 5, 6, 6),
    ]


class TestLatentFusion:
    def test_concat_matches_existing_behavior(self) -> None:
        inputs = _inputs()
        fusion = LatentFusion([4, 2, 5], mode="concat")

        result = fusion(inputs)

        torch.testing.assert_close(result, torch.cat(inputs, dim=2))
        assert result.shape == (2, 3, 11, 6, 6)
        assert fusion.output_channels == 11

    def test_attention_initialises_as_exact_concat(self) -> None:
        inputs = _inputs()
        fusion = LatentFusion([4, 2, 5], mode="attention")

        result = fusion(inputs)
        weights = fusion.attention_weights(inputs)

        torch.testing.assert_close(result, torch.cat(inputs, dim=2))
        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_attention_reweights_streams_dynamically(self) -> None:
        inputs = _inputs()[:2]
        fusion = LatentFusion([4, 2], mode="attention", temperature=0.5)
        first_head, second_head = fusion.score_heads
        assert isinstance(first_head, torch.nn.Linear)
        assert isinstance(second_head, torch.nn.Linear)
        assert first_head.bias is not None
        assert second_head.bias is not None
        with torch.no_grad():
            torch.nn.init.constant_(first_head.bias, 2.0)
            torch.nn.init.constant_(second_head.bias, -2.0)

        weights = fusion.attention_weights(inputs)
        result = fusion(inputs)

        assert torch.all(weights[..., 0] > weights[..., 1])
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.full_like(weights[..., 0], 2.0),
        )
        torch.testing.assert_close(
            result[:, :, :4],
            inputs[0] * weights[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        )

    def test_attention_parameters_receive_gradients(self) -> None:
        inputs = [tensor.requires_grad_() for tensor in _inputs()[:2]]
        fusion = LatentFusion([4, 2], mode="attention")

        fusion(inputs).square().mean().backward()

        for parameter in fusion.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        for tensor in inputs:
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()

    @pytest.mark.parametrize("mode", ["sum", "cross-attention", ""])
    def test_unknown_mode_raises(self, mode: str) -> None:
        with pytest.raises(ValueError, match="Unknown fusion mode"):
            LatentFusion([4, 2], mode=mode)

    def test_rejects_wrong_channel_count(self) -> None:
        fusion = LatentFusion([4, 2], mode="attention")
        inputs = [
            torch.randn(2, 3, 3, 6, 6),
            torch.randn(2, 3, 2, 6, 6),
        ]

        with pytest.raises(ValueError, match="has 3 channels, expected 4"):
            fusion(inputs)

    def test_rejects_mismatching_spatial_shape(self) -> None:
        fusion = LatentFusion([4, 2], mode="attention")
        inputs = [
            torch.randn(2, 3, 4, 6, 6),
            torch.randn(2, 3, 2, 8, 6),
        ]

        with pytest.raises(ValueError, match="matching batch, time, height and width"):
            fusion(inputs)
