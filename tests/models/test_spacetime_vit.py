import pytest
import torch

from cryocast.models.processors import SpaceTimeVitProcessor
from cryocast.types import DataSpace, ProcessorOutput


def _make_processor(
    *,
    n_forecast_steps: int = 2,
    n_history_steps: int = 3,
) -> SpaceTimeVitProcessor:
    combined = DataSpace(name="combined", channels=5, shape=(8, 8))
    target = DataSpace(name="target", channels=2, shape=(8, 8))
    return SpaceTimeVitProcessor(
        data_space=combined,
        data_space_target=target,
        target_channel_offset=1,
        n_forecast_steps=n_forecast_steps,
        n_history_steps=n_history_steps,
        dropout=0.0,
        emb_dim=8,
        forecast_spatial_depth=1,
        heads=2,
        mlp_dim=16,
        patch_size=4,
        spatial_depth=1,
        temporal_depth=1,
    )


class TestSpaceTimeVitProcessor:
    @pytest.mark.parametrize(
        ("n_history_steps", "n_forecast_steps"),
        [(1, 1), (3, 2), (15, 10)],
    )
    def test_rollout_shape_and_finiteness(
        self,
        n_history_steps: int,
        n_forecast_steps: int,
    ) -> None:
        processor = _make_processor(
            n_history_steps=n_history_steps,
            n_forecast_steps=n_forecast_steps,
        )
        inputs = torch.randn(1, n_history_steps, 5, 8, 8)

        result = processor.rollout(inputs)

        assert isinstance(result, ProcessorOutput)
        assert result.loss is None
        assert result.prediction.shape == (1, n_forecast_steps, 5, 8, 8)
        assert torch.isfinite(result.prediction).all()

    def test_non_target_conditioning_channels_are_persistent(self) -> None:
        processor = _make_processor(n_forecast_steps=4)
        inputs = torch.randn(2, 3, 5, 8, 8)

        prediction = processor.rollout(inputs).prediction
        persistent = inputs[:, -1:].expand(-1, 4, -1, -1, -1)

        torch.testing.assert_close(prediction[:, :, :1], persistent[:, :, :1])
        torch.testing.assert_close(prediction[:, :, 3:], persistent[:, :, 3:])

    def test_zero_target_delta_recovers_exact_target_persistence(self) -> None:
        processor = _make_processor(n_forecast_steps=4)
        with torch.no_grad():
            processor.delta_head.weight.zero_()
            processor.delta_head.bias.zero_()
        inputs = torch.randn(2, 3, 5, 8, 8)

        prediction = processor.rollout(inputs).prediction[:, :, 1:3]
        persistence = inputs[:, -1:, 1:3].expand(-1, 4, -1, -1, -1)

        torch.testing.assert_close(prediction, persistence)

    def test_checkpoint_parameters_do_not_depend_on_context_length(self) -> None:
        short = _make_processor(n_history_steps=3, n_forecast_steps=2)
        long = _make_processor(n_history_steps=15, n_forecast_steps=10)

        assert sum(parameter.numel() for parameter in short.parameters()) == sum(
            parameter.numel() for parameter in long.parameters()
        )
        long.load_state_dict(short.state_dict(), strict=True)

    def test_forecast_prefix_is_invariant_to_configured_horizon(self) -> None:
        short = _make_processor(n_history_steps=3, n_forecast_steps=2)
        long = _make_processor(n_history_steps=3, n_forecast_steps=10)
        long.load_state_dict(short.state_dict(), strict=True)
        short.eval()
        long.eval()
        inputs = torch.randn(2, 3, 5, 8, 8)

        short_prediction = short.rollout(inputs).prediction
        long_prefix = long.rollout(inputs).prediction[:, :2]

        torch.testing.assert_close(short_prediction, long_prefix, rtol=1e-5, atol=1e-6)

    def test_history_is_embedded_once_for_one_shot_forecast(self) -> None:
        processor = _make_processor(n_history_steps=3, n_forecast_steps=5)
        calls: list[torch.Size] = []

        handle = processor.patch_embed.register_forward_pre_hook(
            lambda _module, args: calls.append(args[0].shape)
        )
        processor.rollout(torch.randn(2, 3, 5, 8, 8))
        handle.remove()

        assert calls == [torch.Size((6, 5, 8, 8))]

    def test_training_target_does_not_leak_into_forecast(self) -> None:
        processor = _make_processor(n_history_steps=3, n_forecast_steps=2)
        processor.eval()
        inputs = torch.randn(2, 3, 5, 8, 8)
        target_a = torch.randn(2, 2, 2, 8, 8)
        target_b = torch.randn(2, 2, 2, 8, 8)

        prediction_a = processor.rollout(inputs, target_a).prediction
        prediction_b = processor.rollout(inputs, target_b).prediction

        torch.testing.assert_close(prediction_a, prediction_b)

    def test_gradients_flow_through_spatial_temporal_and_output_layers(self) -> None:
        processor = _make_processor()
        processor.train()
        inputs = torch.randn(2, 3, 5, 8, 8)
        temporal_layer = processor.temporal_decoder[0]
        assert isinstance(temporal_layer, torch.nn.TransformerDecoderLayer)
        temporal_weight = temporal_layer.multihead_attn.in_proj_weight
        assert temporal_weight is not None

        target_prediction = processor.rollout(inputs).prediction[:, :, 1:3]
        target_prediction.square().mean().backward()

        gradients = (
            processor.patch_embed.proj.weight.grad,
            temporal_weight.grad,
            processor.delta_head.weight.grad,
        )
        for gradient in gradients:
            assert gradient is not None
            assert torch.count_nonzero(gradient).item() > 0

    def test_rejects_missing_target_slice(self) -> None:
        combined = DataSpace(name="combined", channels=5, shape=(8, 8))
        target = DataSpace(name="target", channels=2, shape=(8, 8))
        with pytest.raises(ValueError, match="does not fit inside combined"):
            SpaceTimeVitProcessor(
                data_space=combined,
                data_space_target=target,
                target_channel_offset=None,
                n_forecast_steps=2,
                n_history_steps=3,
                emb_dim=8,
                heads=2,
                patch_size=4,
                temporal_depth=1,
            )

    def test_rejects_wrong_history_length(self) -> None:
        processor = _make_processor(n_history_steps=3)

        with pytest.raises(ValueError, match="Expected input TCHW"):
            processor.rollout(torch.randn(1, 2, 5, 8, 8))
