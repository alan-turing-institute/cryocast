import pytest
import torch

from cryocast.models.encoders import MissingDataCNNEncoder
from cryocast.types import DataSpace


def _make_encoder(
    conditioning_dropout_probability: float = 0.0,
    missing_fill_value: float | None = None,
) -> MissingDataCNNEncoder:
    input_space = DataSpace(name="float-argo", channels=2, shape=(8, 8))
    return MissingDataCNNEncoder(
        data_space_in=input_space,
        latent_space=(4, 4),
        n_layers=1,
        conditioning_dropout_probability=conditioning_dropout_probability,
        missing_fill_value=missing_fill_value,
    )


class TestMissingDataCNNEncoder:
    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_rejects_invalid_conditioning_dropout(
        self,
        probability: float,
    ) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            _make_encoder(probability)

    def test_identity_initialisation_preserves_observed_channels(self) -> None:
        encoder = _make_encoder()
        observed = torch.randn(3, 2, 8, 8)
        availability = torch.ones(3, 1, 8, 8)

        adapted = encoder.input_adapter(torch.cat((observed, availability), dim=1))

        torch.testing.assert_close(adapted, observed)

    def test_non_finite_inputs_produce_finite_latents(self) -> None:
        encoder = _make_encoder()
        encoder.eval()
        inputs = torch.randn(2, 3, 2, 8, 8)
        inputs[0, 0] = torch.nan
        inputs[0, 1, 0, :4, :4] = torch.nan
        inputs[1, 2, 1, 0, 0] = torch.inf

        result = encoder.rollout(inputs)

        assert result.shape == (2, 3, 4, 4, 4)
        assert torch.isfinite(result).all()

    def test_missing_fill_value_is_removed_and_marked_unavailable(self) -> None:
        encoder = _make_encoder(missing_fill_value=-1.0)
        encoder.eval()
        captured: list[torch.Tensor] = []

        handle = encoder.input_adapter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        inputs = torch.ones(1, 2, 2, 8, 8)
        inputs[:, 0] = -1.0
        result = encoder.rollout(inputs)
        handle.remove()

        assert torch.isfinite(result).all()
        assert len(captured) == 1
        augmented = captured[0]
        assert torch.count_nonzero(augmented[0]) == 0
        torch.testing.assert_close(augmented[1], torch.ones_like(augmented[1]))

    def test_availability_channel_tracks_partial_missingness(self) -> None:
        encoder = _make_encoder()
        encoder.eval()
        captured: list[torch.Tensor] = []

        handle = encoder.input_adapter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        inputs = torch.ones(1, 1, 2, 8, 8)
        inputs[:, :, 0] = torch.nan
        encoder.rollout(inputs)
        handle.remove()

        assert len(captured) == 1
        augmented = captured[0]
        torch.testing.assert_close(
            augmented[:, -1:],
            torch.full((1, 1, 8, 8), 0.5),
        )

    def test_conditioning_dropout_is_training_only(self) -> None:
        encoder = _make_encoder(conditioning_dropout_probability=1.0)
        inputs = torch.ones(1, 2, 2, 8, 8)
        captured: list[torch.Tensor] = []

        handle = encoder.input_adapter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        encoder.train()
        encoder.rollout(inputs)
        encoder.eval()
        encoder.rollout(inputs)
        handle.remove()

        assert len(captured) == 2
        assert torch.count_nonzero(captured[0]) == 0
        torch.testing.assert_close(
            captured[1],
            torch.ones_like(captured[1]),
        )
