import pytest
import torch

from cryocast.models.diffusion import LatentNormaliser

CHANNELS = 2
SHAPE = (16, 16)


def _latents(
    mean: tuple[float, ...] = (5.0, -2.0),
    std: tuple[float, ...] = (3.0, 0.5),
    n_steps: int = 2,
    seed: int = 438,
) -> torch.Tensor:
    """Return [N, T, C, H, W] latents with a known mean and std per channel."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(4, n_steps, CHANNELS, *SHAPE, generator=generator)
    return noise * torch.tensor(std).view(1, 1, -1, 1, 1) + torch.tensor(mean).view(
        1, 1, -1, 1, 1
    )


def _per_channel(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = latent.transpose(0, 2).flatten(1)
    return values.mean(dim=1), values.std(dim=1, unbiased=False)


class TestLatentNormaliser:
    def test_first_update_uses_batch_statistics(self) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        latent = _latents()
        normaliser.update(latent)
        mean, std = _per_channel(latent)
        torch.testing.assert_close(normaliser.running_mean, mean)
        torch.testing.assert_close(normaliser.running_var, std**2)
        assert int(normaliser.num_batches_tracked) == 1

    def test_normalised_latents_have_zero_mean_and_unit_std(self) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        latent = _latents()
        normaliser.update(latent)
        mean, std = _per_channel(normaliser.normalise(latent))
        torch.testing.assert_close(mean, torch.zeros(CHANNELS), atol=1e-5, rtol=0)
        torch.testing.assert_close(std, torch.ones(CHANNELS), atol=1e-4, rtol=0)

    def test_denormalise_inverts_normalise(self) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        latent = _latents()
        normaliser.update(latent)
        torch.testing.assert_close(
            normaliser.denormalise(normaliser.normalise(latent)), latent
        )

    def test_early_updates_average_all_batches(self) -> None:
        normaliser = LatentNormaliser(CHANNELS, momentum=0.01)
        first, second = _latents(mean=(0.0, 0.0)), _latents(mean=(4.0, 2.0), seed=1)
        normaliser.update(first)
        normaliser.update(second)
        expected = (_per_channel(first)[0] + _per_channel(second)[0]) / 2
        torch.testing.assert_close(normaliser.running_mean, expected)

    def test_later_updates_use_momentum(self) -> None:
        normaliser = LatentNormaliser(CHANNELS, momentum=0.5)
        first, second = _latents(mean=(0.0, 0.0)), _latents(mean=(4.0, 2.0), seed=1)
        normaliser.update(first)
        normaliser.update(second)
        third = _latents(mean=(10.0, 10.0), seed=2)
        before = normaliser.running_mean.clone()
        normaliser.update(third)
        torch.testing.assert_close(
            normaliser.running_mean, 0.5 * before + 0.5 * _per_channel(third)[0]
        )

    def test_statistics_are_fixed_outside_training(self) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        normaliser.update(_latents())
        state = {k: v.clone() for k, v in normaliser.state_dict().items()}
        normaliser.eval()
        normaliser.update(_latents(mean=(100.0, 100.0)))
        for key, value in normaliser.state_dict().items():
            assert torch.equal(value, state[key])

    def test_statistics_are_saved_in_the_state_dict(self) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        normaliser.update(_latents())
        restored = LatentNormaliser(CHANNELS)
        restored.load_state_dict(normaliser.state_dict())
        latent = _latents(seed=3)
        torch.testing.assert_close(
            restored.normalise(latent), normaliser.normalise(latent)
        )

    @pytest.mark.parametrize("momentum", [0.0, -0.1, 1.5])
    def test_invalid_momentum_raises(self, momentum: float) -> None:
        with pytest.raises(ValueError, match="momentum must be in"):
            LatentNormaliser(CHANNELS, momentum=momentum)
