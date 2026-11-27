from collections.abc import Callable

import pytest
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cryocast.models.diffusion import LatentNormaliser

CHANNELS = 2


class TestLatentNormaliser:
    def test_first_update_uses_batch_statistics(
        self,
        make_latents: Callable[..., torch.Tensor],
        per_channel_stats: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        latent = make_latents()
        normaliser.update(latent)
        mean, std = per_channel_stats(latent)
        torch.testing.assert_close(normaliser.running_mean, mean)
        torch.testing.assert_close(normaliser.running_var, std**2)
        assert int(normaliser.num_batches_tracked) == 1

    def test_normalised_latents_have_zero_mean_and_unit_std(
        self,
        make_latents: Callable[..., torch.Tensor],
        per_channel_stats: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        latent = make_latents()
        normaliser.update(latent)
        mean, std = per_channel_stats(normaliser.normalise(latent))
        torch.testing.assert_close(mean, torch.zeros(CHANNELS), atol=1e-5, rtol=0)
        torch.testing.assert_close(std, torch.ones(CHANNELS), atol=1e-4, rtol=0)

    def test_denormalise_inverts_normalise(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        latent = make_latents()
        normaliser.update(latent)
        torch.testing.assert_close(
            normaliser.denormalise(normaliser.normalise(latent)), latent
        )

    def test_updates_average_all_batches(
        self,
        make_latents: Callable[..., torch.Tensor],
        per_channel_stats: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        first, second = (
            make_latents(mean=(0.0, 0.0)),
            make_latents(mean=(4.0, 2.0), seed=1),
        )
        normaliser.update(first)
        normaliser.update(second)
        expected = (per_channel_stats(first)[0] + per_channel_stats(second)[0]) / 2
        torch.testing.assert_close(normaliser.running_mean, expected)

    def test_variance_includes_differences_between_batch_means(
        self,
        make_latents: Callable[..., torch.Tensor],
        per_channel_stats: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # Two batches with the same spread but very different means, e.g. two seasons
        normaliser = LatentNormaliser(CHANNELS)
        first = make_latents(mean=(0.0, 0.0), std=(1.0, 1.0))
        second = make_latents(mean=(20.0, -10.0), std=(1.0, 1.0), seed=1)
        normaliser.update(first)
        normaliser.update(second)
        mean, std = per_channel_stats(torch.cat([first, second]))
        torch.testing.assert_close(normaliser.running_mean, mean)
        torch.testing.assert_close(normaliser.running_var, std**2)

    def test_statistics_are_fixed_after_n_batches(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS, n_batches=2)
        normaliser.update(make_latents(mean=(0.0, 0.0)))
        normaliser.update(make_latents(mean=(4.0, 2.0), seed=1))
        state = {k: v.clone() for k, v in normaliser.state_dict().items()}
        normaliser.update(make_latents(mean=(100.0, 100.0), seed=2))
        for key, value in normaliser.state_dict().items():
            assert torch.equal(value, state[key])

    def test_statistics_are_fixed_outside_training(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        normaliser.update(make_latents())
        state = {k: v.clone() for k, v in normaliser.state_dict().items()}
        normaliser.eval()
        normaliser.update(make_latents(mean=(100.0, 100.0)))
        for key, value in normaliser.state_dict().items():
            assert torch.equal(value, state[key])

    def test_statistics_are_saved_in_the_state_dict(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS)
        normaliser.update(make_latents())
        restored = LatentNormaliser(CHANNELS)
        restored.load_state_dict(normaliser.state_dict())
        latent = make_latents(seed=3)
        torch.testing.assert_close(
            restored.normalise(latent), normaliser.normalise(latent)
        )

    def test_is_fitted_after_n_batches(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        normaliser = LatentNormaliser(CHANNELS, n_batches=2)
        normaliser.update(make_latents())
        assert not normaliser.is_fitted
        normaliser.update(make_latents(seed=1))
        assert normaliser.is_fitted

    def test_ema_averaging_keeps_fitted_statistics(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        # With an integer counter, a decay of 0.9 rounds 3 down to 2
        normaliser = LatentNormaliser(CHANNELS, n_batches=3)
        for seed in range(3):
            normaliser.update(make_latents(seed=seed))
        averaged = AveragedModel(
            normaliser, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True
        )
        for _ in range(5):
            averaged.update_parameters(normaliser)
        for name, value in averaged.module.state_dict().items():
            assert torch.equal(value, normaliser.state_dict()[name])
        assert averaged.module.is_fitted

    @pytest.mark.parametrize("n_batches", [0, -1])
    def test_invalid_n_batches_raises(self, n_batches: int) -> None:
        with pytest.raises(ValueError, match="n_batches must be at least 1"):
            LatentNormaliser(CHANNELS, n_batches=n_batches)
