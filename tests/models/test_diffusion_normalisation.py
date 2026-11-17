import pytest
import torch

from cryocast.models.processors import DiffusionProcessor
from cryocast.types import DataSpace

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


def _processor(
    *, normalise_latents: bool, use_autoregressive: bool = True
) -> DiffusionProcessor:
    return DiffusionProcessor(
        data_space=DataSpace(name="combined", channels=CHANNELS, shape=SHAPE),
        n_forecast_steps=2,
        n_history_steps=2,
        timesteps=4,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0.0,
        use_autoregressive=use_autoregressive,
        normalise_latents=normalise_latents,
        loss=torch.nn.MSELoss(),
    )


class TestDiffusionProcessorNormalisation:
    def test_disabled_by_default(self) -> None:
        processor = DiffusionProcessor(
            data_space=DataSpace(name="combined", channels=CHANNELS, shape=SHAPE),
            n_forecast_steps=2,
            n_history_steps=2,
            timesteps=4,
            start_out_channels=8,
            time_embed_dim=256,
            loss=torch.nn.MSELoss(),
        )
        assert processor.latent_normaliser is None
        assert not any("latent_normaliser" in key for key in processor.state_dict())

    @pytest.mark.parametrize(
        "use_autoregressive", [True, False], ids=["autoregressive", "parallel"]
    )
    def test_training_diffuses_normalised_latents(
        self, monkeypatch: pytest.MonkeyPatch, *, use_autoregressive: bool
    ) -> None:
        processor = _processor(
            normalise_latents=True, use_autoregressive=use_autoregressive
        )
        clean: list[torch.Tensor] = []
        conditions: list[torch.Tensor] = []
        q_sample = processor.diffusion.q_sample
        forward = processor.model.forward

        def record_clean(
            x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
        ) -> torch.Tensor:
            clean.append(x_start.detach())
            return q_sample(x_start, t, noise)

        def record_condition(
            y: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
        ) -> torch.Tensor:
            conditions.append(cond.detach())
            return forward(y, t, cond)

        monkeypatch.setattr(processor.diffusion, "q_sample", record_clean)
        monkeypatch.setattr(processor.model, "forward", record_condition)
        x, y = _latents(seed=1), _latents(seed=2)

        output = processor.rollout(x, y)

        # The statistics come from this batch, so both inputs are close to unit scale
        assert int(processor.latent_normaliser.num_batches_tracked) == 1  # type: ignore[union-attr]
        for tensor in (clean[0], conditions[0]):
            per_channel = tensor.unflatten(1, (-1, CHANNELS))
            mean, std = _per_channel(per_channel)
            assert mean.abs().max() < 0.2
            assert (std - 1).abs().max() < 0.2
        assert output.loss is not None
        assert torch.isfinite(output.loss)

    def test_metrics_prediction_is_in_original_units(self) -> None:
        processor = _processor(normalise_latents=True)
        x, y = _latents(seed=1), _latents(seed=2)
        prediction = processor.rollout(x, y).prediction
        # Untrained predictions are noisy, but must be on the scale of the latents
        assert prediction[:, :, 0].mean() > 2.0
        assert prediction[:, :, 1].mean() < 0.0

    @pytest.mark.parametrize(
        "use_autoregressive", [True, False], ids=["autoregressive", "parallel"]
    )
    def test_inference_returns_original_units(
        self, monkeypatch: pytest.MonkeyPatch, *, use_autoregressive: bool
    ) -> None:
        processor = _processor(
            normalise_latents=True, use_autoregressive=use_autoregressive
        )
        processor.rollout(_latents(seed=1), _latents(seed=2))
        processor.eval()
        histories: list[torch.Tensor] = []

        def zero_sample(y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            histories.append(cond)
            return torch.zeros_like(y)

        monkeypatch.setattr(processor, "_run_reverse_diffusion", zero_sample)
        x = _latents(seed=3)
        prediction = processor.rollout(x).prediction

        # A zero sample in normalised units is the running mean in original units
        normaliser = processor.latent_normaliser
        assert normaliser is not None
        expected = normaliser.running_mean.view(1, 1, -1, 1, 1).expand_as(prediction)
        torch.testing.assert_close(prediction, expected)
        # The history passed to the sampler is normalised
        torch.testing.assert_close(histories[0], normaliser.normalise(x).flatten(1, 2))

    def test_normalised_and_plain_processors_load_their_own_checkpoints(self) -> None:
        trained = _processor(normalise_latents=True)
        trained.rollout(_latents(seed=1), _latents(seed=2))
        restored = _processor(normalise_latents=True)
        restored.load_state_dict(trained.state_dict())
        assert restored.latent_normaliser is not None
        assert trained.latent_normaliser is not None
        torch.testing.assert_close(
            restored.latent_normaliser.running_mean,
            trained.latent_normaliser.running_mean,
        )
        plain = _processor(normalise_latents=False)
        plain.load_state_dict(_processor(normalise_latents=False).state_dict())
