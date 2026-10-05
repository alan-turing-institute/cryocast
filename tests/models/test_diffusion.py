from typing import Any

import pytest
import torch

from cryocast.models.diffusion import GaussianDiffusion, UNetDiffusion


class TestGaussianDiffusion:
    def test_q_sample_t0_returns_clean_input(self) -> None:
        diffusion = GaussianDiffusion(timesteps=4)
        clean = torch.randn(2, 1, 4, 4)
        noise = torch.randn_like(clean)
        timesteps = torch.zeros(2, dtype=torch.long)

        sampled = diffusion.q_sample(clean, timesteps, noise)

        assert torch.equal(sampled, clean)

    def test_q_sample_and_calculate_v_match_definitions(self) -> None:
        diffusion = GaussianDiffusion(timesteps=4)
        clean = torch.randn(2, 1, 4, 4)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([1, 3])
        sqrt_alpha = diffusion.sqrt_alphas_cumprod[timesteps].view(2, 1, 1, 1)
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[timesteps].view(
            2, 1, 1, 1
        )

        sampled = diffusion.q_sample(clean, timesteps, noise)
        velocity = diffusion.calculate_v(clean, noise, timesteps)

        assert torch.allclose(
            sampled, sqrt_alpha * clean + sqrt_one_minus_alpha * noise
        )
        assert torch.allclose(
            velocity, sqrt_alpha * noise - sqrt_one_minus_alpha * clean
        )

    def test_p_sample_t0_uses_posterior_mean_without_noise(self) -> None:
        diffusion = GaussianDiffusion(timesteps=4)
        noisy = torch.randn(2, 1, 4, 4)
        predicted_v = torch.randn_like(noisy)
        timesteps = torch.zeros(2, dtype=torch.long)
        sqrt_alpha = diffusion.sqrt_alphas_cumprod[0]
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[0]
        predicted_clean = sqrt_alpha * noisy - sqrt_one_minus_alpha * predicted_v
        expected = (
            diffusion.posterior_mean_coef1[0] * predicted_clean
            + diffusion.posterior_mean_coef2[0] * noisy
        )

        sampled = diffusion.p_sample(noisy, timesteps, predicted_v)

        assert torch.allclose(sampled, expected)


class TestUNetDiffusion:
    def test_forward_preserves_requested_output_shape(self) -> None:
        model = UNetDiffusion(
            input_channels=4,
            output_channels=2,
            timesteps=4,
            start_out_channels=4,
            dropout_rate=0.0,
        )
        noise = torch.randn(2, 2, 16, 16)
        conditioning = torch.randn(2, 4, 16, 16)
        timesteps = torch.tensor([0, 3])

        output = model(noise, timesteps, conditioning)

        assert output.shape == noise.shape
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"kernel_size": 0}, "Kernel size must be greater than 0"),
            ({"start_out_channels": 0}, "Start out channels must be greater than 0"),
        ],
        ids=["kernel_size", "start_out_channels"],
    )
    def test_invalid_argument_raises(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            UNetDiffusion(input_channels=4, output_channels=1, **kwargs)
