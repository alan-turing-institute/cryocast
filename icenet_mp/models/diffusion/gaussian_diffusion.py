"""GaussianDiffusion: Gaussian Diffusion Process Implementation.

Description:
    Implements the forward and reverse processes of a Denoising Diffusion Probabilistic Model (DDPM),
    with support for both cosine and linear beta schedules.

    Key Definitions:
    - Beta (βₜ): Variance schedule controlling the amount of noise added at each timestep.
      A small βₜ means less noise; a large βₜ means more noise.
    - Alpha (αₜ): Defined as (1 - βₜ), representing the retained signal at each step.
    - Alpha Cumprod (ᾱₜ): Cumulative product of alphas over time, representing the overall
      signal preservation from step 0 to t.
"""

import math

import torch
import torch.nn.functional as F

from icenet_mp.types import BetaSchedule


class GaussianDiffusion:
    """Implements the forward and reverse processes of a Denoising Diffusion Probabilistic Model (DDPM).

    It includes support for cosine and linear beta schedules.
    """

    def __init__(
        self, timesteps: int = 1000, beta_schedule: BetaSchedule = BetaSchedule.COSINE
    ) -> None:
        """Initialize diffusion parameters and precompute useful constants.

        Args:
            timesteps (int): Total number of diffusion steps.
            beta_schedule (BetaSchedule): Type of beta schedule to use. Options: BetaSchedule.LINEAR or BetaSchedule.COSINE.

        """
        self.timesteps = timesteps

        if beta_schedule == BetaSchedule.LINEAR:
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == BetaSchedule.COSINE:
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            msg = (
                f"Unsupported beta_schedule: {beta_schedule}. "
                f"Supported schedules: {list(BetaSchedule)}"
            )
            raise ValueError(msg)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance and mean coefficients
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Compute beta schedule using a cosine function.

        Args:
            timesteps (int): Total number of timesteps.
            s (float): Small offset to prevent singularities near 0.
                      Controls how fast the signal decays: a smaller s results in faster
                      corruption early on; a larger s gives a smoother, slower decay.

        Returns:
            torch.Tensor: Beta values of shape (timesteps,).

        """
        t = torch.linspace(0, 1, timesteps + 1)
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return torch.clip(betas, 0, 0.999)

    def p_sample(
        self, x: torch.Tensor, t: torch.Tensor, pred_v: torch.Tensor
    ) -> torch.Tensor:
        """Perform a single reverse diffusion (denoising) step using the v-prediction parameterization.

        This method implements the reverse process using v-prediction rather than epsilon-prediction,
        where the model predicts velocity v_t instead of noise epsilon.

        Args:
            x (torch.Tensor): Current noisy sample x_t at timestep t (shape: [B,n_classes*n_forecast_days,H,W]).
            t (torch.Tensor): Timesteps for each sample in the batch (shape: [B]).
            pred_v (torch.Tensor): Model's predicted velocity (v_theta) at timestep t (shape: [B,n_classes*n_forecast_days,H,W]).

        Returns:
            torch.Tensor: Sample from the previous timestep x_{t-1} (shape: [B,n_classes*n_forecast_days,H,W]).

        """
        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alpha_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        # Convert the predicted velocity (pred_v) back to the predicted clean image (x_0)
        pred_xstart = sqrt_alpha_t * x - sqrt_one_minus_alpha_t * pred_v

        # Use the predicted x_0 to compute the posterior mean of q(x_{t-1} | x_t, x_0)
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * pred_xstart
            + self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )

        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)

        # Add noise scaled by the posterior variance (except at t=0)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    def _extract(
        self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Extract values from a tensor at specific timesteps t and reshape for broadcasting.

        Args:
            a (torch.Tensor): 1D tensor containing precomputed values (e.g., alpha or beta schedule).
            t (torch.Tensor): Timesteps for each sample in the batch (shape: [B]).
            x_shape (Tuple[int]): Target shape for broadcasting (same as input sample x).

        Returns:
            torch.Tensor: Extracted and reshaped values for each timestep in the batch.

        """
        a = a.to(t.device)
        out = a[t]  # (batch_size,) # Reshape for broadcasting: [batch_size, 1, 1, 1, 1]

        return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))

    def calculate_v(
        self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the velocity variable v_t used in v-prediction.

        This function calculates the target v_t given the clean input x_start,
        the noise ε, and the timestep t, based on the formulation:

            v_t = sqrt(ᾱ_t) * ε - sqrt(1 - ᾱ_t) * x_start

        Args:
            x_start (torch.Tensor): Original clean sample (e.g., groundtruth output).
            noise (torch.Tensor): Gaussian noise added to the sample (ε).
            t (torch.Tensor): Timesteps for each sample in the batch (shape: [B]).

        Returns:
            torch.Tensor: Velocity v_t at each timestep, same shape as x_start.

        """
        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * x_start

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add noise to x_start at timestep t, using the forward diffusion process.

        Args:
            x_start (torch.Tensor): Original input tensor (clean image).
            t (torch.Tensor): Timesteps for each sample in the batch (shape: [B]).
            noise (torch.Tensor, optional): Noise to add. If None, standard Gaussian noise is used.

        Returns:
            torch.Tensor: Noisy sample at timestep t.

        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # For t=0, return exactly x_start (no noise)
        is_t0 = (
            (t == 0)
            .to(dtype=x_start.dtype, device=x_start.device)
            .view(-1, *([1] * (len(x_start.shape) - 1)))
        )

        noisy = (
            sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        )

        return noisy * (1 - is_t0) + x_start * is_t0
