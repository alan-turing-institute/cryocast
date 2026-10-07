"""Latent-space diffusion processor.

This module provides a diffusion-based processor for the EncodeProcessDecode
pipeline. It performs denoising in the shared latent space produced by the
encoders, using v-parameterisation with `UNetDiffusion` and
`GaussianDiffusion`.

Training always uses the DDPM v-prediction loss. Sampling (validation, test
and predict) uses either DDPM or DDIM, selected by ``ddim_steps`` and ``eta``:

* ``ddim_steps=None`` (i.e. ``timesteps``) with ``eta=1.0``: ancestral DDPM
  sampling through every trained timestep.
* Otherwise: DDIM sampling through ``ddim_steps`` evenly-spaced trained
  timesteps. ``eta=0`` is deterministic; ``eta=1`` adds DDPM-equivalent noise.

The sampler does not change the network weights, so a checkpoint can be
sampled with different settings from the ones it was trained with.

Evaluating a DDPM-trained run with DDIM: ``cryocast evaluate`` builds the
processor from the run's saved ``files/model_config.yaml`` (the processor is
not stored in the checkpoint's hyperparameters), so command-line overrides of
``model.processor.ddim_steps`` / ``eta`` are ignored. Instead, set them in
that file before evaluating, and restore it afterwards. Any
``1 <= ddim_steps <= timesteps`` and ``0 <= eta <= 1`` are valid; the values
below are only an example (50 steps, deterministic DDIM)::

    cp <run_dir>/files/model_config.yaml <run_dir>/files/model_config.yaml.bak
    # in model_config.yaml, e.g.: model.processor.ddim_steps: 50, model.processor.eta: 0.0
    uv run cryocast evaluate --checkpoint <run_dir>/checkpoints/<ckpt> --config-name <config>
    mv <run_dir>/files/model_config.yaml.bak <run_dir>/files/model_config.yaml

The processor supports two forecasting modes:

* Autoregressive: generates forecast steps sequentially, updating the history
  with each predicted combined latent.
* Parallel: generates all forecast steps jointly in a single diffusion
  process, with forecast steps folded into the channel dimension.

The diffusion model predicts the complete combined latent produced by all input
encoders. No channels are filled by persistence after prediction.
"""

import logging
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from cryocast.losses import LeadTimeWeightedLoss, build_loss
from cryocast.models.diffusion import GaussianDiffusion, UNetDiffusion
from cryocast.types import BetaSchedule, ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor

log = logging.getLogger(__name__)


class DiffusionProcessor(BaseProcessor):
    """Latent-space diffusion processor with v-prediction and DDPM/DDIM sampling.

    Input space:
        TensorNTCHW with shape (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        - Concatenation of every encoder's latent output along the channel dimension

    Output space:
        TensorNTCHW with shape (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
        - Every combined-latent channel is predicted by reverse diffusion
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        timesteps: int = 1000,
        ddim_steps: int | None = None,
        eta: float = 1.0,
        beta_schedule: str = "cosine",
        kernel_size: int = 3,
        start_out_channels: int = 64,
        time_embed_dim: int = 256,
        dropout_rate: float = 0.1,
        normalization: str = "groupnorm",
        activation: str = "SiLU",
        use_autoregressive: bool = True,
        loss: DictConfig | nn.Module,
        **kwargs: Any,
    ) -> None:
        """Initialize the diffusion processor.

        Args:
            timesteps (int): Number of diffusion timesteps. Default is 1000.
            ddim_steps (int | None): Number of timesteps visited when sampling.
                Must satisfy ``1 <= ddim_steps <= timesteps``. ``None`` means
                ``timesteps`` (visit every trained timestep). Default is None.
            eta (float): Sampling stochasticity in ``[0, 1]``. ``eta=0`` gives
                deterministic DDIM sampling; ``eta=1`` matches DDPM's per-step
                noise. With ``ddim_steps == timesteps`` and ``eta=1`` the
                processor uses ancestral DDPM sampling. Default is 1.0.
            beta_schedule (str): Beta schedule used by ``GaussianDiffusion``:
                ``"cosine"`` or ``"linear"``. Default is ``"cosine"``.
            kernel_size (int): Convolution kernel size used in the conditional UNet.
            start_out_channels (int): Base number of channels in the first UNet block.
            time_embed_dim (int): Dimensionality of the diffusion timestep embedding.
            dropout_rate (float): Dropout probability applied inside the UNet blocks.
            normalization (str): Normalization layer type (e.g., "groupnorm").
            activation (str): Activation function used throughout the network (e.g., "SiLU").
            use_autoregressive (bool): Whether to use autoregressive sampling and
                one-step training. Default is True.
            loss (DictConfig | nn.Module): Loss module applied to (pred_v,
                target_v). Set this to ``${loss}`` in the yaml to reuse the
                top-level configured loss.
            **kwargs: Additional arguments passed to ``BaseProcessor``.

        """
        kwargs.pop("computes_loss_in_latent_space", None)
        super().__init__(computes_loss_in_latent_space=True, **kwargs)

        # Instantiate the configured loss function.
        self.loss_fn: nn.Module = (
            loss if isinstance(loss, nn.Module) else build_loss(loss)
        )

        # Diffusion predicts the complete combined latent rather than a target slice.
        c_combined = self.data_space.channels
        if self.data_space_target != self.data_space:
            msg = (
                "DiffusionProcessor requires data_space_target to equal the full "
                "combined latent data_space."
            )
            raise ValueError(msg)
        self.c_combined = c_combined

        self.timesteps = timesteps
        self.use_autoregressive = use_autoregressive

        # Autoregressive training only optimises one forecast step at a time
        if use_autoregressive and isinstance(self.loss_fn, LeadTimeWeightedLoss):
            log.warning(
                "lead_time_exponent=%s has no effect on the autoregressive DDPM "
                "training loss, which is computed on a single forecast step. "
                "Validation and test losses are still lead-time weighted.",
                self.loss_fn.exponent,
            )

        # Configure the reverse-diffusion sampler (validates ddim_steps and eta).
        self.set_sampler(ddim_steps=ddim_steps, eta=eta)

        # UNet conditioning channels: history folded NTCHW -> NCHW.
        cond_channels = c_combined * self.n_history_steps

        # Channels denoised by the UNet: one complete latent frame (AR), or all
        # forecast frames folded into channels (parallel).
        diffused_channels = (
            c_combined if use_autoregressive else c_combined * self.n_forecast_steps
        )
        self.diffused_channels = diffused_channels

        self.model = UNetDiffusion(
            input_channels=cond_channels,
            output_channels=diffused_channels,
            timesteps=timesteps,
            kernel_size=kernel_size,
            start_out_channels=start_out_channels,
            time_embed_dim=time_embed_dim,
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps,
            beta_schedule=BetaSchedule(beta_schedule),
        )

    def _build_metrics_prediction(self, pred_x0: TensorNCHW) -> TensorNTCHW:
        """Restore a training-time x_0 estimate to full combined NTCHW latent shape."""
        b, _, h, w = pred_x0.shape
        if self.use_autoregressive:
            return pred_x0.unsqueeze(1).expand(
                b, self.n_forecast_steps, self.c_combined, h, w
            )
        return pred_x0.reshape(b, self.n_forecast_steps, self.c_combined, h, w)

    def _rollout_inference(self, x: TensorNTCHW) -> ProcessorOutput:
        """Generate a latent forecast using a reverse diffusion process.

        This method selects between two diffusion sampling strategies:

        1. Non-autoregressive (parallel) sampling:
        - The model generates the entire future sequence in a single diffusion process.
        - No temporal dependency exists between forecast steps.

        2. Autoregressive sampling:
        - Forecast steps are generated sequentially.
        - Each step is produced via an independent diffusion process.
        - The conditioning window is updated after each step to incorporate
            previously generated outputs.

        Args:
            x (TensorNTCHW): Encoded history tensor of shape
                (B, n_history_steps, n_latent_channels_total, H, W).

        Returns:
            ProcessorOutput:
            - prediction: (B, n_forecast_steps, n_latent_channels_total, H, W)
            - loss: None

        Notes:
            - The diffusion process follows v-parameterization.
            - Sampling begins from standard Gaussian noise.

        """
        if self.use_autoregressive:
            return ProcessorOutput(prediction=self._sample_autoregressive(x))
        return ProcessorOutput(prediction=self._sample_parallel(x))

    def _rollout_training(self, x: TensorNTCHW, y: TensorNTCHW) -> ProcessorOutput:
        """One training rollout using DDPM v-prediction loss.

        During training, the clean combined forecast latent is corrupted using the forward
        diffusion process by adding noise at a randomly sampled timestep. The
        model is trained to predict the corresponding v-target for the complete
        combined latent.

        Args:
            x (TensorNTCHW): Encoded history tensor of shape
                (B, n_history_steps, n_latent_channels_total, H, W).
            y (TensorNTCHW): Encoded combined forecast latent of shape
                (B, n_forecast_steps, n_latent_channels_total, H, W).

        Returns:
            ProcessorOutput:
            - prediction: (B, n_forecast_steps, n_latent_channels_total, H, W)
                Full combined-latent prediction reconstructed from x_0. Used only
                for metrics/callbacks under ``torch.no_grad``.
            - loss: v-prediction loss value computed by ``self.loss_fn``.

        Notes:
            - In autoregressive mode, the model is trained on forecast step 0
              only, matching the convention used by ``cryocast.models.ddpm.DDPM``.
              Multi-step forecasts come from rolling out the AR loop at inference.
            - In parallel mode, the model is trained on all forecast steps
              jointly (folded into the channel dimension).

        """
        b = x.shape[0]
        device = x.device

        # History frames folded into channels for the 2D UNet.
        cond: TensorNCHW = x.flatten(
            start_dim=1, end_dim=2
        )  # (B, T_hist * C_combined, H, W)

        # Either take the first step (AR0) or fold all steps into channels (parallel)
        y_flat: TensorNCHW = (
            y[:, 0]  # (B, C_combined, H, W)
            if self.use_autoregressive
            else y.reshape(b, self.n_forecast_steps * self.c_combined, *y.shape[-2:])
        )

        # Random diffusion timestep per sample.
        t = torch.randint(0, self.timesteps, (b,), device=device).long()

        # Add noise to the clean target at diffusion step t.
        noise: TensorNCHW = torch.randn_like(y_flat)
        noisy_y: TensorNCHW = self.diffusion.q_sample(y_flat, t, noise)

        # Predict the velocity conditioned on the noisy target and history.
        pred_v: TensorNCHW = self.model(noisy_y, t, cond)

        # Compute the target velocity for the sampled noise and timestep.
        target_v: TensorNCHW = self.diffusion.calculate_v(y_flat, noise, t)

        # Compute the v-prediction training loss
        # We unfold to NTCHW in case the loss requires a time dimension
        # AR only predicts a single step so T=1; parallel predicts all steps
        n_steps = 1 if self.use_autoregressive else self.n_forecast_steps
        loss = self.loss_fn(
            pred_v.unflatten(1, (n_steps, self.c_combined)),
            target_v.unflatten(1, (n_steps, self.c_combined)),
        )

        # Reconstruct x0 for metrics only; this is not used for training.
        with torch.no_grad():
            # calculate_v() has the same formula as the x_0 reconstruction,
            # so we reuse it by passing pred_v as x_start.
            pred_x0: TensorNCHW = self.diffusion.calculate_v(
                x_start=pred_v, noise=noisy_y, t=t
            )
            prediction = self._build_metrics_prediction(pred_x0)

        return ProcessorOutput(prediction=prediction, loss=loss)

    def _run_ddim_reverse_diffusion(
        self, y: TensorNCHW, cond: TensorNCHW
    ) -> TensorNCHW:
        """Iteratively denoise ``y`` using the DDIM sampler.

        Args:
            y (TensorNCHW): Noisy latent to denoise, of shape (B, C, H, W).
            cond (TensorNCHW): History condition folded to channels, of shape
                (B, T_hist * C_combined, H, W).

        Returns:
            TensorNCHW: Denoised latent of shape (B, C, H, W).

        """
        b = y.shape[0]
        device = y.device
        alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        ts = self._ddim_timesteps.to(device)

        for i in range(self.ddim_steps):
            t = ts[i]
            pred_v = self.model(y, t.expand(b), cond)

            alpha_bar_t = alphas_cumprod[t]
            sqrt_ab = alpha_bar_t.sqrt()
            sqrt_1mab = (1.0 - alpha_bar_t).sqrt()

            # Recover x_0 and epsilon from v-prediction.
            pred_x0 = sqrt_ab * y - sqrt_1mab * pred_v
            pred_eps = sqrt_1mab * y + sqrt_ab * pred_v

            # Final step targets x_0 (alpha_bar_prev = 1).
            alpha_bar_prev = (
                alpha_bar_t.new_ones(())
                if i == self.ddim_steps - 1
                else alphas_cumprod[ts[i + 1]]
            )

            sigma = (
                self.eta
                * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).sqrt()
                * (1.0 - alpha_bar_t / alpha_bar_prev).sqrt()
            )
            dir_coeff = (1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0).sqrt()
            noise = torch.randn_like(y) if self.eta > 0.0 else 0.0

            y = alpha_bar_prev.sqrt() * pred_x0 + dir_coeff * pred_eps + sigma * noise

        return y

    def _run_ddpm_reverse_diffusion(
        self, y: TensorNCHW, cond: TensorNCHW
    ) -> TensorNCHW:
        """Iteratively denoise ``y`` from t=timesteps-1 down to t=0.

        Args:
            y (TensorNCHW): Noisy latent to denoise, of shape (B, C, H, W).
            cond (TensorNCHW): History condition folded to channels, of shape
                (B, T_hist * C_combined, H, W).

        Returns:
            TensorNCHW: Denoised latent of shape (B, C, H, W).

        """
        b = y.shape[0]
        device = y.device
        for t_step in reversed(range(self.timesteps)):
            t = torch.full((b,), t_step, dtype=torch.long, device=device)
            pred_v = self.model(y, t, cond)
            y = self.diffusion.p_sample(y, t, pred_v)
        return y

    def _run_reverse_diffusion(self, y: TensorNCHW, cond: TensorNCHW) -> TensorNCHW:
        """Denoise ``y`` with DDPM or DDIM sampling depending on ``ddim_steps`` and ``eta``.

        Uses ancestral DDPM sampling when ``ddim_steps == timesteps`` and
        ``eta == 1``, and DDIM sampling otherwise.

        Args:
            y (TensorNCHW): Noisy latent to denoise, of shape (B, C, H, W).
            cond (TensorNCHW): History condition folded to channels, of shape
                (B, T_hist * C_combined, H, W).

        Returns:
            TensorNCHW: Denoised latent of shape (B, C, H, W).

        """
        if self.uses_ddpm_sampler:
            return self._run_ddpm_reverse_diffusion(y, cond)
        return self._run_ddim_reverse_diffusion(y, cond)

    def _sample_autoregressive(self, x: TensorNTCHW) -> TensorNTCHW:
        """Autoregressive reverse diffusion sampling (one forecast step at a time).

        Each forecast step is generated sequentially via an independent reverse
        diffusion process. After each step, the conditioning window is updated:
        the complete predicted latent frame is appended to the history window.

        Args:
            x (TensorNTCHW): Encoded history tensor of shape
                (B, n_history_steps, n_latent_channels_total, H, W).

        Returns:
            TensorNTCHW: Denoised forecast latent of shape
                (B, n_forecast_steps, n_latent_channels_total, H, W),
                formed by stacking all full-latent per-step predictions.

        Notes:
            - Sampling at each step starts from standard Gaussian noise.
            - The model predicts v-parameterization at every diffusion timestep.
            - The history window slides forward by one frame per step, using
              the model's own prediction as the new observation of the target
              slice.

        """
        cond_window: TensorNTCHW = x.clone()  # (B, T_hist, C_combined, H, W)
        b, _, _, h, w = x.shape
        device = x.device
        all_predictions: list[TensorNCHW] = []

        for _ in range(self.n_forecast_steps):
            cond = cond_window.flatten(
                start_dim=1, end_dim=2
            )  # (B, T_hist * C_combined, H, W)

            # Start each forecast step from Gaussian noise.
            y = torch.randn((b, self.c_combined, h, w), device=device)

            # Iteratively denoise from the final diffusion timestep to zero.
            y = self._run_reverse_diffusion(y, cond)

            all_predictions.append(y)

            # Slide the history window forward using the complete predicted latent.
            cond_window = torch.cat([cond_window[:, 1:], y.unsqueeze(1)], dim=1)

        return torch.stack(all_predictions, dim=1)

    def _sample_parallel(self, x: TensorNTCHW) -> TensorNTCHW:
        """Non-autoregressive (parallel) reverse diffusion sampling.

        This method generates the entire forecast sequence in a single
        diffusion process applied to one joint output tensor. The forecast
        steps are encoded as channels in a single tensor of shape
        (B, n_forecast_steps * C_target, H, W), then reshaped back to NTCHW
        and lifted into the combined latent for the frozen decoder.

        Args:
            x (TensorNTCHW): Encoded history tensor of shape
                (B, n_history_steps, n_latent_channels_total, H, W).

        Returns:
            TensorNTCHW: Denoised forecast latent of shape
                (B, n_forecast_steps, n_latent_channels_total, H, W).

        Notes:
            - Sampling starts from Gaussian noise.
            - The model predicts v-parameterization at each diffusion step.
            - All forecast steps are denoised together as a single object.

        """
        cond = x.flatten(start_dim=1, end_dim=2)

        b, _, h, w = cond.shape
        device = cond.device

        y = torch.randn(
            (b, self.n_forecast_steps * self.c_combined, h, w), device=device
        )

        # Iteratively denoise from the final diffusion timestep to zero.
        y = self._run_reverse_diffusion(y, cond)

        return y.reshape(b, self.n_forecast_steps, self.c_combined, h, w)

    def rollout(
        self,
        x: TensorNTCHW,
        y: TensorNTCHW | None = None,
    ) -> ProcessorOutput:
        """Run the diffusion process in latent space.

        Dispatches between two execution paths:

        1. Training path (``y`` provided):
        - Adds noise to the encoded target latent at a random diffusion timestep.
        - Predicts v conditioned on the encoded history.
        - Returns a v-prediction loss inside ``ProcessorOutput``, computed
          by ``self.loss_fn`` (the configured loss).

        2. Inference path (``y`` is None):
        - Runs reverse diffusion from Gaussian noise conditioned on the encoded history.
        - Autoregressive or parallel depending on ``self.use_autoregressive``.

        Args:
            x (TensorNTCHW): Encoded history tensor of shape
                (B, n_history_steps, n_latent_channels_total, H, W).
            y (TensorNTCHW | None): During training: encoded target latent of
                shape (B, n_forecast_steps, n_latent_channels_target, H, W).
                Otherwise: None.

        Returns:
            ProcessorOutput:
            - prediction: (B, n_forecast_steps, n_latent_channels_total, H, W)
            - loss: v-prediction loss from ``self.loss_fn`` (training only,
                otherwise None)

        """
        if y is not None:
            return self._rollout_training(x, y)
        return self._rollout_inference(x)

    def set_sampler(self, *, ddim_steps: int | None = None, eta: float = 1.0) -> None:
        """Choose how the reverse diffusion process is run at inference time.

        This does not change the network weights, so it can be called on a
        processor loaded from any checkpoint.

        Args:
            ddim_steps (int | None): Number of timesteps visited when sampling.
                Must satisfy ``1 <= ddim_steps <= timesteps``. ``None`` means
                ``timesteps``.
            eta (float): Sampling stochasticity in ``[0, 1]``.

        Raises:
            ValueError: If ``ddim_steps`` or ``eta`` is out of range.

        """
        if ddim_steps is None:
            ddim_steps = self.timesteps
        if not 1 <= ddim_steps <= self.timesteps:
            msg = (
                f"ddim_steps={ddim_steps} must be in the range "
                f"[1, timesteps={self.timesteps}]."
            )
            raise ValueError(msg)
        if not 0.0 <= eta <= 1.0:
            msg = f"eta={eta} must be in the range [0.0, 1.0]."
            raise ValueError(msg)

        self.ddim_steps = ddim_steps
        self.eta = eta

        # Evenly-spaced subset of trained timesteps, highest-first.
        # e.g. timesteps=1000, ddim_steps=50 -> [999, ..., 0].
        # Plain attribute (not a buffer) so checkpoints don't depend on the sampler.
        self._ddim_timesteps = torch.linspace(
            self.timesteps - 1, 0, ddim_steps, dtype=torch.long
        )

    @property
    def uses_ddpm_sampler(self) -> bool:
        """Whether inference uses ancestral DDPM sampling over every timestep."""
        return self.ddim_steps == self.timesteps and self.eta == 1.0
