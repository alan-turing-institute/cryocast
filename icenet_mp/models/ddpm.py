import logging
from typing import Any, ClassVar, NoReturn

import torch
import torch.nn.functional as F

from icenet_mp.models.common import Mask, RestrictRange
from icenet_mp.models.diffusion import GaussianDiffusion, UNetDiffusion
from icenet_mp.types import ModelStepOutput, RangeRestriction, TensorNCHW, TensorNTCHW

from .base_model import BaseModel

log = logging.getLogger(__name__)


class SimpleEncoder2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Simple 2D encoder block using Conv2d, GroupNorm, and SiLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(4, out_channels),
            torch.nn.SiLU(),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward pass through the encoder block.

        Args:
            x (TensorNCHW): Input tensor of shape (B, C, H, W).

        Returns:
            TensorNCHW: Output tensor after applying the block.

        """
        return self.net(x)


class DDPM(BaseModel):
    """Denoising Diffusion Probabilistic Model (DDPM).

    Input space:
        TensorNTCHW with shape (batch_size, n_history_steps + n_history_steps * n_era5_channels, height, width)
        - OSISAF input: T historical steps, singleton channel squeezed
        - ERA5 input: T historical steps times number of channels, resized to OSISAF resolution

    Output space:
        TensorNTCHW with shape (batch_size, n_forecast_steps * n_output_channels, height, width)
        - Forecasted outputs per timestep and channel, flattened along the channel dimension
    """

    # Parameters that should be excluded from hyperparameter logging
    ignored_hparams: ClassVar[frozenset[str]] = BaseModel.ignored_hparams | {"mask_dir"}

    def __init__(  # noqa: PLR0913
        self,
        *,
        activation: str = "SiLU",
        dropout_rate: float = 0.1,
        kernel_size: int = 3,
        learning_rate: float = 5e-4,
        mask_dir: str | None = None,
        mask_type: str | None = None,
        normalization: str = "groupnorm",
        restrict_range: str = "clamp",
        start_out_channels: int = 32,
        time_embed_dim: int = 256,
        timesteps: int = 1000,
        use_autoregressive: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the DDPM processor.

        Args:
            activation (str): Activation function used throughout the network (e.g., "SiLU").
            dropout_rate (float): Dropout probability applied inside the UNet blocks.
            kernel_size (int): Convolution kernel size used in the UNet.
            learning_rate (float): Optimizer learning rate for training. Default is 5e-4.
            mask_dir (str | None): Directory holding `active_mask.npy`/`land_mask.npy`.
                Required when `mask_type` is "active" or "land".
            mask_type (str | None): Output mask to apply during sampling: "active"
                (active+land), "land" (land only), or ``None`` to disable.
            normalization (str): Normalization layer type (e.g., "groupnorm").
            restrict_range (str): How to bound sampled output into [0, 1] before
                masking: none/sigmoid/clamp/tanh. Default is "clamp".
            start_out_channels (int): Base number of channels in the first UNet block.
            time_embed_dim (int): Dimensionality of the timestep embedding.
            timesteps (int): Number of diffusion timesteps. Default is 1000.
            use_autoregressive (bool): Whether to use autoregressive prediction. Default is True.
            **kwargs: Additional arguments passed to ``BaseModel``.

        """
        super().__init__(mask_dir=mask_dir, **kwargs)

        self.use_autoregressive = use_autoregressive
        self.osisaf_key = self.output_space.name

        # Bound sampled output into [0, 1] before masking.
        self.restrict = RestrictRange(
            RangeRestriction(restrict_range), min_val=0, max_val=1
        )

        # Load the requested mask (ACTIVE/LAND/NONE)
        self.mask = Mask(
            mask_type=mask_type,
            output_shape=self.output_space.shape,
            mask_dir=mask_dir,
        )

        era5_space = next(
            space
            for space in self.input_spaces
            if (space["name"] if isinstance(space, dict) else space.name) == "era5"
        )
        osisaf_space = next(
            space
            for space in self.input_spaces
            if (space["name"] if isinstance(space, dict) else space.name)
            == self.osisaf_key
        )

        # Get channels from either dict or object
        if isinstance(era5_space, dict):
            self.era5_space = era5_space["channels"]
        else:
            self.era5_space = era5_space.channels
        if isinstance(osisaf_space, dict):
            self.osisaf_channels = osisaf_space["channels"]
        else:
            self.osisaf_channels = osisaf_space.channels

        # Get the base output channels from output_space
        if isinstance(self.output_space, dict):
            self.base_output_channels = self.output_space["channels"]
        else:
            self.base_output_channels = self.output_space.channels

        # For autoregressive, we predict one step at a time
        if self.use_autoregressive:
            self.output_channels = self.base_output_channels
        else:
            self.output_channels = self.n_forecast_steps * self.base_output_channels

        self.timesteps = timesteps
        self.cond_channels = 64
        self.input_channels = self.cond_channels

        # "InstanceNorm" calculates the mean/std per batch, removing the need for offline preprocessing
        self.era5_norm = torch.nn.InstanceNorm3d(self.era5_space, affine=True)

        # Reduces the many ERA5 channels down to 32 important ones using 1x1 Conv
        self.era5_compressed_channels = 32
        self.era5_projector = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.era5_space, self.era5_compressed_channels, kernel_size=1
            ),
            torch.nn.SiLU(),
        )

        self.osisaf_encoder = SimpleEncoder2D(
            in_channels=self.n_history_steps * self.osisaf_channels,
            out_channels=self.cond_channels // 2,
        )

        # (Compressed_Channels * Time_Steps), preserving time history
        self.era5_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.era5_compressed_channels * self.n_history_steps,
                out_channels=self.cond_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.GroupNorm(4, self.cond_channels // 2),
            torch.nn.SiLU(),
        )

        self.model = UNetDiffusion(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            timesteps=self.timesteps,
            kernel_size=kernel_size,
            start_out_channels=start_out_channels,
            time_embed_dim=time_embed_dim,
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.learning_rate = learning_rate

        # Only emit the ERA5-forecast-missing warning once per model instance
        self._warned_missing_era5_forecast = False

    def forward(self, *args: Any, **kwargs: Any) -> NoReturn:
        msg = "This model uses `training_step`, `validation_step`, and `test_step` instead of `forward()`"
        raise NotImplementedError(msg)

    def sample(
        self,
        batch: dict[str, TensorNTCHW],
    ) -> TensorNCHW:
        """Generate forecasts using a reverse diffusion process.

        This method selects between two diffusion sampling strategies:

        1. Non-autoregressive (parallel) sampling:
        - The model generates the entire future sequence in a single diffusion process.
        - No temporal dependency exists between forecast steps.

        2. Autoregressive sampling:
        - Forecast steps are generated sequentially.
        - Each step is produced via an independent diffusion process.
        - The conditioning tensor is updated after each step to incorporate
            previously generated outputs.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing the input data.

        Returns:
            torch.Tensor:
                Forecast tensor of shape:
                [B, n_forecast_steps * base_output_channels, H, W]

                The output format is identical in both modes.

                - Parallel mode:
                    Produced in a single reverse diffusion process.

                - Autoregressive mode:
                    Constructed by concatenating step-wise diffusion outputs.

        Notes:
            - The diffusion process follows v-parameterization.
            - Sampling begins from standard Gaussian noise.

        """
        if self.use_autoregressive:
            return self._sample_autoregressive(batch)
        x = self.prepare_inputs(batch)
        return self._sample_parallel(x)

    def _sample_parallel(self, x: TensorNCHW) -> TensorNCHW:
        """Non-autoregressive (parallel) reverse diffusion sampling.

        This method generates the entire forecast sequence in a single
        diffusion process applied to one joint output tensor.

        The forecast steps are encoded as channels in a single tensor:

            [B, n_forecast_steps * base_output_channels, H, W]

        There is no sequential dependency between forecast steps because:
            - all steps are represented simultaneously in the same tensor
            - the diffusion process operates on the full tensor jointly

        Args:
            x (TensorNCHW):
                Conditioning tensor of shape [B, C, H, W].

        Returns:
            TensorNCHW:
                Denoised forecast tensor of shape:
                [B, n_forecast_steps * base_output_channels, H, W]

                Produced by a single reverse diffusion process.

        Notes:
            - Sampling starts from Gaussian noise.
            - The model predicts v-parameterization at each diffusion step.
            - All forecast steps are denoised together as a single object.

        """
        shape = (
            x.shape[0],
            self.n_forecast_steps * self.base_output_channels,
            *x.shape[-2:],
        )

        # Start from pure noise
        y = torch.randn(shape, device=self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full_like(
                x[:, 0, 0, 0], t, dtype=torch.long, device=self.device
            )
            pred_v: TensorNCHW = self.model(y, t_batch, x)
            y = self.diffusion.p_sample(y, t_batch, pred_v)

        # Bound into [0, 1] then apply masking
        return self.mask(self.restrict(y))

    def _sample_autoregressive(
        self,
        batch: dict[str, TensorNTCHW],
    ) -> torch.Tensor:
        """Autoregressive reverse diffusion sampling (one forecast step at a time).

        Each forecast step is generated sequentially via an independent reverse
        diffusion process. After each step, the conditioning window is updated:
        the predicted SIC frame is appended to the OSISAF history, and either the
        corresponding ERA5 forecast frame (if provided) or a repetition of the last
        observed ERA5 frame is appended to the ERA5 history.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing:
                    - self.osisaf_key: OSISAF SIC history tensor of shape
                    [B, T, 1, H, W].
                    - era5: ERA5 input tensor of shape [B, T, C, H2, W2].
                    - era5_forecast (optional): ERA5 forecast tensor of shape
                    [B, n_forecast_steps, C, H2, W2]. When absent, the last
                    observed ERA5 frame is repeated for each step.

        Returns:
            torch.Tensor:
                Forecast tensor of shape
                [B, n_forecast_steps * base_output_channels, H, W],
                formed by concatenating all per-step predictions along the
                channel dimension.

        Notes:
            - Sampling at each step starts from standard Gaussian noise.
            - The model predicts v-parameterization at every diffusion timestep.
            - The OSISAF conditioning window slides forward by one frame per step,
            using the model's own prediction as the new observation.

        """
        all_predictions = []
        osisaf = batch[self.osisaf_key].clone()  # [B, T, 1, H, W]
        era5 = batch["era5"].clone()  # [B, T, C, H2, W2]
        era5_forecast = batch.get("era5_forecast")

        for step in range(self.n_forecast_steps):
            current_batch = {
                self.osisaf_key: osisaf[:, -self.n_history_steps :],
                "era5": era5[:, -self.n_history_steps :],
            }
            x = self.prepare_inputs(current_batch)  # [B, cond_channels, H, W]

            B, _, H, W = x.shape  # noqa: N806
            y = torch.randn((B, self.base_output_channels, H, W), device=self.device)

            # Diffusion reverse process
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((B,), t, dtype=torch.long, device=self.device)
                pred_v: torch.Tensor = self.model(y, t_batch, x)
                y = self.diffusion.p_sample(y, t_batch, pred_v)

            # Clamp and store prediction
            y_step = self.mask(self.restrict(y))
            all_predictions.append(y_step)

            # Slide OSISAF window: append prediction as new frame.
            # SSMIS input has multiple channels per step but the model only
            # predicts base_output_channels (SIC). Carry the auxiliary channels
            # forward from the last observed frame and overwrite the SIC slot.
            new_frame = osisaf[:, -1:].clone()  # [B, 1, C_in, H, W]
            new_frame[:, :, : self.base_output_channels] = y_step.unsqueeze(1)
            osisaf = torch.cat([osisaf, new_frame], dim=1)

            # Slide ERA5 window: use forecast if available, else repeat last frame
            if era5_forecast is not None:
                next_era5 = era5_forecast[:, step : step + 1]
            else:
                if not self._warned_missing_era5_forecast:
                    log.warning(
                        "era5_forecast not provided in batch, repeating the last "
                        "observed ERA5 frame for all remaining autoregressive "
                        "forecast steps. This may reduce forecast quality for "
                        "longer horizons.",
                    )
                    self._warned_missing_era5_forecast = True
                next_era5 = era5[:, -1:]
            era5 = torch.cat([era5, next_era5], dim=1)

        # Concatenate all predictions along channel dimension
        return torch.cat(all_predictions, dim=1)

    def prepare_inputs(self, batch: dict[str, TensorNTCHW]) -> TensorNCHW:
        """Encode OSISAF and ERA5 separately, then concatenate.

        ERA5 -> Norm -> Project -> Resize -> Flatten Time -> Encode

        Args:
            batch: Dictionary with
                osisaf key (e.g. 'osisaf-south') [B, T, C, H, W]
                'era5' [B, T, C, H2, W2]

        Returns:
            Conditioning tensor [B, cond_channels, H, W]

        """
        osisaf = batch[self.osisaf_key]  # [B, T, C, H, W]
        era5 = batch["era5"]  # [B, T, C, H2, W2]

        # Handle OSISAF: flatten time and channels together
        B, T, C, H, W = osisaf.shape  # noqa: N806
        osisaf = osisaf.reshape(B, T * C, H, W)

        # Handle ERA5
        # Permute to [B, C, T, H2, W2] for 3D operations
        era5 = era5.permute(0, 2, 1, 3, 4)

        # Normalize (On-the-fly standardization)
        era5 = self.era5_norm(era5)

        # Project (Learnable Feature Selection)
        # [B, C, T, H2, W2] -> [B, 32, T, H2, W2]
        era5 = self.era5_projector(era5)

        # Resize Spatially
        B, C_new, T, H2, W2 = era5.shape  # noqa: N806
        # Flatten batch/channel/time for interpolation
        era5_flat = era5.reshape(B * C_new * T, 1, H2, W2)

        era5_resized = F.interpolate(
            era5_flat,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten Time into Channels
        # Reshape back to [B, C_new, T, H, W] then flatten T into C
        era5_features = era5_resized.reshape(B, C_new * T, H, W)

        # Encode Both
        osisaf_features = self.osisaf_encoder(osisaf)  # [B, cond//2, H, W]

        # era5_features enters the encoder as a 2D tensor with many channels
        era5_features = self.era5_encoder(era5_features)  # [B, cond//2, H, W]

        return torch.cat([osisaf_features, era5_features], dim=1)  # [B, cond, H, W]

    def training_step(
        self, batch: dict[str, TensorNTCHW], _batch_idx: int
    ) -> ModelStepOutput:
        """One training step using DDPM v-prediction loss.

        During training, the clean target (SIC) is corrupted using the forward
        diffusion process by adding noise at a randomly sampled timestep.
        The model is trained to predict the corresponding v-target.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing:
                    - input tensors (used to prepare conditioning inputs)
                    - "target": groundtruth SIC tensor

        Returns:
            ModelStepOutput:
            - prediction: reconstructed noisy SIC (pred_v)
            - target: generated noisy SIC (target_v)
            - loss: training loss value

        """
        # Prepare input tensor by combining osisaf-south and era5
        x = self.prepare_inputs(batch)  # [B, C_cond, H, W]

        # Extract target
        if self.use_autoregressive:
            y = batch["target"][
                :, 0, :, :, :
            ]  # [B, C, H, W] — one step at a time (AR trains on t=0 only)
        else:
            y = batch["target"].flatten(1, 2)  # [B, T*C, H, W] — all steps at once

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()

        # Create noisy version
        noise = torch.randn_like(y)  # [B, C, H, W] (AR) or [B, T*C, H, W] (parallel)
        noisy_y = self.diffusion.q_sample(
            y, t, noise
        )  # [B, C, H, W] (AR) or [B, T*C, H, W] (parallel)

        # Predict v
        pred_v: torch.Tensor = self.model(
            noisy_y, t, x
        )  # [B, C, H, W] (AR) or [B, T*C, H, W] (parallel)

        # Compute target v
        target_v = self.diffusion.calculate_v(
            y, noise, t
        )  # [B, C, H, W] (AR) or [B, T*C, H, W] (parallel)

        # Compute loss
        loss = self.loss(pred_v, target_v)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Convert to NTCHW format to update metrics
        if self.use_autoregressive:
            prediction = pred_v.unsqueeze(1)  # [B, 1, C, H, W]
            target = target_v.unsqueeze(1)  # [B, 1, C, H, W]
        else:
            T, C = self.n_forecast_steps, self.base_output_channels  # noqa: N806
            prediction = pred_v.unflatten(1, (T, C))  # [B, T, C, H, W]
            target = target_v.unflatten(1, (T, C))  # [B, T, C, H, W]

        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)

    def validation_step(
        self, batch: dict[str, TensorNTCHW], _batch_idx: int
    ) -> ModelStepOutput:
        """One validation step using full diffusion sampling.

        During validation, samples are generated by starting from noise and
        iteratively denoising conditioned on the inputs. The final prediction
        is compared to the groundtruth SIC using the configured evaluation loss.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing:
                    - input tensors (used to prepare conditioning inputs)
                    - "target": groundtruth SIC tensor

        Returns:
            ModelStepOutput:
            - prediction: reconstructed SIC (y_hat)
            - target: groundtruth SIC (y)
            - loss: validation loss value

        """
        # Extract target and optional weights
        y = batch["target"].flatten(1, 2)  # [B, T*C, H, W]

        # Generate samples
        y_hat = self.sample(batch)  # [B, T*C, H, W]

        # Calculate loss
        loss = self.loss(y_hat, y)
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Convert to NTCHW format to update metrics and return
        T, C = self.n_forecast_steps, self.base_output_channels  # noqa: N806
        prediction = y_hat.unflatten(1, (T, C))  # [B, T, C, H, W]
        target = y.unflatten(1, (T, C))  # [B, T, C, H, W]

        self.validation_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)

    def test_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,  # noqa: PT019
    ) -> ModelStepOutput:
        """One test step using full diffusion sampling and metric evaluation.

        During testing, predictions are generated by starting from noise
        and running the reverse diffusion process conditioned on the inputs.
        The final reconstructed SIC is compared to the groundtruth target
        using the configured loss and test metrics.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing:
                    - input tensors (used to prepare conditioning inputs)
                    - "target": groundtruth SIC tensor

        Returns:
            ModelStepOutput:
            - prediction: reconstructed SIC (y_hat)
            - target: groundtruth SIC (y)
            - loss: test loss value

        """
        y = batch["target"].flatten(1, 2)  # [B, T*C, H, W]
        y_hat = self.sample(batch)  # [B, T*C, H, W]

        loss = self.loss(y_hat, y)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Convert to NTCHW format to update metrics and return
        T, C = self.n_forecast_steps, self.base_output_channels  # noqa: N806
        prediction = y_hat.unflatten(1, (T, C))  # [B, T, C, H, W]
        target = y.unflatten(1, (T, C))  # [B, T, C, H, W]

        self.test_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
