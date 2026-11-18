import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import hydra
import torch
from omegaconf import DictConfig
from typing_extensions import override

from cryocast.models.decoders import BaseDecoder
from cryocast.models.encoders import BaseEncoder
from cryocast.models.processors import BaseProcessor
from cryocast.types import (
    DataSpace,
    ModelStepOutput,
    ProcessorOutput,
    ProcessorOutputDiffusion,
    RolloutSpace,
    TensorNCHW,
    TensorNTCHW,
)

from .base_model import BaseModel

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys

log = logging.getLogger(__name__)


class EncodeProcessDecode(BaseModel):
    """Model that encodes to latent space, processes, then decodes back."""

    # Parameters that should be excluded from hyperparameter logging (e.g. local paths)
    ignored_hparams: ClassVar[frozenset[str]] = BaseModel.ignored_hparams | {
        "decoder",
        "encoders",
        "mask_dir",
        "processor",
    }

    def __init__(  # noqa: PLR0913
        self,
        *,
        encoders: DictConfig | list[BaseEncoder],
        processor: DictConfig | BaseProcessor,
        decoder: DictConfig | BaseDecoder,
        target_variable_indices: list[int],
        auxiliary_loss_weights: dict[str, float] | None = None,
        mask_dir: str | None = None,
        rollout_space: RolloutSpace | str = RolloutSpace.LATENT,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model.

        Note that we use BaseProcessor.computes_loss_in_latent_space to control:
        - `needs_future_inputs`: training batches hold `future_<input>` keys.
        - the decoder is frozen if the autoencoder and pred_x0 losses are off
        - `rollout_space`: cannot be set to "physical"
        - `training_step`: takes the latent-loss branch.

        Args:
            encoders: DictConfig or list of BaseEncoder, one per input dataset.
            processor: DictConfig or BaseProcessor, the latent-space processor.
            decoder: DictConfig or BaseDecoder, the decoder from latent to output space.
            target_variable_indices: indices of the target variables within the output space.
            auxiliary_loss_weights: weights for auxiliary loss terms. These apply only
                to processors that compute their loss in latent space. Weights must not
                be negative. Supported keys:
                - `autoencoder` (default 1.0): loss on the decoded encoded future inputs.
                - `pred_x0` (default 0.0): loss on the decoded x_0 estimate
                  from a diffusion processor.
                If both weights are 0, the decoder is frozen.
            mask_dir: directory containing masks for the decoder (if needed).
            rollout_space: RolloutSpace.LATENT or RolloutSpace.PHYSICAL (or the
                equivalent string), where to perform the forecast loop.
            **kwargs: forwarded to ``BaseModel`` (spaces, masks, range, skip).

        """
        super().__init__(
            mask_dir=mask_dir,
            target_variable_indices=target_variable_indices,
            **kwargs,
        )

        # Add one encoder per dataset
        try:
            self.encoders: list[BaseEncoder] = (
                [
                    hydra.utils.instantiate(
                        encoders[input_space.name],
                        data_space_in=input_space,
                        latent_space=encoders["latent_space"],
                        latitudes_fn=self.latitudes_fn,
                        longitudes_fn=self.longitudes_fn,
                    )
                    for input_space in self.input_spaces
                ]
                if isinstance(encoders, Mapping)
                else [
                    encoder
                    for input_space in self.input_spaces
                    for encoder in encoders
                    if encoder.name == input_space.name
                ]
            )
        except KeyError as exc:
            msg = (
                f"Error instantiating encoders: {exc}. Please ensure that encoders are "
                f"specified for all input spaces: {self.input_spaces}"
            )
            raise ValueError(msg) from exc

        # Because the encoders are stored as `list[Module]`` to ensure consistent
        # ordering, `self.encoders` will not be automatically registered as Lightning
        # submodules. We therefore do so explicitly here.
        for input_space, module in zip(self.input_spaces, self.encoders, strict=True):
            module_name = f"encoder_{input_space.name}".lower().replace("-", "_")
            self.add_module(module_name, module)

        # Confirm that all encoders have the same output shape
        latent_shapes = {encoder.data_space_out.shape for encoder in self.encoders}
        if len(latent_shapes) != 1:
            msg = (
                f"Expected all encoders to have the same output shape, but found "
                f"{len(latent_shapes)} different shapes: {latent_shapes}"
            )
            raise ValueError(msg)

        # Verify the output channels for each encoder
        for encoder in self.encoders:
            encoder.verify_output_channels(self.device)

        # Add a processor
        combined_latent_space = DataSpace(
            name="combined_latent_space",
            channels=sum(encoder.data_space_out.channels for encoder in self.encoders),
            shape=latent_shapes.pop(),
        )
        self.processor: BaseProcessor = (
            processor
            if isinstance(processor, BaseProcessor)
            else hydra.utils.instantiate(
                processor,
                # Nested configs (e.g. a `loss`) are built by the processor itself
                _recursive_=False,
                data_space=combined_latent_space,
                n_forecast_steps=self.n_forecast_steps,
                n_history_steps=self.n_history_steps,
            )
        )

        # Add a decoder
        self.decoder: BaseDecoder = (
            decoder
            if isinstance(decoder, BaseDecoder)
            else hydra.utils.instantiate(
                decoder,
                data_space_in=combined_latent_space,
                data_space_out=self.output_space,
                mask_dir=mask_dir,
            )
        )

        # Validate the auxiliary loss weights, and freeze the decoder if needed
        self._configure_auxiliary_losses(auxiliary_loss_weights or {})

        # Validate rollout options
        self.rollout_space = RolloutSpace(rollout_space)
        if self.rollout_space == RolloutSpace.PHYSICAL:
            # Physical rollout needs an observed physical state to advance
            input_names = {input_space.name for input_space in self.input_spaces}
            if self.output_space.name not in input_names:
                msg = (
                    f"rollout_space='physical' requires the prediction target group "
                    f"'{self.output_space.name}' to also be a model input, so that "
                    f"the rollout has an observed physical state to advance. Inputs: "
                    f"{sorted(input_names)}."
                )
                raise ValueError(msg)
            # Physical rollout passes the output through the decoder so is incompatible
            # with processors that compute their loss in latent space.
            if self.processor.computes_loss_in_latent_space:
                msg = (
                    f"rollout_space='physical' is incompatible with processor "
                    f"{type(self.processor).__name__}, which computes its loss in "
                    f"latent space."
                )
                raise ValueError(msg)

    def _configure_auxiliary_losses(self, weights: Mapping[str, float]) -> None:
        """Validate the auxiliary loss weights and freeze the decoder if it is unused.

        Args:
            weights: weights for auxiliary loss terms, see `__init__`.

        Raises:
            ValueError: If a weight is negative.

        """
        weights = {"autoencoder": 1.0, "pred_x0": 0.0, **weights}
        for key, weight in weights.items():
            if weight < 0:
                msg = (
                    f"auxiliary_loss_weights.{key} must not be negative, but it is "
                    f"{weight}."
                )
                raise ValueError(msg)
        self.weight_autoencoder = float(weights["autoencoder"])
        self.weight_pred_x0 = float(weights["pred_x0"])

        if self.processor.computes_loss_in_latent_space:
            # Only the auxiliary losses use the decoder. If both are off, freeze it.
            if self.weight_autoencoder == 0 and self.weight_pred_x0 == 0:
                self.decoder.freeze()

            # Without the autoencoder loss, nothing keeps the latent space and the
            # decoder compatible while they train.
            if self.weight_autoencoder == 0 and any(
                parameter.requires_grad
                for module in (*self.encoders, self.decoder)
                for parameter in module.parameters()
            ):
                log.warning(
                    "auxiliary_loss_weights.autoencoder is 0, but the encoders or the "
                    "decoder are trainable. Without the autoencoder loss, the latent "
                    "space can drift away from what the decoder expects."
                )

    @override
    def load_state_dict(
        self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> "_IncompatibleKeys":
        """Load a state dict, ignoring weights for the removed target encoder.

        Checkpoints saved before the target encoder was removed contain
        ``target_encoder.*`` weights. Nothing uses them now, so they are dropped and
        these checkpoints still load strictly.

        Note: this override will be removed in a future update.
        """
        if stale := [key for key in state_dict if key.startswith("target_encoder.")]:
            log.warning(
                "Ignoring %d unused target_encoder checkpoint weights.", len(stale)
            )
            state_dict = {k: v for k, v in state_dict.items() if k not in stale}
        return super().load_state_dict(state_dict, *args, **kwargs)

    @property
    @override
    def needs_future_inputs(self) -> bool:
        # Latent-space losses compare forecasts against encoded inputs, so they need
        # inputs over future steps.
        return self.processor.computes_loss_in_latent_space

    def _pred_x0_loss(
        self,
        processor_output: ProcessorOutput,
        target: TensorNTCHW,
        anchor: TensorNTCHW | None,
    ) -> torch.Tensor:
        """Decode the processor's x_0 estimate and compare it to the target.

        Only the samples in `pred_x0_mask` count. In autoregressive mode, `pred_x0`
        holds only the first forecast step, so we compare it to `target[:, :1]`. In
        parallel mode, we compare all the forecast steps.

        Args:
            processor_output: Output of a diffusion processor training step.
            target: TensorNTCHW with shape (batch, n_forecast_steps, C_out, H, W).
            anchor: Skip-connection anchor with shape (batch, 1, C_out, H, W), or None.

        Returns:
            Scalar loss. It is 0 if no sample is in the mask.

        Raises:
            TypeError: If the processor does not return a ProcessorOutputDiffusion.

        """
        if not isinstance(processor_output, ProcessorOutputDiffusion):
            msg = (
                f"auxiliary_loss_weights.pred_x0 is greater than 0, but "
                f"processor {type(self.processor).__name__} does not return an x_0 "
                f"estimate. Use a diffusion processor or set the weight to 0."
            )
            raise TypeError(msg)

        # Decode every sample so that the decoder is always in the graph. With DDP, a
        # parameter that has no gradient on one process stops training.
        pred_x0, mask = processor_output.pred_x0, processor_output.pred_x0_mask
        decoded = self.decoder.rollout(pred_x0, anchor)
        target = target[:, : pred_x0.shape[1]]

        # The mean over an empty mask is NaN. Return 0, and keep the graph.
        if not mask.any():
            return decoded.sum() * 0.0
        return self.loss(decoded[mask], target[mask])

    def _encode_inputs(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Encode all inputs into latent space, concatenating on the channel dimension.

        Args:
            inputs: Dictionary with one TensorNTCHW entry per input dataset with shape
                (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)

        Returns:
            TensorNTCHW with shape (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

        """
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(inputs[encoder.name]) for encoder in self.encoders
        ]
        return torch.cat(latent_inputs, dim=2)

    def _encode_future_inputs(
        self, batch: dict[str, TensorNTCHW], *, with_grad: bool = False
    ) -> TensorNTCHW:
        """Encode every input over the forecast steps into the combined latent space.

        This is the ground truth for processors that compute their loss in latent space.

        Args:
            batch: Dictionary with a `future_<input>` TensorNTCHW entry per input
                dataset with shape (batch, n_forecast_steps, n_input_channels_k, H_input_k, W_input_k)
            with_grad: Whether to record gradients. Set this only when the result
                also goes into the autoencoder loss. The caller must then detach the
                copy that it gives to the processor as the latent target.

        Returns:
            TensorNTCHW with shape (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

        """
        try:
            future_inputs = {
                input_space.name: batch[f"future_{input_space.name}"]
                for input_space in self.input_spaces
            }
        except KeyError as exc:
            msg = (
                f"Processor {type(self.processor).__name__} computes its loss in "
                f"latent space, so the batch needs future inputs, but {exc} is "
                f"missing."
            )
            raise ValueError(msg) from exc

        # Do not record gradients unless the autoencoder loss needs them. This saves
        # the memory for the encoder activations over the forecast steps.
        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()):
            return self._encode_inputs(future_inputs)

    def _extract_anchor(self, window: TensorNTCHW) -> TensorNCHW | None:
        """Extract the last frame's target variables as the decoder skip-connection anchor.

        Returns None if the decoder has no skip connection configured, in which case
        the anchor is unused.

        Args:
            window: TensorNTCHW holding the target dataset, most recent frame last, e.g.
                the model's own input window or an evolving physical rollout state.

        """
        if not self.decoder.skip_connection:
            return None
        return window[:, -1, self.target_variable_indices, :, :]

    def _forward_rollout_latent(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Rollout to the desired number of forecast steps in latent space.

        - start with multiple `NTCHW` inputs each with shape (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)
        - encode inputs to `NTCHW` latent space (batch, n_history_steps, n_latent_channels, H_latent, W_latent)
        - concatenate inputs in `NTCHW` latent space (batch, n_history_steps, n_latent_channels_total, H_latent, W_latent)
        - process in latent space `NTCHW` (batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent)
        - decode back to `NTCHW` output space (batch, n_forecast_steps, n_output_channels, H_output, W_output)
        - add a skip connection from the most recent target value to every forecast step

        Args:
            inputs: Dictionary with one TensorNTCHW entry per input dataset with shape
                (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)

        Returns:
            TensorNTCHW with shape (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)

        """
        # Encode inputs into latent space: tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        latent_input_combined: TensorNTCHW = self._encode_inputs(inputs)

        # Process in latent space: tensor with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
        latent_output: TensorNTCHW = self.processor.rollout(
            latent_input_combined
        ).prediction

        # Use persistence as the skip-connection anchor, if required
        persistence = self._extract_anchor(inputs[self.output_space.name])

        # Decode to output space: tensor with (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)
        return self.decoder.rollout(
            latent_output, None if persistence is None else persistence.unsqueeze(1)
        )

    def _forward_rollout_physical(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Rollout to the desired number of forecast steps in physical space.

        Per step: encode the window of observed/predicted frames, take ONE processor
        step, decode to a physical field, then roll that field into the window and
        re-encode on the next iteration. Contrast with the default path, which appends
        the processor's raw latent to its own input window and never re-encodes.

        When the decoder has an additive skip connection, it emits a tendency and the
        state advances as `x_{k+1} = clamp(x_k + delta_k, 0, 1)`, so a zero-output
        network reproduces persistence exactly.

        Non-target input groups hold their most recent OBSERVED frame for every forecast
        step. No future information enters: only the input groups, which hold the
        n_history_steps frames ending at the forecast origin, are ever read. Information
        from the n_forecast_steps future frames are held in the "target" and
        "future_<input>" keys, which are not used here.

        Args:
            inputs: Dictionary with one TensorNTCHW entry per input dataset with shape
                (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)

        Returns:
            TensorNTCHW with shape (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)

        """
        target_name = self.output_space.name

        # Non-target groups: hold the newest observed frame for the whole rollout.
        # Only input groups are read, never other batch entries (e.g. future_<input>).
        windows: dict[str, TensorNTCHW] = {  # (B, n_history, C_k, H_k, W_k)
            input_space.name: inputs[input_space.name][:, -1:].expand_as(
                inputs[input_space.name]
            )
            for input_space in self.input_spaces
            if input_space.name != target_name
        }
        target_window = inputs[target_name].clone()  # (B, n_history, C_t, H_t, W_t)

        outputs: list[TensorNCHW] = []
        for _ in range(self.n_forecast_steps):
            # Set target window to the most recent observation/prediction
            windows[target_name] = target_window

            # Encode inputs to latent space
            # -> (B, n_history, C_latent_total, H_latent, W_latent)
            latent: TensorNTCHW = self._encode_inputs(windows)

            # Process in latent space
            # -> (B, C_latent_total, H_latent, W_latent)
            try:
                step_latent: TensorNCHW = self.processor.step(latent.unbind(dim=1))
            except NotImplementedError as exc:
                msg = (
                    f"Processor {type(self.processor).__name__} does not implement an "
                    "autoregressive forward() method."
                )
                raise ValueError(msg) from exc

            # Decode to physical space
            # -> (B, C_out, H_out, W_out)
            raw_output: TensorNCHW = self.decoder(step_latent)

            # Use the evolving rollout state as the skip-connection anchor, if required.
            # -> (B, C_out, H_out, W_out)
            anchor = self._extract_anchor(target_window)
            output = self.decoder.finalise(raw_output, anchor)
            outputs.append(output)

            # Drop the oldest target frame then append the newest with its target
            # variables replaced
            newest = target_window[:, -1].clone()
            newest[:, self.target_variable_indices] = output
            target_window = torch.cat(
                [target_window[:, 1:], newest.unsqueeze(1)], dim=1
            )

        return torch.stack(outputs, dim=1)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model (used for inference).

        Delegates to either the latent-space or physical-space path depending on the
        `rollout_space` setting.
        """
        if self.rollout_space == RolloutSpace.PHYSICAL:
            return self._forward_rollout_physical(inputs)
        return self._forward_rollout_latent(inputs)

    @override
    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the training step.

        The standard path is to call the forward step with `self(batch)` and compute the
        loss by comparing the decoded prediction to the target.

        Processors that set the flag `computes_loss_in_latent_space=True` are a special
        case. We encode the inputs over both the history and the forecast steps into
        latent space, and call `rollout()` on the processor, which returns both a latent
        prediction and a loss. We decode this prediction without gradients to calculate
        metrics.

        The latent loss does not go through the decoder. Thus, we can add two auxiliary
        losses. We skip each one if its weight is zero.

        - Autoencoder loss, weighted by `auxiliary_loss_weights.autoencoder`: we decode
          the encoded forecast-step inputs and compare them to the target. This loss
          trains the encoders and the decoder, and keeps the latent space compatible
          with the decoder. The processor stage always sets this weight to zero.
        - `pred_x0` loss, weighted by
          `auxiliary_loss_weights.pred_x0`: we decode the x_0 estimate from
          a diffusion processor and compare it to the target. This loss trains the
          processor, the decoder and the encoders against the target in physical space.

        Args:
            batch: Dictionary with one NTCHW entry per input dataset (n_history_steps)
                   and a "target" entry (n_forecast_steps). Latent-loss processors also
                   need one "future_<input>" entry per input dataset (n_forecast_steps).

        Returns:
            A ModelStepOutput containing the prediction, target and loss.

        """
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()

        # Custom loss path: use the loss returned by the processor
        if self.processor.computes_loss_in_latent_space:
            use_autoencoder_loss = self.weight_autoencoder > 0
            use_pred_x0_loss = self.weight_pred_x0 > 0

            # Encode history and future inputs into the combined latent space.
            history_latent = self._encode_inputs(batch)
            future_latent = self._encode_future_inputs(
                batch, with_grad=use_autoencoder_loss
            )

            # Process in latent space. Detach the target so that the latent loss
            # cannot decrease by changing its own target.
            processor_output = self.processor.rollout(
                history_latent, future_latent.detach()
            )

            # Get the loss from the processor output
            latent_loss = processor_output.loss
            if latent_loss is None:
                msg = (
                    f"Processor {type(self.processor).__name__} promises to compute "
                    "loss in latent space, but did not return one."
                )
                raise ValueError(msg)
            loss = latent_loss

            # Use persistence as the skip-connection anchor, if required.
            anchor = self._extract_anchor(batch[self.output_space.name])
            anchor_ntchw = None if anchor is None else anchor.unsqueeze(1)

            # Decode the encoded forecast-step inputs and compare them to the target.
            if use_autoencoder_loss:
                autoencoder_loss = self.loss(
                    self.decoder.rollout(future_latent, anchor_ntchw), target
                )
                self.log(
                    "train_autoencoder_loss",
                    autoencoder_loss,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                loss = loss + self.weight_autoencoder * autoencoder_loss

            # Decode the x_0 estimate and compare it to the target.
            if use_pred_x0_loss:
                pred_x0_loss = self._pred_x0_loss(
                    processor_output, target, anchor_ntchw
                )
                self.log(
                    "train_pred_x0_loss",
                    pred_x0_loss,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                loss = loss + self.weight_pred_x0 * pred_x0_loss

            # Log the latent loss on its own if an auxiliary loss was added to it.
            if use_autoencoder_loss or use_pred_x0_loss:
                self.log(
                    "train_latent_loss",
                    latent_loss,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

            # Decode the prediction without gradients, for metrics and callbacks only.
            with torch.no_grad():
                prediction = self.decoder.rollout(
                    processor_output.prediction, anchor_ntchw
                )

        # Standard path: calculate loss by comparing decoded output to target.
        else:
            prediction = self(batch)
            loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log_per_epoch("train_loss", loss, prog_bar=True)
        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
