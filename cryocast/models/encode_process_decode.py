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
    RolloutSpace,
    RolloutStrategy,
    SkipAnchorType,
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
        loss_weights: dict[str, float] | None = None,
        mask_dir: str | None = None,
        rollout_space: RolloutSpace | str = RolloutSpace.LATENT,
        skip_anchor: SkipAnchorType | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model.

        Note that we use BaseProcessor.computes_loss_in_latent_space to control:
        - `needs_future_inputs`: training batches hold `future_<input>` keys.
        - the decoder is frozen if the reconstruction and prediction losses are off
        - `rollout_space`: cannot be set to "physical"
        - `training_step`: takes the latent-loss branch.

        Args:
            encoders: DictConfig or list of BaseEncoder, one per input dataset.
            processor: DictConfig or BaseProcessor, the latent-space processor.
            decoder: DictConfig or BaseDecoder, the decoder from latent to output space.
            target_variable_indices: indices of the target variables within the output space.
            loss_weights: weights for auxiliary loss terms. These apply only to
                processors that compute their loss in latent space. Weights must not be
                negative. Supported keys:
                - `reconstruction` (default 0.0): loss on the decoded encoded future
                  inputs.
                - `prediction` (default 0.0): loss on the decoded differentiable
                  prediction from the processor.
                If both weights are 0, the decoder is frozen.
            mask_dir: directory containing masks for the decoder (if needed).
            rollout_space: RolloutSpace.LATENT or RolloutSpace.PHYSICAL (or the
                equivalent string), where to perform the forecast loop.
            skip_anchor: Whether to use the last, real observed frame or the most recent
                forecast step as the skip connection anchor. If None, use the default
                for the rollout space: INITIAL for latent and PREVIOUS for physical.
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
        self._configure_losses(loss_weights or {})

        # Resolve the rollout strategy. This rejects invalid combinations of rollout
        # space and skip anchor, and uses the default anchor for the space if unset.
        self.rollout_strategy = RolloutStrategy.from_options(rollout_space, skip_anchor)

        # Check that the rollout strategy is compatible with the other components
        if self.rollout_strategy.is_physical:
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
        if (
            self.decoder.skip_connection is None
            and not self.rollout_strategy.uses_default_anchor
        ):
            log.warning(
                "skip_anchor='%s' has no effect because the decoder has no skip "
                "connection.",
                self.rollout_strategy.anchor.value,
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

    def _anchor_nchw(self, window: TensorNTCHW) -> TensorNCHW | None:
        """Extract the target variables from the most recent frame.

        Args:
            window: TensorNTCHW holding the target dataset, most recent frame last.

        Returns:
            TensorNCHW anchor, or None if the decoder has no skip connection.

        """
        if not self.decoder.skip_connection:
            return None
        return window[:, -1, self.target_variable_indices, :, :]

    def _anchor_ntchw(self, window: TensorNTCHW) -> TensorNTCHW | None:
        """Extract the target variables from the most recent frame, keeping T=1.

        Args:
            window: TensorNTCHW holding the target dataset, most recent frame last.

        Returns:
            TensorNTCHW anchor, or None if the decoder has no skip connection.

        """
        anchor = self._anchor_nchw(window)
        return None if anchor is None else anchor.unsqueeze(1)

    def _configure_losses(self, weights: Mapping[str, float]) -> None:
        """Validate the auxiliary loss weights and freeze the decoder if it is unused.

        Args:
            weights: weights for auxiliary loss terms, see `__init__`.

        Raises:
            ValueError: If a key is unknown or a weight is negative.

        """
        defaults = {"reconstruction": 0.0, "prediction": 0.0}
        for key, weight in weights.items():
            if key not in defaults:
                msg = (
                    f"Unknown loss_weights key '{key}'. Supported keys: "
                    f"{sorted(defaults)}."
                )
                raise ValueError(msg)
            if weight < 0:
                msg = f"loss_weights.{key} must not be negative, but it is {weight}."
                raise ValueError(msg)
        self.loss_weights = {
            key: float(weights.get(key, default)) for key, default in defaults.items()
        }

        if self.processor.computes_loss_in_latent_space:
            # Only the auxiliary losses use the decoder. If they are all off, freeze it.
            if not any(self.loss_weights.values()):
                self.decoder.freeze()

            # Without the reconstruction loss, nothing keeps the latent space and the
            # decoder compatible while they train.
            if self.loss_weights["reconstruction"] == 0 and any(
                parameter.requires_grad
                for module in (*self.encoders, self.decoder)
                for parameter in module.parameters()
            ):
                log.warning(
                    "loss_weights.reconstruction is 0, but the encoders or the "
                    "decoder are trainable. Without the reconstruction loss, the latent "
                    "space can drift away from what the decoder expects."
                )

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
                also goes into the reconstruction loss. The caller must then detach the
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

        # Do not record gradients unless the reconstruction loss needs them. This saves
        # the memory for the encoder activations over the forecast steps.
        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()):
            return self._encode_inputs(future_inputs)

    def _forward_rollout_latent(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Rollout to the desired number of forecast steps in latent space.

        - start with multiple `NTCHW` inputs each with shape (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)
        - encode inputs to `NTCHW` latent space (batch, n_history_steps, n_latent_channels, H_latent, W_latent)
        - concatenate inputs in `NTCHW` latent space (batch, n_history_steps, n_latent_channels_total, H_latent, W_latent)
        - process in latent space `NTCHW` (batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent)
        - decode back to `NTCHW` output space (batch, n_forecast_steps, n_output_channels, H_output, W_output)
        - add a skip connection to each forecast step. Its anchor is the most recent
          target value, or the previous forecast step if `skip_anchor` is PREVIOUS

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

        # Use persistence as the skip-connection anchor for the first step, if required
        anchor = self._anchor_ntchw(inputs[self.output_space.name])

        # Decode to output space: tensor with (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)
        return self.decoder.rollout(
            latent_output, anchor, anchor_type=self.rollout_strategy.anchor
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

        The skip connection anchor is always the most recent frame in the target window,
        corresponding to the previous forecast step. This is equivalent to
        `SkipAnchorType.PREVIOUS`.

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
            anchor = self._anchor_nchw(target_window)
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

    def _prediction_loss(
        self,
        prediction: TensorNTCHW,
        mask: torch.Tensor | None,
        target: TensorNTCHW,
        anchor: TensorNTCHW | None,
    ) -> torch.Tensor:
        """Decode the processor's latent prediction and compare it to the target.

        Only the samples in `mask` count. The prediction can hold fewer forecast steps
        than the target (for example, only the first step for an autoregressive
        processor), so we compare it to the same number of target steps.

        Args:
            prediction: Latent prediction from the processor training step, with
                gradients back to the processor. Shape (batch, T, C_latent, H, W), with
                T <= n_forecast_steps.
            mask: Boolean mask of shape (batch,), or None to use all samples.
            target: TensorNTCHW with shape (batch, n_forecast_steps, C_out, H, W).
            anchor: Skip-connection anchor with shape (batch, 1, C_out, H, W), or None.

        Returns:
            Scalar loss. It is 0 if no sample is in the mask.

        Raises:
            ValueError: If the prediction has no gradients.

        """
        if not prediction.requires_grad:
            msg = (
                f"loss_weights.prediction is greater than 0, but processor "
                f"{type(self.processor).__name__} returns a prediction without "
                f"gradients during training. Return a differentiable prediction or "
                f"set the weight to 0."
            )
            raise ValueError(msg)

        # The mean over an empty mask is NaN, so return 0. Keep the decoder parameters
        # in the graph without decoding anything: with DDP, a parameter that has no
        # gradient on one process stops training.
        if mask is not None and not mask.any():
            return (
                sum(
                    (p.sum() for p in self.decoder.parameters() if p.requires_grad),
                    start=prediction.new_zeros(()),
                )
                * 0.0
            )

        # Decode only the samples in the mask. Without a mask, all samples count.
        samples = slice(None) if mask is None else mask
        decoded = self.decoder.rollout(
            prediction[samples],
            None if anchor is None else anchor[samples],
            anchor_type=self.rollout_strategy.anchor,
        )
        return self.loss(decoded, target[samples, : prediction.shape[1]])

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model (used for inference).

        Delegates to either the latent-space or physical-space path depending on the
        `rollout_strategy` setting.
        """
        if self.rollout_strategy.is_physical:
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

        - Reconstruction loss, weighted by `loss_weights.reconstruction`: we decode the
          encoded forecast-step inputs and compare them to the target. This loss trains
          the encoders and the decoder, and keeps the latent space compatible with the
          decoder.
        - Prediction loss, weighted by `loss_weights.prediction`: we decode the latent
          prediction from the processor, with gradients, and compare it to the target.
          Only the samples in `sample_mask` count. This loss trains the processor, the
          decoder and the encoders against the target in physical space.

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
            # Encode history and future inputs into the combined latent space.
            history_latent = self._encode_inputs(batch)
            future_latent = self._encode_future_inputs(
                batch, with_grad=self.loss_weights["reconstruction"] > 0
            )

            # Process in latent space, detaching future inputs so that the latent loss
            # cannot decrease by changing its own target.
            processor_output = self.processor.rollout(
                history_latent, future_latent.detach()
            )
            # Check that the processor returned a loss
            if (latent_loss := processor_output.loss) is None:
                msg = (
                    f"Processor {type(self.processor).__name__} promises to compute "
                    "loss in latent space, but did not return one."
                )
                raise ValueError(msg)
            # Check the number of forecast steps is in the expected range
            n_steps = processor_output.prediction.shape[1]
            if not 1 <= n_steps <= self.n_forecast_steps:
                msg = (
                    f"Processor {type(self.processor).__name__} returned a prediction "
                    f"with {n_steps} forecast steps, but it must have between 1 and "
                    f"{self.n_forecast_steps}."
                )
                raise ValueError(msg)

            # Use persistence as the skip-connection anchor, if required.
            anchor = self._anchor_ntchw(batch[self.output_space.name])

            # Calculate each auxiliary loss that has a non-zero weight.
            losses: dict[str, torch.Tensor] = {}
            if self.loss_weights["reconstruction"] > 0:
                # Decode the encoded forecast-step inputs and compare them to the target
                losses["reconstruction"] = self.loss(
                    self.decoder.rollout(
                        future_latent, anchor, anchor_type=self.rollout_strategy.anchor
                    ),
                    target,
                )
            if self.loss_weights["prediction"] > 0:
                # Decode the differentiable prediction and compare it to the target
                losses["prediction"] = self._prediction_loss(
                    processor_output.prediction,
                    processor_output.sample_mask,
                    target,
                    anchor,
                )

            # Add the weighted auxiliary losses to the latent loss
            loss = sum(
                (self.loss_weights[name] * value for name, value in losses.items()),
                start=latent_loss,
            )

            # Log each term on its own if there are auxiliary losses.
            if losses:
                for name, value in {"latent": latent_loss, **losses}.items():
                    self.log_per_epoch(f"train_{name}_loss", value)

            # Decode the prediction in evaluation mode and without gradients, so that
            # we do not affect BatchNorm running statistics.
            training_mode = self.decoder.training
            self.decoder.eval()
            with torch.no_grad():
                prediction = self.decoder.rollout(
                    processor_output.prediction,
                    anchor,
                    anchor_type=self.rollout_strategy.anchor,
                )
            self.decoder.train(training_mode)
            # If the latent prediction has fewer forecast steps than the target, repeat
            # the last one until we have the correct number
            steps = torch.arange(self.n_forecast_steps, device=prediction.device)
            prediction = prediction[:, steps.clamp(max=n_steps - 1)]

        # Standard path: calculate loss by comparing decoded output to target.
        else:
            prediction = self(batch)
            loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log_per_epoch("train_loss", loss, prog_bar=True)
        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
