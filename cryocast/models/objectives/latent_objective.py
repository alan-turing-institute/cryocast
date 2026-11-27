import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from cryocast.types import SkipAnchorType, TensorNTCHW

from .training_objective import TrainingObjective

if TYPE_CHECKING:
    from cryocast.models.encode_process_decode import EncodeProcessDecode

log = logging.getLogger(__name__)


class LatentObjective(TrainingObjective):
    """Use the loss that the processor calculates in latent space.

    This objective is for processors that set `computes_loss_in_latent_space=True`. We
    encode the inputs over both the history and the forecast steps into latent space,
    and call `rollout()` on the processor, which returns both a latent prediction and a
    loss. We decode this prediction without gradients to calculate metrics.

    The latent loss does not go through the decoder. Thus, we can add two auxiliary
    losses. We skip each one if its weight is zero.

    - Reconstruction loss, weighted by `loss_weights.reconstruction`: we decode the
      encoded forecast-step inputs and compare them to the target. This loss trains the
      encoders and the decoder, and keeps the latent space compatible with the decoder.
    - Prediction loss, weighted by `loss_weights.prediction`: we decode the latent
      prediction from the processor, with gradients, and compare it to the target. Only
      the samples in `sample_mask` count. This loss trains the processor, the decoder
      and the encoders against the target in physical space.

    If both weights are 0, nothing trains the decoder and the finaliser, so we freeze
    them.
    """

    # The latent loss compares forecasts against encoded inputs, so it needs inputs over
    # the forecast steps.
    needs_future_inputs = True

    def __init__(
        self,
        model: "EncodeProcessDecode",
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        """Validate the objective against the model, and freeze what it cannot train.

        The objective does not keep a reference to the model.

        Args:
            model: The model that owns this objective.
            loss_weights: Weights for the `reconstruction` and `prediction` loss terms
                (default 0.0 each). Weights must not be negative.

        Raises:
            ValueError: If a key is unknown, a weight is negative, or the model uses
                physical rollout.

        """
        loss_weights = loss_weights or {}
        self.loss_weights = {"reconstruction": 0.0, "prediction": 0.0}
        for key, weight in loss_weights.items():
            if key not in self.loss_weights:
                msg = (
                    f"Unknown loss_weights key '{key}'. Supported keys: "
                    f"{sorted(self.loss_weights)}."
                )
                raise ValueError(msg)
            if weight < 0:
                msg = f"loss_weights.{key} must not be negative, but it is {weight}."
                raise ValueError(msg)
            self.loss_weights[key] = float(weight)

        # Physical rollout passes the output through the decoder, so the latent loss
        # cannot train it.
        if model.rollout_strategy.is_physical:
            msg = (
                f"rollout_space='physical' is incompatible with processor "
                f"{type(model.processor).__name__}, which computes its loss in "
                f"latent space."
            )
            raise ValueError(msg)

        # Only the auxiliary losses use the decoder and the finaliser. If they are all
        # off, freeze both.
        if not any(self.loss_weights.values()):
            model.decoder.freeze()
            model.finaliser.freeze()

        # Without the reconstruction loss, nothing keeps the latent space and the
        # decoder compatible while they train.
        if self.loss_weights["reconstruction"] == 0 and any(
            parameter.requires_grad
            for module in (*model.encoders, model.decoder)
            for parameter in module.parameters()
        ):
            log.warning(
                "loss_weights.reconstruction is 0, but the encoders or the "
                "decoder are trainable. Without the reconstruction loss, the latent "
                "space can drift away from what the decoder expects."
            )

    @override
    def training_step(
        self,
        model: "EncodeProcessDecode",
        batch: dict[str, TensorNTCHW],
        target: TensorNTCHW,
    ) -> tuple[TensorNTCHW, torch.Tensor]:
        # Encode history and future inputs into the combined latent space.
        history_latent = model._encode_inputs(batch)
        future_latent = self._encode_future_inputs(
            model, batch, with_grad=self.loss_weights["reconstruction"] > 0
        )

        # Process in latent space, detaching future inputs so that the latent loss
        # cannot decrease by changing its own target.
        processor_output = model.processor.rollout(
            history_latent, future_latent.detach()
        )
        # Check that the processor returned a loss
        if (latent_loss := processor_output.loss) is None:
            msg = (
                f"Processor {type(model.processor).__name__} promises to compute "
                "loss in latent space, but did not return one."
            )
            raise ValueError(msg)
        # Check the number of forecast steps is in the expected range
        n_steps = processor_output.prediction.shape[1]
        if not 1 <= n_steps <= model.n_forecast_steps:
            msg = (
                f"Processor {type(model.processor).__name__} returned a prediction "
                f"with {n_steps} forecast steps, but it must have between 1 and "
                f"{model.n_forecast_steps}."
            )
            raise ValueError(msg)

        # Use persistence as the skip-connection anchor, if required.
        anchor = model._anchor_ntchw(batch[model.output_space.name])

        # Relax the forecast towards climatology, if required. The reconstruction
        # loss decodes observations, not a forecast, so it is not relaxed.
        equilibrium = model._equilibrium(batch)

        # Calculate each auxiliary loss that has a non-zero weight.
        losses: dict[str, torch.Tensor] = {}
        if self.loss_weights["reconstruction"] > 0:
            # Decode the encoded forecast-step inputs and compare them to the target.
            # With a PREVIOUS anchor, we use teacher forcing: each step after the first
            # uses the ground-truth for the previous forecast step, not the output of
            # the previous step.
            teacher_anchor = anchor
            if (
                anchor is not None
                and model.rollout_strategy.anchor == SkipAnchorType.PREVIOUS
            ):
                teacher_anchor = torch.cat([anchor, target[:, :-1]], dim=1)
            reconstruction = model.finaliser.rollout(
                model.decoder.rollout(future_latent), anchor=teacher_anchor
            )
            losses["reconstruction"] = model.loss(reconstruction, target)
        if self.loss_weights["prediction"] > 0:
            # Decode the differentiable prediction and compare it to the target
            losses["prediction"] = self._prediction_loss(
                model,
                processor_output.prediction,
                processor_output.sample_mask,
                target,
                anchor,
                equilibrium,
            )

        # Add the weighted auxiliary losses to the latent loss
        loss = sum(
            (self.loss_weights[name] * value for name, value in losses.items()),
            start=latent_loss,
        )

        # Log each term on its own if there are auxiliary losses.
        if losses:
            for name, value in {"latent": latent_loss, **losses}.items():
                model.log_per_epoch(f"train_{name}_loss", value)

        # Decode the prediction in evaluation mode and without gradients, so that
        # we do not affect BatchNorm running statistics in the decoder or finaliser.
        training_modes = {
            module: module.training for module in (model.decoder, model.finaliser)
        }
        for module in training_modes:
            module.eval()
        try:
            with torch.no_grad():
                prediction = model._decode_forecast(
                    processor_output.prediction, anchor, equilibrium
                )
        finally:
            for module, training_mode in training_modes.items():
                module.train(training_mode)
        # If the latent prediction has fewer forecast steps than the target, repeat
        # the last one until we have the correct number
        steps = torch.arange(model.n_forecast_steps, device=prediction.device)
        return prediction[:, steps.clamp(max=n_steps - 1)], loss

    @staticmethod
    def _encode_future_inputs(
        model: "EncodeProcessDecode",
        batch: dict[str, TensorNTCHW],
        *,
        with_grad: bool = False,
    ) -> TensorNTCHW:
        """Encode every input over the forecast steps into the combined latent space.

        This is the ground truth for the latent loss.

        Args:
            model: The model that owns this objective.
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
                for input_space in model.input_spaces
            }
        except KeyError as exc:
            msg = (
                f"Processor {type(model.processor).__name__} computes its loss in "
                f"latent space, so the batch needs future inputs, but {exc} is "
                f"missing."
            )
            raise ValueError(msg) from exc

        # Do not record gradients unless the reconstruction loss needs them. This saves
        # the memory for the encoder activations over the forecast steps.
        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()):
            return model._encode_inputs(future_inputs)

    @staticmethod
    def _prediction_loss(
        model: "EncodeProcessDecode",
        prediction: TensorNTCHW,
        mask: torch.Tensor | None,
        target: TensorNTCHW,
        anchor: TensorNTCHW | None,
        equilibrium: TensorNTCHW | None,
    ) -> torch.Tensor:
        """Decode the processor's latent prediction and compare it to the target.

        Only the samples in `mask` count. The prediction can hold fewer forecast steps
        than the target (for example, only the first step for an autoregressive
        processor), so we compare it to the same number of target steps.

        Args:
            model: The model that owns this objective.
            prediction: Latent prediction from the processor training step, with
                gradients back to the processor. Shape (batch, T, C_latent, H, W), with
                T <= n_forecast_steps.
            mask: Boolean mask of shape (batch,), or None to use all samples.
            target: TensorNTCHW with shape (batch, n_forecast_steps, C_out, H, W).
            anchor: Skip-connection anchor with shape (batch, 1, C_out, H, W), or None.
            equilibrium: Climatology to relax towards, with shape (batch,
                n_forecast_steps, C_out, H, W), or None.

        Returns:
            Scalar loss. It is 0 if no sample is in the mask.

        Raises:
            ValueError: If the prediction has no gradients.

        """
        if not prediction.requires_grad:
            msg = (
                f"loss_weights.prediction is greater than 0, but processor "
                f"{type(model.processor).__name__} returns a prediction without "
                f"gradients during training. Return a differentiable prediction or "
                f"set the weight to 0."
            )
            raise ValueError(msg)

        # The mean over an empty mask is NaN, so return 0. Keep the decoder and finaliser
        # parameters in the graph without decoding anything: with DDP, a parameter that
        # has no gradient on one process stops training.
        if mask is not None and not mask.any():
            return (
                sum(
                    (
                        p.sum()
                        for module in (model.decoder, model.finaliser)
                        for p in module.parameters()
                        if p.requires_grad
                    ),
                    start=prediction.new_zeros(()),
                )
                * 0.0
            )

        # Decode only the samples in the mask. Without a mask, all samples count.
        samples = slice(None) if mask is None else mask
        decoded = model._decode_forecast(
            prediction[samples],
            None if anchor is None else anchor[samples],
            None if equilibrium is None else equilibrium[samples],
        )
        return model.loss(decoded, target[samples, : prediction.shape[1]])
