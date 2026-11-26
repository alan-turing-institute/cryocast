import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import hydra
import torch
from omegaconf import DictConfig
from typing_extensions import override

from cryocast.models.common import ExponentialRelaxation
from cryocast.models.decoders import BaseDecoder
from cryocast.models.encoders import BaseEncoder
from cryocast.models.finalisers import Finaliser
from cryocast.models.objectives import TrainingObjective, build_training_objective
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
        "finaliser",
        "mask_dir",
        "processor",
    }
    # Relaxation has no weights, so a trained model can use a different relaxation
    checkpoint_overrides: ClassVar[frozenset[str]] = frozenset({"relaxation_tau"})

    def __init__(  # noqa: PLR0913
        self,
        *,
        encoders: DictConfig | list[BaseEncoder],
        processor: DictConfig | BaseProcessor,
        decoder: DictConfig | BaseDecoder,
        finaliser: Mapping[str, Any] | Finaliser | None = None,
        target_variable_indices: list[int],
        loss_weights: dict[str, float] | None = None,
        mask_dir: str | None = None,
        rollout_space: RolloutSpace | str = RolloutSpace.LATENT,
        skip_anchor: SkipAnchorType | str | None = None,
        relaxation_tau: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model.

        The processor flag `computes_loss_in_latent_space` selects the training
        objective. See `build_training_objective`.

        Args:
            encoders: DictConfig or list of BaseEncoder, one per input dataset.
            processor: DictConfig or BaseProcessor, the latent-space processor.
            decoder: DictConfig or BaseDecoder, the decoder from latent to raw output
                space.
            target_variable_indices: indices of the target variables within the output space.
            finaliser: Options or Finaliser, the finaliser that turns raw decoder output
                into a forecast state.
            loss_weights: weights for auxiliary loss terms. These apply only to
                processors that compute their loss in latent space. See LatentObjective.
            mask_dir: directory containing masks for the finaliser (if needed).
            rollout_space: RolloutSpace.LATENT or RolloutSpace.PHYSICAL (or the
                equivalent string), where to perform the forecast loop.
            skip_anchor: Whether to use the last, real observed frame or the most recent
                forecast step as the skip connection anchor. If None, use the default
                for the rollout space: INITIAL for latent and PREVIOUS for physical.
            relaxation_tau: The folding time, in forecast steps, of the relaxation of
                each forecast step towards climatology, or None for no relaxation.
                Relaxation needs a PREVIOUS skip anchor and a `climatology` entry in
                each batch.
            **kwargs: forwarded to ``BaseModel``.

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
            )
        )

        # Add a finaliser, which turns raw decoder output into a forecast state
        self.finaliser: Finaliser = (
            finaliser
            if isinstance(finaliser, Finaliser)
            else Finaliser(
                data_space=self.output_space, mask_dir=mask_dir, **(finaliser or {})
            )
        )

        # Resolve the rollout strategy. This validates the options and uses the default
        # anchor for the space if it is unset.
        self.rollout_strategy = RolloutStrategy.from_options(
            rollout_space, skip_anchor, relaxation_tau
        )

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
        if (
            self.rollout_strategy.relaxation_tau is not None
            and not self.rollout_strategy.is_physical
            and self.finaliser.skip_connection is None
        ):
            # Without a skip connection, no step carries over to the next step in a
            # latent rollout. Each step is then a fixed blend with climatology.
            msg = (
                "relaxation in a latent rollout requires a finaliser skip connection, "
                "so that each relaxed step carries over to the next step. Set "
                "finaliser.skip_connection, use rollout_space='physical', or remove "
                "relaxation."
            )
            raise ValueError(msg)
        if (
            self.finaliser.skip_connection is None
            and not self.rollout_strategy.uses_default_anchor
        ):
            log.warning(
                "skip_anchor='%s' has no effect because the finaliser has no skip "
                "connection.",
                self.rollout_strategy.anchor.value,
            )

        self.finaliser.relaxation = (
            None
            if relaxation_tau is None
            else ExponentialRelaxation(tau=relaxation_tau)
        )

        # Choose how to calculate the training loss
        self.objective: TrainingObjective = build_training_objective(self, loss_weights)

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
        return self.objective.needs_future_inputs

    def _anchor_nchw(self, window: TensorNTCHW) -> TensorNCHW | None:
        """Extract the target variables from the most recent frame.

        Args:
            window: TensorNTCHW holding the target dataset, most recent frame last.

        Returns:
            TensorNCHW anchor, or None if the finaliser has no skip connection.

        """
        if not self.finaliser.skip_connection:
            return None
        return window[:, -1, self.target_variable_indices, :, :]

    def _anchor_ntchw(self, window: TensorNTCHW) -> TensorNTCHW | None:
        """Extract the target variables from the most recent frame, keeping T=1.

        Args:
            window: TensorNTCHW holding the target dataset, most recent frame last.

        Returns:
            TensorNTCHW anchor, or None if the finaliser has no skip connection.

        """
        anchor = self._anchor_nchw(window)
        return None if anchor is None else anchor.unsqueeze(1)

    def _decode_forecast(
        self,
        latent: TensorNTCHW,
        anchor: TensorNTCHW | None,
        equilibrium: TensorNTCHW | None,
        teacher: TensorNTCHW | None = None,
    ) -> TensorNTCHW:
        """Decode a latent forecast with the skip anchor and relaxation of the strategy.

        Args:
            latent: TensorNTCHW latent forecast.
            anchor: Skip-connection anchor with shape (batch, 1, C_out, H, W), or None.
            equilibrium: Climatology to relax towards, with shape (batch, T, C_out, H,
                W), or None.
            teacher: Ground truth with the same shape as the forecast, or None. With a
                PREVIOUS anchor, each step after the first uses the ground truth of the
                step before it as its anchor, not the forecast (teacher forcing).

        """
        anchor_type = self.rollout_strategy.anchor
        if (
            teacher is not None
            and anchor is not None
            and anchor_type == SkipAnchorType.PREVIOUS
        ):
            anchor = torch.cat([anchor, teacher[:, :-1]], dim=1)
            anchor_type = SkipAnchorType.INITIAL
        return self.finaliser.rollout(
            self.decoder.rollout(latent),
            anchor=anchor,
            anchor_type=anchor_type,
            equilibrium=equilibrium,
        )

    def _equilibrium(self, batch: Mapping[str, TensorNTCHW]) -> TensorNTCHW | None:
        """Return the climatology to relax towards, or None if relaxation is off.

        Climatology comes from the training periods only, so it does not leak future
        information.

        Args:
            batch: Dictionary that holds a `climatology` TensorNTCHW entry with shape
                (batch, n_forecast_steps, n_output_channels, H_output, W_output).

        Raises:
            ValueError: If relaxation is on but the batch has no climatology.

        """
        if self.finaliser.relaxation is None:
            return None
        if (climatology := batch.get("climatology")) is None:
            msg = (
                "relaxation is on, but the batch has no 'climatology' entry. "
                "Climatology is only available if the data module has a training "
                "period that covers the target dataset."
            )
            raise ValueError(msg)
        return climatology

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

    def _forward_rollout_latent(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Rollout to the desired number of forecast steps in latent space.

        - start with multiple `NTCHW` inputs each with shape (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)
        - encode inputs to `NTCHW` latent space (batch, n_history_steps, n_latent_channels, H_latent, W_latent)
        - concatenate inputs in `NTCHW` latent space (batch, n_history_steps, n_latent_channels_total, H_latent, W_latent)
        - process in latent space `NTCHW` (batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent)
        - decode back to `NTCHW` output space (batch, n_forecast_steps, n_output_channels, H_output, W_output)
        - add a skip connection to each forecast step. Its anchor is the most recent
          target value, or the previous forecast step if `skip_anchor` is PREVIOUS
        - relax each forecast step towards climatology, if required

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
        return self._decode_forecast(latent_output, anchor, self._equilibrium(inputs))

    def _forward_rollout_physical(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Rollout to the desired number of forecast steps in physical space.

        Per step: encode the window of observed/predicted frames, take ONE processor
        step, decode to a physical field, then roll that field into the window and
        re-encode on the next iteration. Contrast with the default path, which appends
        the processor's raw latent to its own input window and never re-encodes.

        When the finaliser has an additive skip connection, the decoder produces a
        residual and the state advances as `x_{k+1} = clamp(x_k + residual_k, 0, 1)`, so
        a zero-output network reproduces persistence exactly.

        The skip connection anchor is the most recent frame in the target window, which
        is the previous forecast step (a PREVIOUS anchor).

        If relaxation is on, each step is relaxed towards climatology before it goes
        into the window.

        Non-target input groups hold their most recent OBSERVED frame for every forecast
        step. No future information enters. Only the input groups (the n_history_steps
        frames that end at the forecast origin) and "climatology" (from the training
        periods only) are read. The "target" and "future_<input>" keys hold the future
        frames, and are never read.

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

        # Relax each step towards climatology, if required
        # -> (B, n_forecast, C_out, H_out, W_out) or None
        equilibrium = self._equilibrium(inputs)

        outputs: list[TensorNCHW] = []
        for step in range(self.n_forecast_steps):
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

            # Use the evolving rollout state as the skip-connection anchor
            output = self.finaliser(
                raw_output,
                anchor=self._anchor_nchw(target_window),
                equilibrium=None if equilibrium is None else equilibrium[:, step],
            )
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

        The training objective calculates the prediction and the loss. See
        `cryocast.models.objectives`.

        Args:
            batch: Dictionary with one NTCHW entry per input dataset (n_history_steps)
                   and a "target" entry (n_forecast_steps). Latent-loss processors also
                   need one "future_<input>" entry per input dataset (n_forecast_steps).

        Returns:
            A ModelStepOutput containing the prediction, target and loss.

        """
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()
        prediction, loss = self.objective.training_step(self, batch, target)

        # Log metrics; computation will be done at epoch end
        self.log_per_epoch("train_loss", loss, prog_bar=True)
        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
