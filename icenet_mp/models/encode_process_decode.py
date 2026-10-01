from collections.abc import Mapping
from typing import Any, ClassVar

import hydra
import torch
from omegaconf import DictConfig
from typing_extensions import override

from icenet_mp.models.decoders import BaseDecoder
from icenet_mp.models.encoders import BaseEncoder
from icenet_mp.models.processors import BaseProcessor
from icenet_mp.types import (
    DataSpace,
    ModelStepOutput,
    RolloutSpace,
    TensorNCHW,
    TensorNTCHW,
)

from .base_model import BaseModel


class EncodeProcessDecode(BaseModel):
    """Model that encodes to latent space, processes, then decodes back."""

    # Parameters that should be excluded from hyperparameter logging (e.g. local paths)
    ignored_hparams: ClassVar[frozenset[str]] = BaseModel.ignored_hparams | {
        "decoder",
        "encoders",
        "mask_dir",
        "processor",
    }

    def __init__(
        self,
        *,
        encoders: DictConfig | list[BaseEncoder],
        processor: DictConfig | BaseProcessor,
        decoder: DictConfig | BaseDecoder,
        target_variable_indices: list[int],
        mask_dir: str | None = None,
        rollout_space: RolloutSpace | str = RolloutSpace.LATENT,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model.

        Args:
            encoders: DictConfig or list of BaseEncoder, one per input dataset.
            processor: DictConfig or BaseProcessor, the latent-space processor.
            decoder: DictConfig or BaseDecoder, the decoder from latent to output space.
            target_variable_indices: indices of the target variables within the output space.
            mask_dir: directory containing masks for the decoder (if needed).
            rollout_space: RolloutSpace.LATENT or RolloutSpace.PHYSICAL (or the
                equivalent string), where to perform the forecast loop.
            **kwargs: forwarded to ``BaseModel`` (spaces, masks, range, skip).

        """
        super().__init__(mask_dir=mask_dir, **kwargs)

        # Check that the number of variable indices provided matches the number of
        # channels in the output space.
        if self.output_space.channels != len(target_variable_indices):
            msg = (
                f"output_space has {self.output_space.channels} channel(s) but "
                f"target_variable_indices selects {len(target_variable_indices)}; "
                f"check that predict.target.variables is set correctly."
            )
            raise ValueError(msg)
        self.target_variable_indices = target_variable_indices

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

        # Add an additional encoder that encodes the target dataset into latent space
        # This will be used by any processors that need to compute latent space losses.
        try:
            self.target_encoder: BaseEncoder = (
                hydra.utils.instantiate(
                    encoders[self.output_space.name],
                    data_space_in=DataSpace(
                        name="target",
                        channels=self.output_space.channels,
                        shape=self.output_space.shape,
                    ),
                    latent_space=encoders["latent_space"],
                    latitudes_fn=self.latitudes_fn,
                    longitudes_fn=self.longitudes_fn,
                )
                if isinstance(encoders, Mapping)
                else encoders.pop(
                    next(
                        idx
                        for idx, encoder in enumerate(encoders)
                        if encoder.name == "target"
                    )
                )
            )
        except KeyError as exc:
            msg = (
                f"Error instantiating target encoder: {exc}. Please ensure that an "
                f"encoder is specified for '{self.output_space.name}', even if it is "
                f"not one of the input spaces: {self.input_spaces}."
            )
            raise ValueError(msg) from exc

        # Verify the output channels for each encoder
        for encoder in (*self.encoders, self.target_encoder):
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
                data_space=combined_latent_space,
                data_space_target=self.target_encoder.data_space_out,
                n_forecast_steps=self.n_forecast_steps,
                n_history_steps=self.n_history_steps,
                target_channel_offset=self._find_target_channel_offset(),
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

        # Validate rollout options
        self.rollout_space = self._validate_rollout_options(rollout_space)

        # The physical rollout drives the decoder itself and computes the loss on the
        # decoded field, so it cannot host a processor that owns the training signal
        # in latent space (whose decoder is frozen below).
        if (
            self.rollout_space == RolloutSpace.PHYSICAL
            and self.processor.computes_loss_in_latent_space
        ):
            msg = (
                f"rollout_space='physical' is incompatible with processor "
                f"{type(self.processor).__name__}, which computes its loss in latent "
                f"space (the decoder is frozen and never trained on that path)."
            )
            raise ValueError(msg)

        # Freeze unused modules
        self._freeze_unused_modules()

    @property
    def multistage_only(self) -> bool:
        return self.processor.computes_loss_in_latent_space

    def _encode_inputs(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Encode all input datasets and concatenate along the channel dimension.

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

    def _find_target_channel_offset(self) -> int | None:
        """Find the channel offset of the target dataset within the combined latent space, if present."""
        offset = 0
        for encoder, input_space in zip(self.encoders, self.input_spaces, strict=True):
            if input_space.name == self.output_space.name:
                return offset
            offset += encoder.data_space_out.channels
        return None

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
        step. No future information enters: only `inputs[...]`, which holds the
        n_history_steps frames ending at the forecast origin, is ever read; the ground
        truth lives under the separate reserved key "target" and is not touched here.

        Args:
            inputs: Dictionary with one TensorNTCHW entry per input dataset with shape
                (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)

        Returns:
            TensorNTCHW with shape (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)

        """
        target_name = self.output_space.name

        # Non-target groups: hold the newest observed frame for the whole rollout.
        windows: dict[str, TensorNTCHW] = {  # (B, n_history, C_k, H_k, W_k)
            name: tensor[:, -1:].expand_as(tensor)
            for name, tensor in inputs.items()
            if name not in {target_name, "target"}
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

            # Drop the oldest target frame; append the newest with its target variables replaced
            # Replace the oldest target frame by the newest, with
            newest = target_window[:, -1].clone()
            newest[:, self.target_variable_indices] = output
            target_window = torch.cat(
                [target_window[:, 1:], newest.unsqueeze(1)], dim=1
            )

        return torch.stack(outputs, dim=1)

    def _freeze_unused_modules(self) -> None:
        """Freeze unused modules."""
        # Processors that compute loss in latent space do not touch the decoder.
        # However, processors that do not do this, do not touch the target_encoder.
        # We therefore explicitly freeze the unused modules.
        if self.processor.computes_loss_in_latent_space:
            self.decoder.freeze()
        else:
            self.target_encoder.freeze()

    def _validate_rollout_options(
        self, rollout_space: RolloutSpace | str
    ) -> RolloutSpace:
        """Reject a rollout_space setting that cannot work, before anything is built."""
        rollout_space = RolloutSpace(rollout_space)

        if rollout_space == RolloutSpace.PHYSICAL:
            input_names = {input_space.name for input_space in self.input_spaces}
            if self.output_space.name not in input_names:
                msg = (
                    f"rollout_space='physical' requires the prediction target group "
                    f"'{self.output_space.name}' to also be a model input, so that "
                    f"the rollout has an observed physical state to advance. Inputs: "
                    f"{sorted(input_names)}."
                )
                raise ValueError(msg)

        return rollout_space

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model (used for inference).

        Delegates to either the latent-space or physical-space path depending on the
        `rollout_space` setting.
        """
        if self.rollout_space == RolloutSpace.PHYSICAL:
            return self._forward_rollout_physical(inputs)
        return self._forward_rollout_latent(inputs)

    @override
    def train(self, mode: bool = True) -> "EncodeProcessDecode":
        """Set training mode, with decoder frozen if computing loss in latent space."""
        super().train(mode)
        if mode:
            self._freeze_unused_modules()
        return self

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the training step.

        The standard path is to call the forward step with `self(batch)` and compute the
        loss by comparing the decoded prediction to the target.

        Processors that set the flag `computes_loss_in_latent_space=True` are a special
        case. We encode the inputs and target into latent space, call `rollout()` on the
        processor, which returns both a latent prediction and a loss. We pass this
        prediction through a frozen decoder in order to calculate metrics, but we use
        the loss returned by the processor for backpropagation.

        Args:
            batch: Dictionary with one NTCHW entry per input dataset (n_history_steps)
                   and a "target" entry (n_forecast_steps).

        Returns:
            A ModelStepOutput containing the prediction, target and loss.

        """
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()

        # Custom loss path: use the loss returned by the processor
        if self.processor.computes_loss_in_latent_space:
            expected_chw = self.target_encoder.data_space_in.chw
            if tuple(target.shape[2:]) != expected_chw:
                msg = (
                    f"Target CHW shape ({tuple(target.shape[2:])}) does not match the "
                    f"shape expected by the '{self.target_encoder.name}' encoder "
                    f"({expected_chw})"
                )
                raise ValueError(msg)

            # Encode inputs into latent space
            latent_input_combined = self._encode_inputs(batch)

            # Process in latent space
            processor_output = self.processor.rollout(
                latent_input_combined, self.target_encoder.rollout(target)
            )

            # Get the loss from the processor output
            loss = processor_output.loss
            if loss is None:
                msg = (
                    f"Processor {type(self.processor).__name__} promises to compute "
                    "loss in latent space, but did not return one."
                )
                raise ValueError(msg)

            # Decode under no_grad for metrics/callbacks only.
            # Use persistence as the skip-connection anchor, if required.
            anchor = self._extract_anchor(batch[self.output_space.name])
            with torch.no_grad():
                prediction = self.decoder.rollout(
                    processor_output.prediction,
                    None if anchor is None else anchor.unsqueeze(1),
                )

        # Standard path: calculate loss by comparing decoded output to target.
        else:
            prediction = self(batch)
            loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
