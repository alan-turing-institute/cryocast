import copy
import logging
from typing import TYPE_CHECKING, Any, ClassVar

import hydra
import torch
from omegaconf import DictConfig
from typing_extensions import override

from icenet_mp.models import BaseModel
from icenet_mp.types import DataSpace, TensorNTCHW

from .encoder_stage import EncoderStage

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder

logger = logging.getLogger(__name__)


class DecoderStage(BaseModel):
    # Parameters that should be excluded from hyperparameter logging
    ignored_hparams: ClassVar[frozenset[str]] = BaseModel.ignored_hparams | {"encoders"}

    def __init__(
        self,
        decoder: DictConfig,
        encoders: list[EncoderStage],
        target_dataset_name: str,
        target_variable_indices: list[int],
        mask_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a DecoderStage with multiple frozen encoders and a trainable decoder."""
        super().__init__(mask_dir=mask_dir, **kwargs)

        # We require at least two history steps to train the decoder
        if self.n_history_steps < 2:  # noqa: PLR2004
            msg = f"DecoderStage requires at least two history steps: found {self.n_history_steps}."
            raise ValueError(msg)

        # Copy encoders from EncoderStages, freeze their parameters and register them.
        self.encoder_names = [encoder.dataset_name for encoder in encoders]
        self.encoders = [
            copy.deepcopy(encoder.encoder).freeze() for encoder in encoders
        ]
        for encoder in self.encoders:
            self.add_module(encoder.name, encoder)

        # Verify the output channels for each encoder
        for encoder in self.encoders:
            encoder.verify_output_channels(self.device)

        # Build combined latent DataSpace by summing channels across all encoders
        total_channels = sum(
            encoder.data_space_out.channels for encoder in self.encoders
        )
        combined_latent_space = DataSpace(
            channels=total_channels,
            name="combined_latent",
            shape=self.encoders[0].data_space_out.shape,
        )

        # Decode from combined latent space to the configured output space
        self.target_name = target_dataset_name
        self.target_variable_indices = target_variable_indices
        if self.output_space.channels != len(target_variable_indices):
            msg = (
                f"output_space has {self.output_space.channels} channel(s) but "
                f"target_variable_indices selects {len(target_variable_indices)}; "
                f"check that predict.target.variables is set correctly."
            )
            raise ValueError(msg)
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            data_space_in=combined_latent_space,
            data_space_out=self.output_space,
            mask_dir=mask_dir,
        )

    @classmethod
    def from_template(
        cls,
        *,
        decoder: DictConfig,
        encoders: list[EncoderStage],
        target_dataset_name: str,
        target_variable_indices: list[int],
        mask_dir: str | None = None,
    ) -> "DecoderStage":
        """Create a DecoderStage from a list of trained EncoderStages."""
        return cls(
            decoder=decoder,
            encoders=encoders,
            target_dataset_name=target_dataset_name,
            target_variable_indices=target_variable_indices,
            mask_dir=mask_dir,
            hemisphere=encoders[0].hemisphere,
            input_spaces=[s.to_dict() for s in encoders[0].input_spaces],
            lr_scheduler=copy.deepcopy(encoders[0].lr_scheduler_cfg),
            n_forecast_steps=encoders[0].n_forecast_steps,
            n_history_steps=encoders[0].n_history_steps,
            name=f"{target_dataset_name}_decoder".replace("-", "_"),
            optimizer=copy.deepcopy(encoders[0].optimizer_cfg),
            output_space=encoders[0].output_space.to_dict(),
            scheduler=copy.deepcopy(encoders[0].scheduler_cfg),
            loss=copy.deepcopy(encoders[0].loss_cfg),
            metrics=copy.deepcopy(encoders[0].metrics),
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - squeeze time dimension from each input to get [NCHW]
        - encode each dataset with its corresponding frozen encoder
        - concatenate latent representations along the channel dimension
        - decode to target space via rollout() so masking/restrict_range apply
        """
        latents = [
            encoder(inputs[name].squeeze(1))
            for name, encoder in zip(self.encoder_names, self.encoders, strict=True)
        ]
        combined = torch.cat(latents, dim=1).unsqueeze(1)
        return self.decoder.rollout(combined, inputs["persistence"])

    def process_batch(
        self,
        batch: dict[str, TensorNTCHW],
    ) -> dict[str, TensorNTCHW]:
        """Extract only the two time steps from each relevant batch element.

        Inputs use t=-2 (yesterday) while the target uses t=-1 (today). We also extract
        a persistence entry using t=-2 but sliced to only the target variables.

        This is because we want the decoder to learn an NCHW -> NCHW mapping but also to
        include the most recent target value as a skip connection for each forecast so
        that the decoder will learn to predict residuals. If we use the same time step
        for both input and target, the model will learn that these residuals are zero.
        """
        return {
            name: batch[name][:, -2, :, :, :].unsqueeze(1)
            for name in self.encoder_names
        } | {
            "target": batch[self.target_name][
                :, -1, self.target_variable_indices, :, :
            ].unsqueeze(1),
            "persistence": batch[self.target_name][
                :, -2, self.target_variable_indices, :, :
            ].unsqueeze(1),
        }

    @override
    def train(self, mode: bool = True) -> "DecoderStage":
        """Set training mode, but keep the frozen encoders in eval mode."""
        super().train(mode)
        if mode:
            for encoder in self.encoders:
                encoder.eval()
        return self
