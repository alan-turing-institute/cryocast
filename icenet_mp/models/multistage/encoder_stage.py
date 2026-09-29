import copy
import logging
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig

from icenet_mp.models import BaseModel, EncodeProcessDecode
from icenet_mp.types import DataSpace, TensorNTCHW

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder
    from icenet_mp.models.encoders import BaseEncoder

logger = logging.getLogger(__name__)


class EncoderStage(BaseModel):
    def __init__(
        self,
        channel_names: list[str],
        data_space_in: DataSpace,
        encoder: DictConfig,
        decoder: DictConfig,
        latent_space: tuple[int, int],
        **kwargs: Any,
    ) -> None:
        """Initialise an EncoderStage with a trainable encoder and a disposable decoder."""
        super().__init__(**kwargs)

        # Store channel names
        self.channel_names = channel_names

        # Encode from a single input space to a latent space. For most datasets this
        # space is one of the model's raw input spaces, found by name. The target
        # dataset is a special case (a subset of variables from a dataset that may
        # also be a raw input in its own right, read from the batch's "target" key
        # rather than a dataset-named key), so its DataSpace is passed in directly.
        self.encoder: BaseEncoder = hydra.utils.instantiate(
            encoder,
            data_space_in=data_space_in,
            latent_space=latent_space,
            latitudes_fn=self.latitudes_fn,
            longitudes_fn=self.longitudes_fn,
        )

        # Verify the output channels for the encoder
        self.encoder.verify_output_channels(self.device)

        # Decode from the latent space back to the original input space. This decoder
        # is disposable and reconstructs the input (not the forecast target), so masking
        # and skip connections are disabled here.
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            data_space_in=self.encoder.data_space_out,
            data_space_out=self.encoder.data_space_in,
            mask_type=None,
            restrict_range=None,
            skip_connection=None,
        )

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset that this model is fitting."""
        return self.encoder.data_space_in.name

    @classmethod
    def from_template(
        cls,
        *,
        channel_names: list[str],
        data_space_in: DataSpace,
        dataset: str,
        decoder: DictConfig,
        encoder: DictConfig,
        template: EncodeProcessDecode,
    ) -> "EncoderStage":
        """Create an EncoderStage from an existing EncodeProcessDecode template."""
        return cls(
            channel_names=channel_names,
            data_space_in=data_space_in,
            decoder=decoder,
            encoder=encoder,
            hemisphere=template.hemisphere,
            input_spaces=[s.to_dict() for s in template.input_spaces],
            latent_space=template.encoders[0].data_space_out.shape,
            latitudes_fn=template.latitudes_fn,
            longitudes_fn=template.longitudes_fn,
            lr_scheduler=copy.deepcopy(template.lr_scheduler_cfg),
            n_forecast_steps=template.n_forecast_steps,
            n_history_steps=template.n_history_steps,
            name=f"{dataset}_encoder".replace("-", "_"),
            optimizer=copy.deepcopy(template.optimizer_cfg),
            output_space=template.output_space.to_dict(),
            scheduler=copy.deepcopy(template.scheduler_cfg),
            loss=copy.deepcopy(template.loss_cfg),
            metrics=copy.deepcopy(template.metrics),
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - squeeze time dimension to get `NCHW` (batch, n_input_channels, H_input, W_input)
        - encode into latent space `NCHW` (batch, n_latent_channels_total, H_latent, W_latent)
        - decode to target space via rollout() so restrict_range applies `NTCHW` (batch, 1, n_output_channels, H_output, W_output),
        """
        latent = self.encoder(inputs["target"].squeeze(1)).unsqueeze(1)
        # Ignore persistence as we want to learn the best autoencoder
        return self.decoder.rollout(latent, None)

    def process_batch(self, batch: dict[str, TensorNTCHW]) -> dict[str, TensorNTCHW]:
        """Extract only the first time step of only the relevant batch element.

        This is because we want the encoder to learn an NCHW -> NCHW mapping and to
        ensure that each input date is only used once per epoch.
        """
        return {"target": batch[self.dataset_name][:, 0, :, :, :].unsqueeze(1)}
