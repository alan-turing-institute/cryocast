import copy
import logging
from typing import Any, ClassVar

from omegaconf import DictConfig

from cryocast.models import EncodeProcessDecode
from cryocast.types import RolloutStrategy

from .decoder_stage import DecoderStage

logger = logging.getLogger(__name__)


class ProcessorStage(EncodeProcessDecode):
    # Parameters that should be excluded from hyperparameter logging
    ignored_hparams: ClassVar[frozenset[str]] = EncodeProcessDecode.ignored_hparams | {
        "decoder_model",
    }

    def __init__(
        self,
        processor: DictConfig,
        decoder_model: DecoderStage,
        mask_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a ProcessorStage with a trainable processor.

        - copy encoders from DecoderStage and freeze their parameters
        - copy the decoder and finaliser from DecoderStage and freeze their parameters
        - copy target_variable_indices from checkpoint or from DecoderStage
        """
        kwargs.setdefault(
            "target_variable_indices", decoder_model.target_variable_indices
        )
        super().__init__(
            encoders=[
                copy.deepcopy(encoder).freeze() for encoder in decoder_model.encoders
            ],
            processor=processor,
            decoder=copy.deepcopy(decoder_model.decoder).freeze(),
            finaliser=copy.deepcopy(decoder_model.finaliser).freeze(),
            mask_dir=mask_dir,
            **kwargs,
        )

    @classmethod
    def from_template(
        cls,
        *,
        processor: DictConfig,
        decoder_model: DecoderStage,
        rollout_strategy: RolloutStrategy,
        mask_dir: str | None = None,
    ) -> "ProcessorStage":
        """Create a ProcessorStage from a trained DecoderStage.

        The encoders, decoder and training settings come from the DecoderStage. Give the
        rollout strategy of the full model, so that the processor trains in the same
        mode as the full model is finetuned and evaluated.

        Args:
            processor: Config for the trainable processor.
            decoder_model: Trained DecoderStage that supplies the frozen encoders and
                decoder.
            rollout_strategy: The rollout strategy of the full model.
            mask_dir: Directory that contains the masks for the finaliser, if any.

        """
        return cls(
            decoder_model=decoder_model,
            hemisphere=decoder_model.hemisphere,
            input_spaces=[s.to_dict() for s in decoder_model.input_spaces],
            loss=copy.deepcopy(decoder_model.loss_cfg),
            mask_dir=mask_dir,
            lr_scheduler=copy.deepcopy(decoder_model.lr_scheduler_cfg),
            n_forecast_steps=decoder_model.n_forecast_steps,
            n_history_steps=decoder_model.n_history_steps,
            name=f"processor_{decoder_model.n_history_steps}_to_{decoder_model.n_forecast_steps}",
            optimizer=copy.deepcopy(decoder_model.optimizer_cfg),
            output_space=decoder_model.output_space.to_dict(),
            processor=processor,
            rollout_space=rollout_strategy.space,
            skip_anchor=rollout_strategy.anchor,
            relaxation_tau=rollout_strategy.relaxation_tau,
            scheduler=copy.deepcopy(decoder_model.scheduler_cfg),
            metrics=copy.deepcopy(list(decoder_model.metric_cfgs.values())),
        )
