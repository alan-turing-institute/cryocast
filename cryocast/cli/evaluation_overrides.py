"""Apply safe evaluation-only model overrides and expose ignored requests.

A checkpoint's model architecture and learned weights remain authoritative.
The diffusion sampling schedule is the one exception: changing DDPM/DDIM
sampling does not modify any trained network weights.
"""

import logging
from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf

from cryocast.model_service import ModelService
from cryocast.models.processors import DiffusionProcessor

from .hydra import HydraInvocation

log = logging.getLogger(__name__)

_SAMPLER_PATHS = frozenset({"model.processor.ddim_steps", "model.processor.eta"})
_MISSING = object()
_MAX_REPORTED_PATHS = 5


def _requested_model_paths(overrides: tuple[str, ...]) -> dict[str, bool]:
    """Get explicit model keys and whether each represents a delete operation.

    Values come from the already-composed Hydra configuration rather than
    parsing override expressions, so quoted/list interpolation stays intact.
    """
    requested: dict[str, bool] = {}
    for override in overrides:
        # Hydra accepts model.foo=bar, +model.foo=bar, ++model.foo=bar
        # and ~model.foo (deletion). Group selections use model=variant.
        key = override.partition("=")[0].lstrip("+~").split("@", maxsplit=1)[0]
        if key == "model" or key.startswith("model."):
            requested[key] = override.startswith("~")
    return requested


def _different_model_fields(
    requested: object, actual: object, path: str = "model"
) -> list[str]:
    """List leaf paths that differ between two model config trees."""
    if isinstance(requested, Mapping) and isinstance(actual, Mapping):
        fields: list[str] = []
        for name in sorted(requested.keys() | actual.keys()):
            fields.extend(
                _different_model_fields(
                    requested.get(name, _MISSING),
                    actual.get(name, _MISSING),
                    f"{path}.{name}",
                )
            )
        return fields
    return [] if requested == actual else [path]


def apply_evaluation_model_overrides(
    service: ModelService,
    requested_config: DictConfig,
    invocation: HydraInvocation | None,
) -> None:
    """Apply explicitly requested safe sampler settings, warn about ignored model edits.

    Only explicit Hydra model paths can change a checkpoint model. The
    configuration selected by --config-name may describe another architecture,
    but is not used to rebuild a saved checkpoint model.
    """
    if invocation is None:
        return

    paths = _requested_model_paths(invocation.overrides)
    if not paths and invocation.config_name in (None, "sample"):
        return

    # The saved model and sidecar config were already loaded and validated by
    # ModelService.from_checkpoint. Never swap architecture or mutate weights.
    processor = getattr(service.model, "processor", None)
    supported_paths = {
        path
        for path in _SAMPLER_PATHS
        if path in paths
        and not paths[path]
        and OmegaConf.select(requested_config, path, default=_MISSING) is not _MISSING
        and isinstance(processor, DiffusionProcessor)
    }

    if supported_paths and isinstance(processor, DiffusionProcessor):
        ddim_steps = (
            OmegaConf.select(requested_config, "model.processor.ddim_steps")
            if "model.processor.ddim_steps" in supported_paths
            else processor.ddim_steps
        )
        eta = (
            OmegaConf.select(requested_config, "model.processor.eta")
            if "model.processor.eta" in supported_paths
            else processor.eta
        )
        # Validation remains in the processor. Invalid settings fail rather
        # than quietly falling back to the checkpoint's sampler.
        processor.set_sampler(ddim_steps=ddim_steps, eta=eta)
        for path in sorted(supported_paths):
            value = OmegaConf.select(requested_config, path)
            OmegaConf.update(service.config, path, value, force_add=True)
        log.info(
            "Applied evaluation-only diffusion sampler overrides: %s.",
            ", ".join(sorted(supported_paths)),
        )

    for path, is_delete in paths.items():
        if path in supported_paths:
            continue
        requested_value = OmegaConf.select(requested_config, path, default=_MISSING)
        effective_value = OmegaConf.select(service.config, path, default=_MISSING)
        if is_delete or requested_value != effective_value:
            log.warning(
                "Ignoring evaluation override %r: the checkpoint's model "
                "configuration and trained weights take precedence. Only "
                "explicit DiffusionProcessor ddim_steps/eta sampler overrides "
                "are supported without rebuilding the model.",
                path,
            )

    # A selected config file may define an entirely different model without
    # any model.* CLI override. Report such differences, but exclude the
    # sampler leaves already applied above.
    if (
        invocation.config_name not in (None, "sample")
        and "model" in requested_config
        and "model" in service.config
    ):
        requested_model = OmegaConf.to_container(requested_config.model, resolve=False)
        effective_model = OmegaConf.to_container(service.config.model, resolve=False)
        changed = [
            path
            for path in _different_model_fields(requested_model, effective_model)
            if path not in supported_paths
        ]
        if changed:
            sample = ", ".join(changed[:_MAX_REPORTED_PATHS])
            extra = (
                f" (+{len(changed) - _MAX_REPORTED_PATHS} more)"
                if len(changed) > _MAX_REPORTED_PATHS
                else ""
            )
            log.warning(
                "The model configuration in --config-name=%s differs from "
                "the checkpoint at %s%s; those changes will not replace "
                "the checkpoint model. Specify model.processor.ddim_steps/eta "
                "explicitly for evaluation-only sampler changes.",
                invocation.config_name,
                sample,
                extra,
            )
