"""Combine the current config with the config that a checkpoint was trained with."""

import logging
import reprlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_checkpoint_config(checkpoint_path: Path) -> DictConfig | None:
    """Load the config saved alongside a checkpoint, migrating legacy settings.

    Args:
        checkpoint_path: Path to a checkpoint file in a run's ``checkpoints``
            directory. The config is read from the run's ``files/model_config.yaml``.

    Returns:
        The checkpoint config, or None (with a warning) if it could not be loaded.

    """
    config_path = checkpoint_path.parent.parent / "files" / "model_config.yaml"
    try:
        ckpt_config = DictConfig(OmegaConf.load(config_path))
    except (NotADirectoryError, FileNotFoundError):
        log.warning(
            "Could not load the checkpoint configuration from %s, so the values "
            "from the provided config file will be used instead. This may cause "
            "problems if the values differ from those used during training.",
            config_path,
        )
        return None
    log.debug("Loaded checkpoint configuration from %s.", config_path)

    # Checkpoints from before 'predict' was split into 'variables' and 'window'
    # are translated, since the current defaults would otherwise be used.
    if (predict := ckpt_config.pop("predict", None)) is not None:
        log.warning(
            "Checkpoint configuration %s uses the legacy 'predict' key. This "
            "has been translated into 'variables' and 'window' settings.",
            config_path,
        )
        # Legacy checkpoints used every variable from every dataset group as input.
        # A missing target variable list selected every variable in the target
        # group, which is expressed as an empty list.
        target = predict["target"]
        if "variables" not in ckpt_config:
            ckpt_config["variables"] = {
                "input": {},
                "target": {target["group_name"]: target.get("variables", [])},
            }
        if "window" not in ckpt_config:
            ckpt_config["window"] = {
                "n_forecast_steps": predict.get("n_forecast_steps", 1),
                "n_history_steps": predict.get("n_history_steps", 1),
            }
    return ckpt_config


def merge_checkpoint_config(config: DictConfig, ckpt_config: DictConfig) -> DictConfig:
    """Combine the current config with the config that a checkpoint was trained with.

    The current config takes precedence, except for the "model", "variables" and
    "window" sections, which describe the trained model. The "model" and "window"
    checkpoint configs are merged over the current ones, apart from
    "window.batch_size", while the "variables" checkpoint config replaces the
    current one entirely. A warning lists any current values that are replaced.

    Args:
        config: The current config.
        ckpt_config: The config that the checkpoint was trained with.

    Returns:
        The combined config.

    """
    combined = DictConfig(OmegaConf.merge(ckpt_config, config))
    for key in ("model", "window"):
        combined[key] = OmegaConf.merge(combined.get(key, {}), ckpt_config.get(key, {}))
    # We must use the same variables that the checkpoint was trained with
    if "variables" in ckpt_config:
        combined["variables"] = ckpt_config["variables"]
    # Batch size does not affect the trained model, so this can be overridden
    if "batch_size" in config.get("window", {}):
        combined["window"]["batch_size"] = config["window"]["batch_size"]

    _warn_about_replaced_values(config, combined)
    return combined


def _leaf_values(cfg: object, prefix: str) -> dict[str, Any]:
    """Flatten a config section to dotted keys, treating lists as single values."""
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg, Mapping):
        return {prefix: cfg}
    return {
        key: value
        for name, item in cfg.items()
        for key, value in _leaf_values(item, f"{prefix}.{name}").items()
    }


def _warn_about_replaced_values(config: DictConfig, combined: DictConfig) -> None:
    """Warn about any current config values that are not used in the combined config."""
    # If the models differ then there is no point in listing each difference
    sections = ["model", "variables", "window"]
    configured_model = OmegaConf.select(config, "model.name")
    combined_model = OmegaConf.select(combined, "model.name")
    if configured_model and configured_model != combined_model:
        log.warning(
            "Using the '%s' model from the checkpoint rather than the configured "
            "'%s' model.",
            combined_model,
            configured_model,
        )
        sections.remove("model")

    replaced: dict[str, tuple[Any, Any]] = {}
    for section in sections:
        used = _leaf_values(combined.get(section, {}), section)
        replaced |= {
            key: (value, used.get(key, "unset"))
            for key, value in _leaf_values(config.get(section, {}), section).items()
            if value != used.get(key, "unset")
        }
    if replaced:
        log.warning(
            "Using settings from the checkpoint rather than the current config: %s.",
            "; ".join(
                f"{key}={reprlib.repr(used)} (configured as {reprlib.repr(configured)})"
                for key, (configured, used) in replaced.items()
            ),
        )
