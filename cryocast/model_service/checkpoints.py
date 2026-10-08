"""Find, load and upgrade model checkpoints and the configs they were trained with."""

import logging
import pickle
import re
import reprlib
import shutil
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf

from cryocast.exceptions import OutdatedCheckpointError
from cryocast.utils import to_plain_types

if TYPE_CHECKING:
    from cryocast.data import CommonDataModule
    from cryocast.models import BaseModel

log = logging.getLogger(__name__)

# Older checkpoints may reference the package by its previous name
LEGACY_PACKAGE = "icenet_mp"
LEGACY_PACKAGE_PATTERN = re.compile(rf"\b{LEGACY_PACKAGE}(?=\.)")


def find_checkpoint_file(checkpoint_dir: Path) -> Path:
    """Find the checkpoint to resume training from in a directory of checkpoints.

    Args:
        checkpoint_dir: A directory that may contain checkpoints.

    Returns:
        The path of the directory's ``last*.ckpt`` file.

    Raises:
        FileNotFoundError: If the directory has no ``last*.ckpt`` file.

    """
    if not (matches := sorted(checkpoint_dir.glob("last*.ckpt"))):
        msg = f"No resumable checkpoint (last*.ckpt) found in {checkpoint_dir}."
        raise FileNotFoundError(msg)
    log.debug("Found checkpoint at %s.", matches[-1])
    return matches[-1]


@contextmanager
def suggest_upgrade_on_failure(checkpoint_file: Path) -> Generator[None]:
    """Suggest upgrading a checkpoint if it cannot be loaded with weights_only=True.

    Raises:
        OutdatedCheckpointError: If loading the checkpoint fails because it contains
            objects other than plain Python types and tensors.

    """
    try:
        yield
    except pickle.UnpicklingError as exc:
        msg = (
            f"Checkpoint {checkpoint_file} contains objects that cannot be loaded "
            "safely, probably because it was saved by an older version of CryoCast "
            "or by icenet-mp. Upgrade it with "
            f"'cryocast checkpoint upgrade {checkpoint_file}'."
        )
        raise OutdatedCheckpointError(msg) from exc


def load_checkpoint(checkpoint_file: Path) -> dict[str, Any]:
    """Load a checkpoint, allowing only plain Python types and tensors.

    Raises:
        FileNotFoundError: If there is no file at the given path.
        OutdatedCheckpointError: If the checkpoint contains any other objects.

    """
    _verify_checkpoint_file(checkpoint_file)
    with suggest_upgrade_on_failure(checkpoint_file):
        return torch.load(checkpoint_file, map_location="cpu", weights_only=True)


def load_checkpoint_config(checkpoint_file: Path) -> DictConfig | None:
    """Load the config saved alongside a checkpoint, migrating legacy settings.

    Args:
        checkpoint_file: Path to a checkpoint file in a run's ``checkpoints``
            directory. The config is read from the run's ``files/model_config.yaml``.

    Returns:
        The checkpoint config, or None (with a warning) if it could not be loaded.

    """
    config_path = checkpoint_file.parent.parent / "files" / "model_config.yaml"
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

    # Warn about any current config values that are not used, summarising a
    # different model in one line rather than listing every difference
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

    return combined


def verify_model_matches_data(
    model: "BaseModel", data_module: "CommonDataModule"
) -> None:
    """Check that the data has the shape that a model was trained with.

    Inputs are matched by name, since models look up each input by its name.

    Raises:
        ValueError: If the input spaces, output space or window lengths differ.

    """
    model_inputs = {space.name: space for space in model.input_spaces}
    data_inputs = {space.name: space for space in data_module.input_spaces}
    mismatches = [
        f"input '{name}' is {model_inputs.get(name, 'missing')} in the model but "
        f"{data_inputs.get(name, 'missing')} in the data"
        for name in sorted(model_inputs.keys() | data_inputs.keys())
        if model_inputs.get(name) != data_inputs.get(name)
    ]
    if model.output_space != data_module.output_space:
        mismatches.append(
            f"output is {model.output_space} in the model but "
            f"{data_module.output_space} in the data"
        )
    mismatches.extend(
        f"{key} is {getattr(model, key)} in the model but "
        f"{getattr(data_module, key)} in the data"
        for key in ("n_history_steps", "n_forecast_steps")
        if getattr(model, key) != getattr(data_module, key)
    )
    if mismatches:
        msg = (
            "The checkpointed model does not match the configured data: "
            + "; ".join(mismatches)
            + ". Check the 'variables' and 'window' settings."
        )
        raise ValueError(msg)


def upgrade_checkpoint(checkpoint_file: Path) -> None:
    """Upgrade a checkpoint saved by an older version of this code.

    The checkpoint is rewritten so that it references ``cryocast`` classes and contains
    only plain Python types, so that it can be loaded with ``weights_only=True``. The
    ``files/model_config.yaml`` is also updated if necessary. Each file is backed up to
    ``<name>.bak`` first.

    Args:
        checkpoint_file: The path to the checkpoint to upgrade.

    Raises:
        FileExistsError: If a backup already exists, e.g. because the file has
            already been upgraded.

    """
    _verify_checkpoint_file(checkpoint_file)
    backup_path = _backup_path(checkpoint_file)

    # Load the checkpoint, resolving any icenet_mp classes as their cryocast versions
    pickle_module = SimpleNamespace(
        **{**vars(pickle), "Unpickler": _LegacyPackageUnpickler}
    )
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        pickle_module=pickle_module,
        weights_only=False,  # this is an older checkpoint that the user has chosen
    )

    # Write the upgraded checkpoint before replacing the original
    tmp_path = checkpoint_file.with_name(f"{checkpoint_file.name}.tmp")
    torch.save(_rename_legacy_package(to_plain_types(checkpoint)), tmp_path)
    shutil.move(checkpoint_file, backup_path)
    shutil.move(tmp_path, checkpoint_file)
    log.info("Upgraded %s, keeping the original at %s.", checkpoint_file, backup_path)

    # Update the run's model config if it references the legacy package
    config_path = checkpoint_file.parent.parent / "files" / "model_config.yaml"
    if config_path.is_file() and LEGACY_PACKAGE_PATTERN.search(
        text := config_path.read_text()
    ):
        backup_path = _backup_path(config_path)
        shutil.move(config_path, backup_path)
        config_path.write_text(LEGACY_PACKAGE_PATTERN.sub("cryocast", text))
        log.info("Upgraded %s, keeping the original at %s.", config_path, backup_path)


class _LegacyPackageUnpickler(pickle.Unpickler):
    """Resolve classes pickled as icenet_mp.X as cryocast.X."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
        if module.split(".", maxsplit=1)[0] == LEGACY_PACKAGE:
            module = "cryocast" + module.removeprefix(LEGACY_PACKAGE)
        return super().find_class(module, name)


def _verify_checkpoint_file(checkpoint_file: Path) -> None:
    """Check that a checkpoint file exists.

    Raises:
        FileNotFoundError: If there is no file at the given path.

    """
    if not checkpoint_file.is_file():
        msg = f"Could not find checkpoint file {checkpoint_file}."
        raise FileNotFoundError(msg)


def _backup_path(path: Path) -> Path:
    """Return a backup path for a file, refusing to overwrite an existing backup."""
    backup_path = path.with_name(f"{path.name}.bak")
    if backup_path.exists():
        msg = f"{backup_path} already exists: has {path} already been upgraded?"
        raise FileExistsError(msg)
    return backup_path


def _rename_legacy_package(value: object) -> object:
    """Recursively replace icenet_mp.X with cryocast.X in strings."""
    if isinstance(value, str):
        return LEGACY_PACKAGE_PATTERN.sub("cryocast", value)
    if isinstance(value, Mapping):
        return {
            _rename_legacy_package(key): _rename_legacy_package(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(_rename_legacy_package(item) for item in value)
    return value


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
