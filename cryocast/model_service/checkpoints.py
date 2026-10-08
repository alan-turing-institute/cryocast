"""Find, load and upgrade model checkpoints and the configs they were trained with."""

import logging
import pickle
import re
import reprlib
import shutil
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from omegaconf import DictConfig, OmegaConf

from cryocast.exceptions import OutdatedCheckpointError
from cryocast.types import DataSpace
from cryocast.utils import to_plain_types

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointFile:
    """A checkpoint file that has been verified to exist, with an absolute path."""

    path: Path

    def __post_init__(self) -> None:
        """Resolve the checkpoint path and check that the file exists.

        Raises:
            FileNotFoundError: If there is no file at the given path.

        """
        object.__setattr__(self, "path", self.path.resolve())  # resolve path
        if not self.path.is_file():
            msg = f"Could not find checkpoint file {self.path}."
            raise FileNotFoundError(msg)

    @classmethod
    def _leaf_values(cls, cfg: object, prefix: str) -> dict[str, Any]:
        """Flatten a config section to dotted keys, treating lists as single values."""
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=False)
        if not isinstance(cfg, Mapping):
            return {prefix: cfg}
        return {
            key: value
            for name, item in cfg.items()
            for key, value in cls._leaf_values(item, f"{prefix}.{name}").items()
        }

    @classmethod
    def find_last(cls, checkpoint_dir: Path) -> "CheckpointFile":
        """Find the checkpoint to resume training from in a directory of checkpoints.

        Args:
            checkpoint_dir: A directory that may contain checkpoints.

        Returns:
            The ``last*.ckpt`` file in the directory if it exists.

        Raises:
            FileNotFoundError: If the directory has no ``last*.ckpt`` file.

        """
        if not (matches := sorted(checkpoint_dir.glob("last*.ckpt"))):
            msg = f"No resumable checkpoint (last*.ckpt) found in {checkpoint_dir}."
            raise FileNotFoundError(msg)
        log.debug("Found checkpoint at %s.", matches[-1])
        return cls(matches[-1])

    @property
    def config_path(self) -> Path:
        """The config saved alongside the checkpoint, in the files directory."""
        return self.path.parent.parent / "files" / "model_config.yaml"

    def load(self) -> dict[str, Any]:
        """Load the checkpoint, allowing only plain Python types and tensors.

        Raises:
            OutdatedCheckpointError: If the checkpoint contains any other objects.

        """
        with self.suggest_upgrade_on_failure():
            return torch.load(self.path, map_location="cpu", weights_only=True)

    def _load_config(self) -> DictConfig | None:
        """Load the config saved alongside the checkpoint.

        Returns:
            The checkpoint config, or None (with a warning) if it could not be loaded.

        Raises:
            OutdatedCheckpointError: If the config predates the current format.

        """
        try:
            ckpt_config = DictConfig(OmegaConf.load(self.config_path))
        except (NotADirectoryError, FileNotFoundError):
            log.warning(
                "Could not load the checkpoint configuration from %s, so the values "
                "from the provided config file will be used instead. This may cause "
                "problems if the values differ from those used during training.",
                self.config_path,
            )
            return None
        log.debug("Loaded checkpoint configuration from %s.", self.config_path)

        # Configs from before 'predict' was split into 'variables' and 'window' must
        # be upgraded, since the current defaults would otherwise be used.
        if "predict" in ckpt_config:
            msg = (
                f"Checkpoint configuration {self.config_path} uses the legacy "
                "'predict' key. Upgrade it with "
                f"'cryocast checkpoint upgrade {self.path}'."
            )
            raise OutdatedCheckpointError(msg)
        return ckpt_config

    def merge_config(self, config: DictConfig) -> DictConfig:
        """Combine the current config with the config that the checkpoint was trained with.

        The current config takes precedence, except for the "model", "variables" and
        "window" sections, which describe the trained model. The "model" and "window"
        checkpoint configs are merged over the current ones, apart from
        "window.batch_size", while the "variables" checkpoint config replaces the
        current one entirely. If the current config names a different model, the
        "model" checkpoint config also replaces the current one entirely. A warning
        lists any current values that are replaced.

        Args:
            config: The current config.

        Returns:
            The combined config, or the current config (with a warning) if there is no
            saved checkpoint config.

        """
        if (ckpt_config := self._load_config()) is None:
            return config

        combined = DictConfig(OmegaConf.merge(ckpt_config, config))
        for key in ("model", "window"):
            combined[key] = OmegaConf.merge(
                combined.get(key, {}), ckpt_config.get(key, {})
            )
        # We must use the same variables that the checkpoint was trained with
        if "variables" in ckpt_config:
            combined["variables"] = ckpt_config["variables"]
        # Batch size does not affect the trained model, so this can be overridden
        if "batch_size" in config.get("window", {}):
            combined["window"]["batch_size"] = config["window"]["batch_size"]

        # A different model must not inherit any of the configured model's settings,
        # so use the checkpoint model config outright and summarise this in one line
        # rather than listing every difference
        sections = ["model", "variables", "window"]
        configured_model = OmegaConf.select(config, "model.name")
        checkpoint_model = OmegaConf.select(ckpt_config, "model.name")
        if configured_model and configured_model != checkpoint_model:
            combined["model"] = ckpt_config.get("model", {})
            log.warning(
                "Using the '%s' model from the checkpoint rather than the configured "
                "'%s' model.",
                checkpoint_model,
                configured_model,
            )
            sections.remove("model")

        replaced: dict[str, tuple[Any, Any]] = {}
        for section in sections:
            used = self._leaf_values(combined.get(section, {}), section)
            replaced |= {
                key: (value, used.get(key, "unset"))
                for key, value in self._leaf_values(
                    config.get(section, {}), section
                ).items()
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

    @contextmanager
    def suggest_upgrade_on_failure(self) -> Generator[None]:
        """Suggest upgrading the checkpoint if it cannot be loaded safely.

        Raises:
            OutdatedCheckpointError: If loading the checkpoint fails because it contains
                objects other than plain Python types and tensors.

        """
        try:
            yield
        except pickle.UnpicklingError as exc:
            msg = (
                f"Checkpoint {self.path} contains objects that cannot be loaded "
                "safely, probably because it was saved by an older version of this "
                f"code. Upgrade it with 'cryocast checkpoint upgrade {self.path}'."
            )
            raise OutdatedCheckpointError(msg) from exc


class LegacyCheckpointFile:
    """A checkpoint file saved by an older version of this code.

    Upgrading rewrites the checkpoint, and the config saved alongside it, into the
    current format so that they can be loaded safely. Only files that need upgrading
    are rewritten, so upgrading twice changes nothing. Each rewritten file is first
    backed up to ``<name>.bak``, unless an earlier upgrade has already done so, so
    that the backup always holds the original file.
    """

    # Older checkpoints may reference this package by its previous name
    _PACKAGE: ClassVar[str] = "icenet_mp"
    _PACKAGE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(rf"\b{_PACKAGE}(?=\.)")

    class _LegacyDataSpace(DataSpace):
        """A DataSpace that can be unpickled from its legacy public attributes."""

        def __setstate__(self, state: dict[str, Any]) -> None:
            """Initialise from 'channels', 'name' and 'shape', with or without '_'."""
            super().__init__(
                **{key.removeprefix("_"): value for key, value in state.items()}
            )

    class _Unpickler(pickle.Unpickler):
        """Update legacy classes to the equivalent cryocast.X class."""

        def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
            if module.split(".", maxsplit=1)[0] == LegacyCheckpointFile._PACKAGE:
                module = "cryocast" + module.removeprefix(LegacyCheckpointFile._PACKAGE)
            cls = super().find_class(module, name)
            # DataSpace attributes have since been made private
            return LegacyCheckpointFile._LegacyDataSpace if cls is DataSpace else cls

    def __init__(self, checkpoint_file: CheckpointFile) -> None:
        """Wrap a checkpoint file that may need upgrading."""
        self.checkpoint_file = checkpoint_file

    @classmethod
    def _rename(cls, value: object) -> object:
        """Recursively replace legacy class names with cryocast.X in strings."""
        if isinstance(value, str):
            return cls._PACKAGE_PATTERN.sub("cryocast", value)
        if isinstance(value, Mapping):
            return {cls._rename(key): cls._rename(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(cls._rename(item) for item in value)
        return value

    @staticmethod
    def _replace(path: Path, write: Callable[[Path], None]) -> None:
        """Replace a file, backing up the original unless it has already been."""
        tmp_path = path.with_name(f"{path.name}.tmp")
        write(tmp_path)
        if not (backup_path := path.with_name(f"{path.name}.bak")).exists():
            shutil.move(path, backup_path)
        shutil.move(tmp_path, path)
        log.info("Upgraded %s, keeping the original at %s.", path, backup_path)

    def upgrade(self) -> CheckpointFile:
        """Upgrade the checkpoint and its config to the current format.

        The checkpoint is rewritten to reference ``cryocast`` classes and to contain
        only plain Python types, so that it can be loaded with ``weights_only=True``.
        The config is rewritten to reference ``cryocast`` and to replace the legacy
        'predict' settings with 'variables' and 'window' ones.

        Returns:
            The upgraded checkpoint file.

        """
        checkpoint_path = self.checkpoint_file.path
        config_path = self.checkpoint_file.config_path

        # Upgrade the checkpoint if it cannot be loaded safely. Every legacy checkpoint
        # contains enums or paths, so one that can be loaded safely is already current.
        if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                pickle_module=SimpleNamespace(
                    **{**vars(pickle), "Unpickler": self._Unpickler}
                ),
                weights_only=False,  # this is an older checkpoint that the user chose
            )
            self._replace(
                checkpoint_path,
                lambda path: torch.save(self._rename(to_plain_types(checkpoint)), path),
            )
        else:
            log.info("Checkpoint %s is already up to date.", checkpoint_path)

        # Upgrade the config if it references the old package name or uses 'predict'
        if not config_path.is_file():
            return self.checkpoint_file
        text = config_path.read_text()
        config = DictConfig(
            OmegaConf.create(self._PACKAGE_PATTERN.sub("cryocast", text))
        )
        if (predict := config.pop("predict", None)) is not None:
            # Legacy checkpoints used every variable from every dataset group as
            # input. A missing target variable list selected every variable in the
            # target group, which is expressed as an empty list.
            target = predict["target"]
            if "variables" not in config:
                config["variables"] = {
                    "input": {},
                    "target": {target["group_name"]: target.get("variables", [])},
                }
            if "window" not in config:
                config["window"] = {
                    "n_forecast_steps": predict.get("n_forecast_steps", 1),
                    "n_history_steps": predict.get("n_history_steps", 1),
                }
        if predict is not None or self._PACKAGE_PATTERN.search(text):
            self._replace(config_path, lambda path: OmegaConf.save(config, path))
        else:
            log.info("Configuration %s is already up to date.", config_path)
        return self.checkpoint_file
