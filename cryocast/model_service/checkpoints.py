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
from torch.serialization import get_unsafe_globals_in_checkpoint

from cryocast.exceptions import (
    CheckpointUpgradeError,
    OutdatedCheckpointError,
    UntrustedCheckpointError,
)
from cryocast.types import DataSpace
from cryocast.utils import to_plain_types

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointFile:
    """A checkpoint file that has been verified to exist, with an absolute path."""

    path: Path

    def __init__(self, path: str | Path) -> None:
        """Resolve the checkpoint path and check that the file exists.

        Raises:
            FileNotFoundError: If there is no file at the given path.

        """
        object.__setattr__(self, "path", Path(path).resolve())
        if not self.path.is_file():
            msg = f"Could not find checkpoint file {self.path}."
            raise FileNotFoundError(msg)

    @classmethod
    def find_last(cls, checkpoint_dir: Path) -> "CheckpointFile":
        """Find the checkpoint to resume training from in a directory of checkpoints.

        Args:
            checkpoint_dir: A directory that may contain checkpoints.

        Lightning saves ``last-v1.ckpt``, ``last-v2.ckpt`` and so on when ``last.ckpt``
        already exists, so the most recently modified ``last*.ckpt`` file is chosen,
        falling back to the highest version number if modification times are equal.

        Returns:
            The most recent ``last*.ckpt`` file in the directory if it exists.

        Raises:
            FileNotFoundError: If the directory has no ``last*.ckpt`` file.

        """

        def recency(path: Path) -> tuple[int, int]:
            version = re.fullmatch(r"last-v(\d+)\.ckpt", path.name)
            return path.stat().st_mtime_ns, int(version[1]) if version else 0

        if not (matches := sorted(checkpoint_dir.glob("last*.ckpt"), key=recency)):
            msg = f"No resumable checkpoint (last*.ckpt) found in {checkpoint_dir}."
            raise FileNotFoundError(msg)
        log.debug("Found checkpoint at %s.", matches[-1])
        return cls(matches[-1])

    @property
    def config_path(self) -> Path:
        """The config saved alongside the checkpoint, in the files directory."""
        return self.path.parent.parent / "files" / "model_config.yaml"

    def _flatten(self, cfg: object, prefix: str) -> dict[str, Any]:
        """Flatten a config section to dotted keys, treating lists as single values."""
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=False)
        if not isinstance(cfg, Mapping):
            return {prefix: cfg}
        return {
            key: value
            for name, item in cfg.items()
            for key, value in self._flatten(item, f"{prefix}.{name}").items()
        }

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

    def load(self) -> dict[str, Any]:
        """Load the checkpoint, allowing only plain Python types and tensors.

        Raises:
            OutdatedCheckpointError: If the checkpoint contains any other objects.

        """
        with self.suggest_upgrade_on_failure():
            return torch.load(self.path, map_location="cpu", weights_only=True)

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
            used = self._flatten(combined.get(section, {}), section)
            replaced |= {
                key: (value, used.get(key, "unset"))
                for key, value in self._flatten(
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
                "code. Upgrading it runs any code embedded in the file, so only do so "
                "if you trust its source, with 'cryocast checkpoint upgrade --trust "
                f"{self.path}'."
            )
            raise OutdatedCheckpointError(msg) from exc


class LegacyCheckpointFile:
    """A checkpoint file saved by an older version of this code.

    Upgrading rewrites the checkpoint, and the config saved alongside it, into the
    current format so that they can be loaded safely. Only files that need upgrading are
    rewritten, so upgrading twice changes nothing. Each rewritten file is first backed
    up to ``<name>.bak``, unless an earlier upgrade has already done so. This means that
    the backup always holds the original file.

    Reading a checkpoint that cannot be loaded safely runs the full pickle machinery, so
    a malicious file can execute arbitrary code. This is only done when the caller
    explicitly trusts the checkpoint.

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
    def _rename(cls, value: object, renamed: set[str]) -> object:
        """Recursively replace legacy class names with cryocast.X in strings.

        Args:
            value: The value to rename.
            renamed: Collects each string that was renamed.

        """
        if isinstance(value, str):
            if cls._PACKAGE_PATTERN.search(value):
                renamed.add(value)
            return cls._PACKAGE_PATTERN.sub("cryocast", value)
        if isinstance(value, Mapping):
            return {
                cls._rename(key, renamed): cls._rename(item, renamed)
                for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return type(value)(cls._rename(item, renamed) for item in value)
        return value

    @staticmethod
    def _replace(path: Path, write: Callable[[Path], None]) -> None:
        """Replace a file, backing up the original unless it has already been.

        Args:
            path: The file to replace.
            write: Writes the replacement to the given path, raising if it cannot, in
                which case the original file is left unchanged.

        """
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            write(tmp_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        if not (backup_path := path.with_name(f"{path.name}.bak")).exists():
            shutil.move(path, backup_path)
        shutil.move(tmp_path, path)
        log.info("Upgraded %s, keeping the original at %s.", path, backup_path)

    def upgrade(self, *, trusted: bool = False) -> CheckpointFile:  # noqa: C901
        """Upgrade the checkpoint and its config to the current format.

        The checkpoint is rewritten to reference ``cryocast`` classes and to contain
        only plain Python types, so that it can be loaded with ``weights_only=True``.
        The config is rewritten to reference ``cryocast`` and to replace the legacy
        'predict' settings with 'variables' and 'window' ones.

        Args:
            trusted: Whether the checkpoint comes from a trusted source. A checkpoint
                that cannot be loaded safely can only be upgraded by unpickling it
                without restrictions, which can execute arbitrary code, so this is
                refused unless the checkpoint is trusted.

        Returns:
            The upgraded checkpoint file.

        Raises:
            UntrustedCheckpointError: If the checkpoint cannot be loaded safely and is
                not trusted.
            CheckpointUpgradeError: If the upgraded checkpoint would still not load
                safely, in which case no files are changed.

        """
        checkpoint_path = self.checkpoint_file.path
        config_path = self.checkpoint_file.config_path

        # Upgrade the checkpoint if it cannot be loaded safely or if it references the
        # old package name, which may be stored as a plain string such as a '_target_'
        if unsafe := get_unsafe_globals_in_checkpoint(checkpoint_path):
            if not trusted:
                msg = (
                    f"Checkpoint {checkpoint_path} references "
                    f"{', '.join(sorted(unsafe))}, so upgrading it means unpickling it "
                    "without restrictions, which can execute arbitrary code. If you "
                    "trust its source, upgrade it with 'cryocast checkpoint upgrade "
                    f"--trust {checkpoint_path}'."
                )
                raise UntrustedCheckpointError(msg)
            ckpt_state = torch.load(
                checkpoint_path,
                map_location="cpu",
                pickle_module=SimpleNamespace(
                    **{**vars(pickle), "Unpickler": self._Unpickler}
                ),
                weights_only=False,  # this is an older checkpoint that the user chose
            )
        else:
            ckpt_state = self.checkpoint_file.load()
        renamed: set[str] = set()
        ckpt_state = self._rename(to_plain_types(ckpt_state), renamed)

        if unsafe or renamed:

            def write_upgraded_checkpoint(tmp_path: Path) -> None:
                torch.save(ckpt_state, tmp_path)
                if remaining := get_unsafe_globals_in_checkpoint(tmp_path):
                    msg = (
                        f"Could not upgrade checkpoint {checkpoint_path}, as it contains "
                        "objects that cannot be converted to plain types: "
                        f"{', '.join(sorted(remaining))}. The original file is unchanged."
                    )
                    raise CheckpointUpgradeError(msg)

            self._replace(checkpoint_path, write_upgraded_checkpoint)
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
            self._replace(config_path, lambda tmp_pth: OmegaConf.save(config, tmp_pth))
        else:
            log.info("Configuration %s is already up to date.", config_path)
        return self.checkpoint_file
