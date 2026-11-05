import logging
import os
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

from filelock import FileLock
from optuna import Study
from optuna.samplers import (
    BaseSampler,
    GPSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

log = logging.getLogger(__name__)


class SamplerStore:
    """Reads and writes a pickled Optuna sampler to disk.

    Guards against concurrent access to the sampler by gating it behind a file lock.
    """

    sampler_map: ClassVar[dict[str, type[BaseSampler]]] = {
        "gp": GPSampler,
        "qmc": QMCSampler,
        "random": RandomSampler,
        "tpe": TPESampler,
    }

    def __init__(
        self, study_path: Path, sampler_cls: str, seed: int, *, timeout: float = 60.0
    ) -> None:
        """Initialise a SamplerStore at `study_path` for the named sampler type."""
        self._sampler_path = study_path / "sampler.pkl"
        self._sampler_cls = sampler_cls
        self._seed = seed
        self._lock = FileLock(str(study_path / "sampler.pkl.lock"), timeout=timeout)

    def load(self) -> BaseSampler:
        """Wait until a lock is avalable then read the sampler."""
        with self._lock:
            return self._read()

    def _read(self) -> BaseSampler:
        """Read the sampler from disk, or construct a fresh one if none exists yet."""
        try:
            with self._sampler_path.open("rb") as f_sampler:
                return pickle.load(f_sampler)  # noqa: S301
        except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError):
            log.debug("Sampler could not be loaded from %s.", self._sampler_path)
            return self.temporary()

    @contextmanager
    def lock(self, study: Study) -> Generator[None]:
        """Allow a study to perform sampler operations atomically.

        Reloads the latest sampler state from disk onto `study`, runs the wrapped block,
        then persists the resulting state back to disk. This ensures that a single
        read-mutate-write cycle is performed atomically.
        """
        with self._lock:
            study.sampler = self._read()
            yield
            self._write(study.sampler)

    def _write(self, sampler: BaseSampler) -> None:
        """Replace the saved state only after a complete write in the same directory."""
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=self._sampler_path.parent,
                prefix=f".{self._sampler_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f_sampler:
                temporary_path = Path(f_sampler.name)
                pickle.dump(sampler, f_sampler)
                f_sampler.flush()
                os.fsync(f_sampler.fileno())
            temporary_path.replace(self._sampler_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def temporary(self) -> BaseSampler:
        """Construct an in-memory sampler of the configured class."""
        sampler_cls = self.sampler_map.get(self._sampler_cls)
        if sampler_cls is None:
            msg = (
                f"Unknown sampler '{self._sampler_cls}', expected one of "
                f"{self.sampler_map.keys()}"
            )
            raise ValueError(msg) from None
        return sampler_cls(seed=self._seed)  # type: ignore[call-arg]
