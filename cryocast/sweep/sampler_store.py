import logging
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
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
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
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
            with self._sampler_path.open("wb") as f_sampler:
                pickle.dump(study.sampler, f_sampler)

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
