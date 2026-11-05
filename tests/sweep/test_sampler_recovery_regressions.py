"""Sampler persistence regression tests for corrupted files and interrupted writes."""

import os
import pickle
from pathlib import Path
from typing import BinaryIO
from unittest.mock import MagicMock

import pytest
from filelock import FileLock, Timeout
from optuna import create_study
from optuna.samplers import RandomSampler, TPESampler

from cryocast.sweep.sampler_store import SamplerStore


class TestSamplerFileRecovery:
    """A damaged file should not prevent a fresh seeded sampler from starting."""

    @pytest.mark.parametrize(
        "bytes_on_disk",
        [b"", b"garbage pickle bytes", b"\x80"],
        ids=["empty", "invalid", "truncated-protocol"],
    )
    def test_corrupt_state_is_replaced_only_after_a_successful_lock_cycle(
        self, tmp_path: Path, bytes_on_disk: bytes
    ) -> None:
        path = tmp_path / "sampler.pkl"
        path.write_bytes(bytes_on_disk)
        store = SamplerStore(tmp_path, "random", seed=4521)

        # load does not mutate a damaged file or silently rewrite it.
        loaded = store.load()
        expected = RandomSampler(seed=4521)
        assert isinstance(loaded, RandomSampler)
        assert pickle.dumps(loaded) == pickle.dumps(expected)
        assert path.read_bytes() == bytes_on_disk

        study = create_study(sampler=RandomSampler(seed=9000))
        with store.lock(study):
            assert isinstance(study.sampler, RandomSampler)
            assert pickle.dumps(study.sampler) == pickle.dumps(expected)

        with path.open("rb") as file:
            persisted = pickle.load(file)  # noqa: S301
        assert isinstance(persisted, RandomSampler)
        assert pickle.dumps(persisted) == pickle.dumps(expected)

    def test_interrupted_mutation_does_not_write_half_updated_sampler(
        self, tmp_path: Path
    ) -> None:
        store = SamplerStore(tmp_path, "random", seed=91)
        study = create_study(sampler=RandomSampler(seed=92))
        with store.lock(study):
            sampler = study.sampler
            assert isinstance(sampler, RandomSampler)
            _ = sampler._rng.rng.random()

        path = tmp_path / "sampler.pkl"
        before = path.read_bytes()

        def interrupt() -> None:
            with store.lock(study):
                sampler = study.sampler
                assert isinstance(sampler, RandomSampler)
                _ = sampler._rng.rng.random()
                msg = "trial interrupted"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="trial interrupted"):
            interrupt()

        assert path.read_bytes() == before
        # The lock must have been released after the exception.
        with store.lock(study):
            assert isinstance(study.sampler, RandomSampler)
        assert path.read_bytes() == before

    def test_exception_during_first_lock_does_not_create_a_sampler_file(
        self, tmp_path: Path
    ) -> None:
        store = SamplerStore(tmp_path, "random", seed=31)
        study = create_study(sampler=RandomSampler(seed=999))
        msg = "failed suggestion"
        with pytest.raises(RuntimeError, match="failed suggestion"), store.lock(study):
            raise RuntimeError(msg)

        assert not (tmp_path / "sampler.pkl").exists()
        with store.lock(study):
            assert isinstance(study.sampler, RandomSampler)
        assert (tmp_path / "sampler.pkl").is_file()

    def test_new_process_style_store_restores_advanced_rng(
        self, tmp_path: Path
    ) -> None:
        store = SamplerStore(tmp_path, "random", seed=137)
        original = create_study(sampler=RandomSampler(seed=900))
        with store.lock(original):
            sampler = original.sampler
            assert isinstance(sampler, RandomSampler)
            first = sampler._rng.rng.random()

        reload_store = SamplerStore(tmp_path, "random", seed=999)
        fresh = create_study(sampler=RandomSampler(seed=999))
        with reload_store.lock(fresh):
            sampler = fresh.sampler
            assert isinstance(sampler, RandomSampler)
            second = sampler._rng.rng.random()

        reference = RandomSampler(seed=137)
        assert first == reference._rng.rng.random()
        assert second == reference._rng.rng.random()

    @pytest.mark.parametrize(
        ("method", "initial_contents"),
        [("load", None), ("lock", b"persisted sampler placeholder")],
        ids=["read-timeout", "transaction-timeout"],
    )
    def test_external_lock_times_out_without_corrupting_sampler_state(
        self, tmp_path: Path, method: str, initial_contents: bytes | None
    ) -> None:
        path = tmp_path / "sampler.pkl"
        if initial_contents is not None:
            path.write_bytes(initial_contents)
        store = SamplerStore(tmp_path, "random", seed=31, timeout=0.02)
        held = FileLock(str(tmp_path / "sampler.pkl.lock"))

        def attempt() -> None:
            if method == "load":
                store.load()
            else:
                with store.lock(create_study(sampler=RandomSampler(seed=1))):
                    pytest.fail("Should never acquire an externally held lock")

        with held, pytest.raises(Timeout):
            attempt()
        if initial_contents is None:
            assert not path.exists()
        else:
            assert path.read_bytes() == initial_contents


class TestSamplerAlgorithms:
    """All supported named samplers are constructible without a W&B service."""

    @pytest.mark.parametrize("name", ["random", "tpe"])
    def test_named_sampler_can_suggest_and_complete_a_trial(
        self, tmp_path: Path, name: str
    ) -> None:
        store = SamplerStore(tmp_path, name, seed=91)
        sampler = store.temporary()
        assert isinstance(sampler, (RandomSampler, TPESampler))
        study = create_study(sampler=sampler)
        trial = study.ask()
        value = trial.suggest_float("learning_rate", 0.001, 0.01, log=True)
        study.tell(trial, value)
        assert 0.001 <= value <= 0.01
        assert study.best_value == value

    def test_persisted_sampler_is_still_the_requested_algorithm_after_reload(
        self, tmp_path: Path
    ) -> None:
        store = SamplerStore(tmp_path, "tpe", seed=91)
        first = create_study(sampler=RandomSampler(seed=900))
        with store.lock(first):
            assert isinstance(first.sampler, TPESampler)

        second = create_study(sampler=RandomSampler(seed=901))
        with SamplerStore(tmp_path, "tpe", seed=93).lock(second):
            assert isinstance(second.sampler, TPESampler)


@pytest.mark.parametrize("existing_state", [False, True])
@pytest.mark.parametrize("failed_stage", ["serialise", "sync", "replace"])
def test_failed_sampler_commit_preserves_previous_state_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
    *,
    existing_state: bool,
) -> None:
    """A failed disk commit must preserve the last complete sampler and allow retry."""
    store = SamplerStore(tmp_path, "random", seed=137)
    study = create_study(sampler=RandomSampler(seed=900))
    reference = RandomSampler(seed=137)
    if existing_state:
        with store.lock(study):
            sampler = study.sampler
            assert isinstance(sampler, RandomSampler)
            assert sampler._rng.rng.random() == reference._rng.rng.random()
    path = tmp_path / "sampler.pkl"
    previous = path.read_bytes() if existing_state else None
    error = OSError("sampler commit interrupted")

    def partial_dump(_sampler: object, file: BinaryIO) -> None:
        file.write(b"partial pickle")
        raise error

    def advance_sampler() -> None:
        with store.lock(study):
            sampler = study.sampler
            assert isinstance(sampler, RandomSampler)
            _ = sampler._rng.rng.random()

    with monkeypatch.context() as patch:
        if failed_stage == "serialise":
            patch.setattr(pickle, "dump", partial_dump)
        elif failed_stage == "sync":
            patch.setattr(os, "fsync", MagicMock(side_effect=error))
        else:
            patch.setattr(Path, "replace", MagicMock(side_effect=error))
        with pytest.raises(OSError, match="sampler commit interrupted") as caught:
            advance_sampler()
        assert caught.value is error

    assert (path.read_bytes() if path.exists() else None) == previous
    expected_files = {"sampler.pkl.lock"}
    if existing_state:
        expected_files.add("sampler.pkl")
    assert {file.name for file in tmp_path.iterdir()} == expected_files

    with store.lock(study):
        sampler = study.sampler
        assert isinstance(sampler, RandomSampler)
        assert sampler._rng.rng.random() == reference._rng.rng.random()
    assert isinstance(store.load(), RandomSampler)
    assert {file.name for file in tmp_path.iterdir()} == {
        "sampler.pkl",
        "sampler.pkl.lock",
    }
