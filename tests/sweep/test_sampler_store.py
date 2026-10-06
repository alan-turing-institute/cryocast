import pickle
from pathlib import Path

import pytest
from filelock import FileLock, Timeout
from optuna import create_study
from optuna.samplers import RandomSampler

from icenet_mp.sweep.sampler_store import SamplerStore


def build_store(tmp_path: Path, *, seed: int = 0) -> SamplerStore:
    """Build a SamplerStore at `tmp_path` for a `random` sampler."""
    return SamplerStore(tmp_path, "random", seed)


class TestSamplerStoreLoad:
    """Tests for SamplerStore.load."""

    def test_constructs_a_fresh_sampler_when_no_pickle_exists(
        self, tmp_path: Path
    ) -> None:
        store = build_store(tmp_path)
        assert isinstance(store.load(), RandomSampler)

    def test_reads_an_existing_pickle(self, tmp_path: Path) -> None:
        (tmp_path / "sampler.pkl").write_bytes(pickle.dumps(RandomSampler(seed=99)))
        store = build_store(tmp_path)
        assert isinstance(store.load(), RandomSampler)

    def test_falls_back_to_a_fresh_sampler_on_a_truncated_pickle(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "sampler.pkl").write_bytes(b"")
        store = build_store(tmp_path)
        assert isinstance(store.load(), RandomSampler)

    def test_raises_for_an_unknown_sampler_cls(self, tmp_path: Path) -> None:
        store = SamplerStore(tmp_path, "not-a-sampler", seed=0)
        with pytest.raises(ValueError, match="Unknown sampler"):
            store.load()


class TestSamplerStoreTemporary:
    """Tests for SamplerStore.temporary."""

    def test_constructs_a_fresh_sampler_of_the_configured_class(
        self, tmp_path: Path
    ) -> None:
        store = build_store(tmp_path)
        assert isinstance(store.temporary(), RandomSampler)

    def test_ignores_an_existing_pickle(self, tmp_path: Path) -> None:
        """Unlike `load`, `temporary` never reads persisted sampler state."""
        persisted_bytes = pickle.dumps(RandomSampler(seed=123))
        (tmp_path / "sampler.pkl").write_bytes(persisted_bytes)
        store = build_store(tmp_path)
        assert pickle.dumps(store.temporary()) != persisted_bytes

    def test_raises_for_an_unknown_sampler_cls(self, tmp_path: Path) -> None:
        store = SamplerStore(tmp_path, "not-a-sampler", seed=0)
        with pytest.raises(ValueError, match="Unknown sampler"):
            store.temporary()


class TestSamplerStoreLock:
    """Tests for SamplerStore.lock."""

    def test_persists_the_studys_sampler_to_disk(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        study = create_study(sampler=RandomSampler(seed=0))
        with store.lock(study):
            pass
        assert (tmp_path / "sampler.pkl").exists()

    def test_reloads_the_latest_state_onto_the_study(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        persisted_bytes = pickle.dumps(RandomSampler(seed=123))
        (tmp_path / "sampler.pkl").write_bytes(persisted_bytes)

        study = create_study(sampler=RandomSampler(seed=0))
        with store.lock(study):
            reloaded_bytes = pickle.dumps(study.sampler)

        assert reloaded_bytes == persisted_bytes

    def test_does_not_duplicate_a_concurrent_trials_samples(
        self, tmp_path: Path
    ) -> None:
        """Concurrent trials must not clobber each other's sampler state.

        A slow process must re-read `sampler.pkl` before sampling, not overwrite a
        faster concurrent process's already-persisted RNG advance with its own stale
        in-memory copy.
        """
        store = build_store(tmp_path)
        slow_study = create_study(sampler=RandomSampler(seed=0))

        with store.lock(slow_study):
            pass  # the "slow" process asks early, without sampling yet

        concurrent_study = create_study(sampler=RandomSampler(seed=0))
        with store.lock(concurrent_study):
            concurrent_sampler = concurrent_study.sampler
            assert isinstance(concurrent_sampler, RandomSampler)
            concurrent_value = concurrent_sampler._rng.rng.random()

        with store.lock(slow_study):
            slow_sampler = slow_study.sampler
            assert isinstance(slow_sampler, RandomSampler)
            slow_value = slow_sampler._rng.rng.random()

        assert slow_value != concurrent_value


class TestSamplerStoreTimeout:
    """Tests for SamplerStore's lock timeout."""

    def test_raises_instead_of_blocking_forever_on_a_stuck_lock(
        self, tmp_path: Path
    ) -> None:
        stuck_lock = FileLock(str(tmp_path / "sampler.pkl.lock"))
        stuck_lock.acquire()
        try:
            store = SamplerStore(tmp_path, "random", seed=0, timeout=0.05)
            study = create_study(sampler=RandomSampler(seed=0))
            with (
                pytest.raises(Timeout, match="could not be acquired"),
                store.lock(study),
            ):
                pass
        finally:
            stuck_lock.release()

    def test_default_timeout_is_finite(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        assert store._lock.timeout > 0
