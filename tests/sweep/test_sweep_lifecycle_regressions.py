"""Regression tests for persisted sweep runs, failed trials, and safe restarts.

Use real temporary SQLite studies and sampler files. No W&B API or training
job is started. The intent is to catch state loss that an in-memory mock misses.
"""

import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from omegaconf import OmegaConf
from optuna.trial import TrialState

from cryocast.sweep import OptunaSweep

SWEEP: dict[str, Any] = {
    "entity": "offline-regression",
    "name": "restart-regression",
    "sampler": "random",
    "seed": 721,
    "parameters": {
        "train.optimizer.lr": {
            "type": "float",
            "low": 0.0001,
            "high": 0.1,
            "log": True,
        },
        "model.width": {"type": "int", "low": 2, "high": 8, "step": 2},
        "model.kind": {"type": "categorical", "choices": ["small", "large"]},
    },
}


def make_sweep(root: Path, *, seed: int = 721) -> OptunaSweep:
    """Create the real on-disk config, SQLite study, and sampler namespace."""
    root.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.create(
        {
            "base_path": str(root),
            "train": {"optimizer": {"lr": 0.01}},
            "model": {"width": 4, "kind": "small"},
        }
    )
    OptunaSweep({**SWEEP, "seed": seed}).initialise_study(config, "regression-study")
    return OptunaSweep.from_path(root / "sweeps" / "regression-study")


def values_from_ask(sweep: OptunaSweep) -> dict[str, int | float | str]:
    """Return a real Optuna trial's suggestions without depending on trial ID."""
    _, overrides = sweep.ask()
    return {parameter.name: value for parameter, value in overrides}


class TestPersistedSamplerSequences:
    """A process restart must not rewind or resample an advanced RNG."""

    def test_reopening_a_study_preserves_the_exact_random_sampler_sequence(
        self, tmp_path: Path
    ) -> None:
        uninterrupted = make_sweep(tmp_path / "uninterrupted")
        expected = [values_from_ask(uninterrupted) for _ in range(5)]

        resumed = make_sweep(tmp_path / "resumed")
        actual = [values_from_ask(resumed)]
        for _ in range(4):
            resumed = OptunaSweep.from_path(resumed.study_path)
            actual.append(values_from_ask(resumed))

        assert actual == expected
        assert [trial.number for trial in resumed.study.get_trials()] == list(range(5))

    def test_failed_and_pruned_trials_do_not_rewind_sampler_state(
        self, tmp_path: Path
    ) -> None:
        expected_sweep = make_sweep(tmp_path / "comparison")
        expected = [values_from_ask(expected_sweep) for _ in range(4)]

        sweep = make_sweep(tmp_path / "interrupted")
        first, first_overrides = sweep.ask()
        sweep.tell(first, state=TrialState.FAIL)

        sweep = OptunaSweep.from_path(sweep.study_path)
        second, second_overrides = sweep.ask()
        sweep.tell(second, state=TrialState.PRUNED)

        sweep = OptunaSweep.from_path(sweep.study_path)
        third, third_overrides = sweep.ask()
        sweep.tell(third, 0.35)

        sweep = OptunaSweep.from_path(sweep.study_path)
        fourth, fourth_overrides = sweep.ask()
        actual = [
            {parameter.name: value for parameter, value in overrides}
            for overrides in (
                first_overrides,
                second_overrides,
                third_overrides,
                fourth_overrides,
            )
        ]

        assert actual == expected
        assert fourth.number == 3
        assert [trial.state for trial in sweep.study.get_trials()] == [
            TrialState.FAIL,
            TrialState.PRUNED,
            TrialState.COMPLETE,
            TrialState.RUNNING,
        ]
        assert sweep.study.best_trial.number == 2

    def test_corrupt_sampler_file_recovers_without_losing_sqlite_trials(
        self, tmp_path: Path
    ) -> None:
        sweep = make_sweep(tmp_path)
        first, _ = sweep.ask()
        sweep.tell(first, 0.75)

        sampler_file = sweep.study_path / "sampler.pkl"
        sampler_file.write_bytes(b"\x80broken-pickle")

        resumed = OptunaSweep.from_path(sweep.study_path)
        second, overrides = resumed.ask()
        assert second.number == 1
        assert len(overrides) == len(SWEEP["parameters"])
        assert resumed.study.trials[0].value == 0.75
        assert resumed.study.trials[0].state == TrialState.COMPLETE
        assert resumed.study.trials[1].state == TrialState.RUNNING
        assert isinstance(
            pickle.loads(sampler_file.read_bytes()),  # noqa: S301
            type(resumed.study.sampler),
        )


class TestSweepOutcomes:
    """Completed, failed, and pruned trials must survive study reloads."""

    @pytest.mark.parametrize(
        "state", [TrialState.FAIL, TrialState.PRUNED], ids=["failed", "pruned"]
    )
    def test_tell_persists_a_noncompleted_state_after_reload(
        self, tmp_path: Path, state: TrialState
    ) -> None:
        sweep = make_sweep(tmp_path)
        trial, _ = sweep.ask()
        result = sweep.tell(trial, state=state)

        reloaded = OptunaSweep.from_path(sweep.study_path)
        assert result.state == state
        assert len(reloaded.study.get_trials()) == 1
        assert reloaded.study.trials[0].state == state
        assert reloaded.study.trials[0].value is None
        with pytest.raises(ValueError, match="Record does not exist"):
            _ = reloaded.study.best_trial

    def test_best_completed_trial_survives_failed_and_running_trials(
        self, tmp_path: Path
    ) -> None:
        sweep = make_sweep(tmp_path)
        for outcome in (0.6, None, 0.2):
            trial, _ = sweep.ask()
            if outcome is None:
                sweep.tell(trial, state=TrialState.FAIL)
            else:
                sweep.tell(trial, outcome)
        running, _ = sweep.ask()

        reloaded = OptunaSweep.from_path(sweep.study_path)
        assert running.number == 3
        assert reloaded.study.best_trial.number == 2
        assert reloaded.study.best_value == 0.2
        assert [trial.state for trial in reloaded.study.get_trials()] == [
            TrialState.COMPLETE,
            TrialState.FAIL,
            TrialState.COMPLETE,
            TrialState.RUNNING,
        ]

    def test_double_tell_rejects_accidental_score_overwrite(
        self, tmp_path: Path
    ) -> None:
        sweep = make_sweep(tmp_path)
        trial, _ = sweep.ask()
        sweep.tell(trial, 0.5)

        with pytest.raises(ValueError, match="Cannot tell"):
            sweep.tell(trial.number, 0.01)

        reloaded = OptunaSweep.from_path(sweep.study_path)
        assert reloaded.study.best_value == 0.5
        assert len(reloaded.study.trials) == 1

    def test_skip_if_finished_is_an_idempotent_retry(self, tmp_path: Path) -> None:
        sweep = make_sweep(tmp_path)
        trial, _ = sweep.ask()
        completed = sweep.tell(trial, 0.5)
        retried = sweep.tell(trial.number, 0.01, skip_if_finished=True)

        assert retried.number == completed.number
        assert retried.value == completed.value == 0.5
        assert OptunaSweep.from_path(sweep.study_path).study.best_value == 0.5

    def test_recovery_can_submit_a_new_trial_after_two_failures(
        self, tmp_path: Path
    ) -> None:
        sweep = make_sweep(tmp_path)
        for state in (TrialState.PRUNED, TrialState.FAIL):
            trial, _ = sweep.ask()
            sweep.tell(trial, state=state)
            sweep = OptunaSweep.from_path(sweep.study_path)

        recovered, overrides = sweep.ask()
        result = sweep.tell(recovered, 0.125)
        assert recovered.number == 2
        assert result.state == TrialState.COMPLETE
        assert len(overrides) == 3
        assert OptunaSweep.from_path(sweep.study_path).study.best_value == 0.125


class TestSweepResumption:
    """A resumed model must not silently receive newly sampled parameters."""

    def test_resume_keeps_all_trained_values_and_persists_provenance(
        self, tmp_path: Path
    ) -> None:
        sweep = make_sweep(tmp_path)
        original, _ = sweep.ask()
        sweep.tell(original, 0.9)

        trained = tmp_path / "trained.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "train": {"optimizer": {"lr": 0.0007}},
                    "model": {"width": 6, "kind": "large"},
                }
            ),
            trained,
        )
        trial, overrides = sweep.resume(trained)
        values = {parameter.name: value for parameter, value in overrides}
        assert values == {
            "train.optimizer.lr": 0.0007,
            "model.width": 6,
            "model.kind": "large",
        }
        assert trial.number == 1

        sweep.tell(trial, 0.4)
        restored = OptunaSweep.from_path(sweep.study_path)
        assert restored.study.trials[1].params == values
        assert restored.study.trials[1].user_attrs == {"resumed_from": str(trained)}
        assert restored.study.best_trial.number == 1

    def test_missing_resume_parameters_cannot_enqueue_a_partial_trial(
        self, tmp_path: Path
    ) -> None:
        sweep = make_sweep(tmp_path)
        trained = tmp_path / "incomplete.yaml"
        OmegaConf.save(
            OmegaConf.create({"train": {"optimizer": {"lr": 0.0007}}}),
            trained,
        )
        with pytest.raises(ValueError, match="Could not find parameter") as exc:
            sweep.resume(trained)
        assert "model.width" in str(exc.value)
        assert "model.kind" in str(exc.value)
        assert sweep.study.trials == []
        assert not (sweep.study_path / "sampler.pkl").exists()

    def test_explicit_yaml_string_choice_survives_resume(self, tmp_path: Path) -> None:
        sweep = make_sweep(tmp_path)
        sweep.parameters["model.kind"] = {
            "type": "categorical",
            "choices": ["off", "null", "large"],
        }
        # Persist the updated search space instead of mutating only in memory.
        (sweep.study_path / "optuna.yaml").write_text(
            yaml.safe_dump({**SWEEP, "parameters": sweep.parameters})
        )
        sweep = OptunaSweep.from_path(sweep.study_path)
        path = tmp_path / "trained.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "train": {"optimizer": {"lr": 0.002}},
                    "model": {"width": 4, "kind": "off"},
                }
            ),
            path,
        )
        trial, overrides = sweep.resume(path)
        config = sweep.generate_trial_config(overrides)
        assert OmegaConf.select(config, "model.kind") == "off"
        assert (
            OmegaConf.select(config, "reporting.loggers.wandb.config")["model.kind"]
            == "off"
        )
        assert trial.params["model.kind"] == "off"


def test_parameter_construction_failure_does_not_allocate_a_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject invalid parameter declarations before mutating the SQLite study."""
    sweep = make_sweep(tmp_path)
    error = ValueError("invalid search space")
    monkeypatch.setattr(
        "cryocast.sweep.optuna_sweep.build_parameter", MagicMock(side_effect=error)
    )

    with pytest.raises(ValueError, match="invalid search space") as caught:
        sweep.ask()

    assert caught.value is error
    assert OptunaSweep.from_path(sweep.study_path).study.trials == []
    assert not (sweep.study_path / "sampler.pkl").exists()


@pytest.mark.parametrize("method", ["ask", "resume"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failed_parameter_suggestion_does_not_leave_a_running_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    error_type: type[BaseException],
) -> None:
    """A failed or interrupted suggestion must finish its allocated trial."""
    sweep = make_sweep(tmp_path)
    trained = tmp_path / "trained.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "train": {"optimizer": {"lr": 0.0007}},
                "model": {"width": 6, "kind": "large"},
            }
        ),
        trained,
    )
    error = error_type("parameter suggestion interrupted")
    parameter = sweep._built_parameters[1]

    def request_trial() -> None:
        if method == "ask":
            sweep.ask()
        else:
            sweep.resume(trained)

    with monkeypatch.context() as patch:
        patch.setattr(type(parameter), "suggest", MagicMock(side_effect=error))
        with pytest.raises(
            error_type, match="parameter suggestion interrupted"
        ) as caught:
            request_trial()
        assert caught.value is error

    restored = OptunaSweep.from_path(sweep.study_path)
    assert [trial.state for trial in restored.study.trials] == [TrialState.FAIL]
    assert restored.study.trials[0].value is None
    recovered, _ = restored.ask()
    restored.tell(recovered, 0.125)
    assert restored.study.best_trial.number == 1
    assert restored.study.best_value == 0.125
