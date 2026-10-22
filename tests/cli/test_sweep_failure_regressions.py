"""End-to-end sweep CLI regressions using SQLite but no remote services."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from optuna.trial import TrialState

from cryocast.model_service import ModelService
from cryocast.sweep import OptunaSweep

from .conftest import CustomCliRunner


def make_study(tmp_path: Path) -> Path:
    """Initialise the real Optuna and saved Hydra configuration."""
    sweep = OptunaSweep(
        {
            "name": "isolated-cli",
            "sampler": "random",
            "seed": 23,
            "entity": "offline-entity",
            "parameters": {
                "train.optimizer.lr": {
                    "type": "float",
                    "low": 0.00001,
                    "high": 0.01,
                }
            },
        }
    )
    sweep.initialise_study(
        OmegaConf.create(
            {"base_path": str(tmp_path), "train": {"optimizer": {"lr": 0.001}}}
        ),
        "local-regression",
    )
    return sweep.study_path


def trainer_with_score(score: float) -> SimpleNamespace:
    """Use the same ModelCheckpoint interface as the real sweep CLI."""
    callback = MagicMock(spec=ModelCheckpoint)
    callback.best_model_score = torch.tensor(score)
    return SimpleNamespace(checkpoint_callbacks=[callback])


class TestFailedTrialRecovery:
    """A failure must never leave stale RUNNING rows in the persisted study."""

    def test_generation_error_marks_trial_failed_before_starting_training(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = make_study(tmp_path)

        def fail_generation(*_args: object, **_kwargs: object) -> None:
            msg = "bad trial configuration"
            raise ValueError(msg)

        def fail_training(*_args: object, **_kwargs: object) -> None:
            pytest.fail("No model should be started after config generation fails")

        monkeypatch.setattr(OptunaSweep, "generate_trial_config", fail_generation)
        monkeypatch.setattr(ModelService, "from_config", fail_training)
        result = runner.call(["sweep", "trial", "--sweep-path", str(path)])

        assert isinstance(result.exception, ValueError)
        assert "bad trial configuration" in str(result.exception)
        restored = OptunaSweep.from_path(path)
        assert [trial.state for trial in restored.study.trials] == [TrialState.FAIL]

    def test_failed_training_can_be_followed_by_successful_new_trial(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = make_study(tmp_path)
        calls = 0

        class FakeService:
            def train(self) -> SimpleNamespace:
                return trainer_with_score(0.42)

        def on_create(_cfg: object) -> FakeService:
            nonlocal calls
            calls += 1
            if calls == 1:
                msg = "interrupted training"
                raise RuntimeError(msg)
            return FakeService()

        monkeypatch.setattr(ModelService, "from_config", on_create)
        first = runner.call(["sweep", "trial", "--sweep-path", str(path)])
        assert isinstance(first.exception, RuntimeError)

        second = runner.call(["sweep", "trial", "--sweep-path", str(path)])
        assert second.exit_code == 0, second.output

        restored = OptunaSweep.from_path(path)
        assert [trial.state for trial in restored.study.trials] == [
            TrialState.FAIL,
            TrialState.COMPLETE,
        ]
        assert restored.study.best_trial.number == 1
        assert restored.study.best_trial.value == pytest.approx(0.42)
        assert calls == 2

    def test_missing_last_checkpoint_fails_cleanly_after_valid_resume(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = make_study(tmp_path)
        checkpoint_dir = tmp_path / "existing-run" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        files = checkpoint_dir.parent / "files"
        files.mkdir()
        OmegaConf.save(
            OmegaConf.create({"train": {"optimizer": {"lr": 0.0003}}}),
            files / "model_config.yaml",
        )

        def no_model(*_args: object, **_kwargs: object) -> None:
            pytest.fail("No model should load without a last checkpoint")

        monkeypatch.setattr(ModelService, "from_checkpoint", no_model)
        result = runner.call(
            [
                "sweep",
                "trial",
                "--sweep-path",
                str(path),
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        )

        assert result.exit_code != 0
        restored = OptunaSweep.from_path(path)
        assert [trial.state for trial in restored.study.trials] == [TrialState.FAIL]
        assert restored.study.trials[0].params == {"train.optimizer.lr": 0.0003}
        assert restored.study.trials[0].user_attrs["resumed_from"] == str(
            files / "model_config.yaml"
        )

    def test_multistage_training_crash_does_not_leave_a_running_trial(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = make_study(tmp_path)

        class InterruptedModel:
            def train_multistage(self, *, checkpoint_dir: Path | None = None) -> None:
                assert checkpoint_dir is None
                msg = "interrupted stage"
                raise RuntimeError(msg)

        monkeypatch.setattr(
            ModelService, "from_config", lambda _config: InterruptedModel()
        )
        result = runner.call(
            ["sweep", "trial", "--sweep-path", str(path), "--multistage"]
        )

        assert isinstance(result.exception, RuntimeError)
        restored = OptunaSweep.from_path(path)
        assert restored.study.trials[0].state == TrialState.FAIL

    def test_two_checkpoint_callbacks_cannot_be_ambiguously_scored(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = make_study(tmp_path)
        callback_a = trainer_with_score(0.2).checkpoint_callbacks[0]
        callback_b = trainer_with_score(0.4).checkpoint_callbacks[0]

        class FakeService:
            def train(self) -> SimpleNamespace:
                return SimpleNamespace(checkpoint_callbacks=[callback_a, callback_b])

        monkeypatch.setattr(ModelService, "from_config", lambda _config: FakeService())
        with caplog.at_level(logging.WARNING):
            result = runner.call(["sweep", "trial", "--sweep-path", str(path)])

        assert result.exit_code == 0, result.output
        assert "unique ModelCheckpoint callback" in caplog.text
        assert OptunaSweep.from_path(path).study.trials[0].state == TrialState.FAIL

    def test_summarise_reports_failed_pruned_and_running_without_best_trial(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = make_study(tmp_path)
        sweep = OptunaSweep.from_path(path)
        for state in (TrialState.FAIL, TrialState.PRUNED):
            trial, _ = sweep.ask()
            sweep.tell(trial, state=state)
        sweep.ask()  # deliberately unfinished

        with caplog.at_level(logging.INFO):
            result = runner.call(["sweep", "summarise", "--sweep-path", str(path)])

        assert result.exit_code == 0, result.output
        assert "3 trial(s): 0 completed, 1 running, 1 pruned, 1 failed" in caplog.text
        assert "Nothing to summarise" in caplog.text


@pytest.mark.parametrize("multistage", [False, True])
def test_keyboard_interrupt_finishes_trial_and_allows_a_new_run(
    tmp_path: Path,
    runner: CustomCliRunner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    multistage: bool,
) -> None:
    """Ctrl-C must not leave an abandoned trial marked as running in SQLite."""
    path = make_study(tmp_path)
    service = MagicMock()
    train = service.train_multistage if multistage else service.train
    train.side_effect = KeyboardInterrupt
    monkeypatch.setattr(ModelService, "from_config", lambda _config: service)
    for name in ("WANDB_SWEEP_ID", "WANDB_ENTITY", "WANDB_PROJECT"):
        monkeypatch.setenv(name, "isolated-test")
    args = ["sweep", "trial", "--sweep-path", str(path)]
    if multistage:
        args.append("--multistage")

    interrupted = runner.call(args)

    assert interrupted.exit_code != 0
    assert [trial.state for trial in OptunaSweep.from_path(path).study.trials] == [
        TrialState.FAIL
    ]
    train.side_effect = None
    train.return_value = trainer_with_score(0.25)
    completed = runner.call(args)

    assert completed.exit_code == 0, completed.output
    restored = OptunaSweep.from_path(path)
    assert [trial.state for trial in restored.study.trials] == [
        TrialState.FAIL,
        TrialState.COMPLETE,
    ]
    assert restored.study.best_trial.number == 1
    assert restored.study.best_value == pytest.approx(0.25)
