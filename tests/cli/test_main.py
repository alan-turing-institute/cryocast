import logging

import pytest

from cryocast.cli.main import run

from .conftest import CustomCliRunner


class TestBaseCLI:
    expected_patterns_help = (
        r"Usage: imp \[OPTIONS\] COMMAND \[ARGS\]...",
        r"Entrypoint for imp CLI application.",
        r"--install-completion\s+Install completion for the current shell.",
        r"--show-completion\s+Show completion for the current shell",
        r"--help\s+-h\s+Show this message and exit.",
        r"datasets\s+Manage datasets",
        r"evaluate\s+Evaluate a pre-trained model",
        r"sweep\s+Generate W&B sweeps with Optuna-sampled",
        r"train\s+Train a model",
    )

    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["--help"],
            expected_patterns=self.expected_patterns_help,
        )

    def test_short_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["-h"],
            expected_patterns=self.expected_patterns_help,
        )


class TestRunEntrypoint:
    def test_mps_not_implemented_error_exits_with_code_1(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _raise_app() -> None:
            msg = "aten::foo is not currently implemented for the MPS device."
            raise NotImplementedError(msg)

        monkeypatch.setattr("cryocast.cli.main.app", _raise_app)

        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
            run()

        assert exc_info.value.code == 1
        assert "PYTORCH_ENABLE_MPS_FALLBACK=1" in caplog.text

    def test_other_not_implemented_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_app() -> None:
            msg = "Some unrelated feature is not implemented."
            raise NotImplementedError(msg)

        monkeypatch.setattr("cryocast.cli.main.app", _raise_app)

        with pytest.raises(NotImplementedError, match="Some unrelated feature"):
            run()
