from pathlib import Path

import pytest
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService

from .conftest import CustomCliRunner


class FakeModelService:
    def __init__(self) -> None:
        """Initialise evaluation-call tracking."""
        self.evaluate_calls = 0

    def evaluate(self) -> None:
        self.evaluate_calls += 1


class TestEvaluateCLI:
    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["evaluate", "--help"],
            expected_patterns=[
                r"Usage: imp evaluate \[OPTIONS\] \[overrides\]...",
                r"Evaluate a pre-trained model",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--checkpoint\s+<str>\s+Path of a trained model checkpoint",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--help\s+-h\s+Show this message and exit.",
                r"--save-layer\s+<str>\s+Dotted path of a model submodule to hook",
                r"--save-predictions\s+Write predictions for the configured",
            ],
        )

    def test_evaluate_loads_resolved_checkpoint_and_runs(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compose the config, resolve the checkpoint path, and run evaluate()."""
        service = FakeModelService()
        captured: list[tuple[DictConfig, Path]] = []

        def fake_from_checkpoint(
            config: DictConfig, checkpoint: Path
        ) -> FakeModelService:
            captured.append((config, checkpoint))
            return service

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)
        checkpoint = tmp_path / "model.ckpt"

        result = runner.call(
            [
                "evaluate",
                "--config-name",
                "sample",
                "--checkpoint",
                str(checkpoint),
            ]
        )

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert captured[0][1] == checkpoint.resolve()
        assert captured[0][0].model.name == "quick-test"
        assert (
            list(captured[0][0].evaluate.callbacks.activation_saver.layer_paths) == []
        )
        assert captured[0][0].evaluate.callbacks.prediction_writer.enabled is False
        assert service.evaluate_calls == 1

    def test_checkpoint_resolves_a_relative_path(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A relative --checkpoint is resolved against the current directory."""
        service = FakeModelService()
        captured: list[tuple[DictConfig, Path]] = []

        def fake_from_checkpoint(
            config: DictConfig, checkpoint: Path
        ) -> FakeModelService:
            captured.append((config, checkpoint))
            return service

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)
        monkeypatch.chdir(tmp_path)

        result = runner.call(
            [
                "evaluate",
                "--config-name",
                "sample",
                "--checkpoint",
                "model.ckpt",
            ]
        )

        assert result.exit_code == 0, result.output
        assert captured[0][1] == (tmp_path / "model.ckpt").resolve()

    def test_repeated_save_layer_flags_update_activation_saver_config(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeat --save-layer to hook multiple submodules in one run."""
        service = FakeModelService()
        captured: list[DictConfig] = []

        def fake_from_checkpoint(
            config: DictConfig, _checkpoint: Path
        ) -> FakeModelService:
            captured.append(config)
            return service

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)

        result = runner.call(
            [
                "evaluate",
                "--config-name",
                "sample",
                "--checkpoint",
                str(tmp_path / "model.ckpt"),
                "--save-layer",
                "processor.conv1",
                "--save-layer",
                "decoder.model",
            ]
        )

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert list(captured[0].evaluate.callbacks.activation_saver.layer_paths) == [
            "processor.conv1",
            "decoder.model",
        ]
        assert service.evaluate_calls == 1

    def test_save_predictions_updates_prediction_writer_config(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Forward --save-predictions to the prediction writer as an enabled flag."""
        service = FakeModelService()
        captured: list[DictConfig] = []

        def fake_from_checkpoint(
            config: DictConfig, _checkpoint: Path
        ) -> FakeModelService:
            captured.append(config)
            return service

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)

        result = runner.call(
            [
                "evaluate",
                "--config-name",
                "sample",
                "--checkpoint",
                str(tmp_path / "model.ckpt"),
                "--save-predictions",
            ]
        )

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert captured[0].evaluate.callbacks.prediction_writer.enabled is True
        assert service.evaluate_calls == 1
