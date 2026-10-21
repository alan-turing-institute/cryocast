from pathlib import Path

import pytest
from omegaconf import DictConfig

from cryocast.model_service import ModelService
from cryocast.model_service.checkpoints import CheckpointFile

from .conftest import CustomCliRunner


class FakeModelService:
    def __init__(self) -> None:
        """Initialise recorded training calls as (checkpoint_dir, multistage)."""
        self.calls: list[tuple[Path | None, bool]] = []

    def train(self) -> None:
        self.calls.append((None, False))

    def train_multistage(self, *, checkpoint_dir: Path | None = None) -> None:
        self.calls.append((checkpoint_dir, True))


class TestTrainCLI:
    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["train", "--help"],
            expected_patterns=[
                r"Usage: cryocast train \[OPTIONS\] \[overrides\]...",
                r"Train a model",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--checkpoint-dir\s+<path>\s+Path to a directory of existing",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--help\s+-h\s+Show this message and exit.",
                r"--multistage\s+Train an EncodeProcessDecode model in",
            ],
        )

    def test_default_training_forwards_composed_config(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FakeModelService()
        captured: list[DictConfig] = []

        def fake_from_config(config: DictConfig) -> FakeModelService:
            captured.append(config)
            return service

        monkeypatch.setattr(ModelService, "from_config", fake_from_config)

        result = runner.call(["train", "--config-name", "sample"])

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert captured[0].model.name == "quick-test"
        assert service.calls == [(None, False)]

    def test_multistage_training_resolves_checkpoint_directory(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FakeModelService()

        def fake_from_config(_config: DictConfig) -> FakeModelService:
            return service

        monkeypatch.setattr(ModelService, "from_config", fake_from_config)
        checkpoint_dir = tmp_path / "checkpoints"

        result = runner.call(
            [
                "train",
                "--config-name",
                "sample",
                "--multistage",
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        )

        assert result.exit_code == 0, result.output
        assert service.calls == [(checkpoint_dir.resolve(), True)]

    def test_multistage_without_checkpoint_dir_passes_none(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FakeModelService()

        def fake_from_config(_config: DictConfig) -> FakeModelService:
            return service

        monkeypatch.setattr(ModelService, "from_config", fake_from_config)

        result = runner.call(["train", "--config-name", "sample", "--multistage"])

        assert result.exit_code == 0, result.output
        assert service.calls == [(None, True)]

    def test_checkpoint_dir_without_multistage_loads_model_from_checkpoint(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resuming single-stage training rebuilds the model from its checkpoint."""
        service = FakeModelService()
        captured: list[CheckpointFile] = []

        def fake_from_checkpoint(
            _config: DictConfig, checkpoint_file: CheckpointFile
        ) -> FakeModelService:
            captured.append(checkpoint_file)
            return service

        def fail_from_config(_config: DictConfig) -> FakeModelService:
            pytest.fail("from_config should not be used when resuming")

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)
        monkeypatch.setattr(ModelService, "from_config", fail_from_config)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "last.ckpt").write_text("checkpoint")

        result = runner.call(
            [
                "train",
                "--config-name",
                "sample",
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        )

        assert result.exit_code == 0, result.output
        assert captured == [CheckpointFile(checkpoint_dir.resolve() / "last.ckpt")]
        assert service.calls == [(None, False)]

    def test_checkpoint_dir_without_last_checkpoint_fails(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
    ) -> None:
        """Resuming single-stage training needs a last*.ckpt file to resume from."""
        result = runner.call(
            [
                "train",
                "--config-name",
                "sample",
                "--checkpoint-dir",
                str(tmp_path),
            ]
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_checkpoint_dir_resolves_a_relative_path(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A relative --checkpoint-dir is resolved against the current directory."""
        service = FakeModelService()

        def fake_from_config(_config: DictConfig) -> FakeModelService:
            return service

        monkeypatch.setattr(ModelService, "from_config", fake_from_config)
        monkeypatch.chdir(tmp_path)

        result = runner.call(
            [
                "train",
                "--config-name",
                "sample",
                "--multistage",
                "--checkpoint-dir",
                "checkpoints",
            ]
        )

        assert result.exit_code == 0, result.output
        assert service.calls == [((tmp_path / "checkpoints").resolve(), True)]
