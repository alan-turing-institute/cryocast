from pathlib import Path

import pytest
from omegaconf import DictConfig

from cryocast.model_service import ModelService

from .conftest import CustomCliRunner


class FakeModelService:
    def __init__(self) -> None:
        """Initialise recorded training calls."""
        self.calls: list[tuple[Path | None, bool]] = []

    def train(
        self, checkpoint_dir: Path | None = None, *, multistage: bool = False
    ) -> None:
        self.calls.append((checkpoint_dir, multistage))


class TestTrainCLI:
    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["train", "--help"],
            expected_patterns=[
                r"Usage: imp train \[OPTIONS\] \[overrides\]...",
                r"Train a model",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--checkpoint-dir\s+<str>\s+Path to a directory of existing",
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

    def test_checkpoint_dir_without_multistage_still_resolves(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Accept --checkpoint-dir alone; the command has no multistage-only gating."""
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
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        )

        assert result.exit_code == 0, result.output
        assert service.calls == [(checkpoint_dir.resolve(), False)]

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
