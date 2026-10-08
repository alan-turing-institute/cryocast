from pathlib import Path

import pytest

from cryocast.model_service.checkpoints import CheckpointFile, LegacyCheckpointFile

from .conftest import CustomCliRunner


class TestCheckpointCLI:
    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["checkpoint", "upgrade", "--help"],
            expected_patterns=[
                r"Usage: cryocast checkpoint upgrade \[OPTIONS\] {checkpoints}...",
                r"Upgrade checkpoints saved by older versions of the code.",
                r"checkpoints\s+<path>\s+One or more checkpoint files to upgrade",
                r"--trust\s+Trust the checkpoints to run arbitrary code.",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    @pytest.mark.parametrize(
        ("options", "trusted"),
        [([], False), (["--trust"], True)],
        ids=["untrusted", "trusted"],
    )
    def test_upgrade_resolves_and_upgrades_each_checkpoint(
        self,
        options: list[str],
        trusted: bool,  # noqa: FBT001
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        upgraded: list[tuple[CheckpointFile, bool]] = []

        def fake_upgrade(
            legacy_checkpoint: LegacyCheckpointFile, *, trusted: bool = False
        ) -> CheckpointFile:
            upgraded.append((legacy_checkpoint.checkpoint_file, trusted))
            return legacy_checkpoint.checkpoint_file

        monkeypatch.setattr(LegacyCheckpointFile, "upgrade", fake_upgrade)
        monkeypatch.chdir(tmp_path)
        for name in ("a.ckpt", "b.ckpt"):
            (tmp_path / name).write_text("checkpoint")

        result = runner.call(["checkpoint", "upgrade", *options, "a.ckpt", "b.ckpt"])

        assert result.exit_code == 0, result.output
        assert upgraded == [
            (CheckpointFile(tmp_path.resolve() / "a.ckpt"), trusted),
            (CheckpointFile(tmp_path.resolve() / "b.ckpt"), trusted),
        ]
