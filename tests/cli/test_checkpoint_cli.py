from pathlib import Path

import pytest

from cryocast.cli import checkpoint

from .conftest import CustomCliRunner


class TestCheckpointCLI:
    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["checkpoint", "upgrade", "--help"],
            expected_patterns=[
                r"Usage: cryocast checkpoint upgrade \[OPTIONS\] {checkpoints}...",
                r"Upgrade checkpoints saved by older versions of CryoCast or by",
                r"checkpoints\s+<path>\s+One or more checkpoint files to upgrade",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    def test_upgrade_resolves_and_upgrades_each_checkpoint(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        upgraded: list[Path] = []
        monkeypatch.setattr(checkpoint, "upgrade_checkpoint", upgraded.append)
        monkeypatch.chdir(tmp_path)

        result = runner.call(["checkpoint", "upgrade", "a.ckpt", "b.ckpt"])

        assert result.exit_code == 0, result.output
        assert upgraded == [tmp_path / "a.ckpt", tmp_path / "b.ckpt"]
