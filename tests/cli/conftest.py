import re
from collections.abc import Sequence

import pytest
from typer.testing import CliRunner, Result

from icenet_mp.cli.main import app


class CustomCliRunner(CliRunner):
    def __init__(self) -> None:
        """A custom CLI runner for IceNet-MP tests."""
        super().__init__(env={"COLUMNS": "120"})
        self.colorstrip = re.compile(r"\x1b\[[0-9;]*m")

    def call(self, commands: Sequence[str]) -> Result:
        return super().invoke(app, commands, prog_name="imp")

    def check_output(
        self, commands: Sequence[str], expected_patterns: Sequence[str]
    ) -> None:
        """Check if the output contains all expected patterns."""
        output = self.output(commands)
        for pattern in expected_patterns:
            found_match = any(re.search(pattern, line) for line in output)
            assert found_match, f"Pattern '{pattern}' not found in output."

    def output(self, commands: Sequence[str]) -> list[str]:
        """Invoke the CLI commands and return the output as a list of strings."""
        result = self.call(commands)
        assert result.exit_code == 0, (
            f"Command failed with exit code {result.exit_code}: {result.output}"
        )
        if result.exception:
            raise result.exception
        return [self.colorstrip.sub("", line) for line in result.output.split("\n")]


@pytest.fixture
def runner() -> CustomCliRunner:
    """A custom CLI runner for IceNet-MP tests."""
    return CustomCliRunner()
