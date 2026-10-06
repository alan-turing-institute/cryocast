import pytest

import icenet_mp.cli.feature_importance as feature_importance_module

from .conftest import CustomCliRunner


class TestFeatureImportanceCLI:
    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["feature-importance", "--help"],
            expected_patterns=[
                r"Usage: imp feature-importance \[OPTIONS\] \[overrides\]...",
                r"Fit a Random Forest and print input variables ranked by importance",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    def test_prints_ranked_variables(
        self, runner: CustomCliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            feature_importance_module,
            "compute_feature_importance",
            lambda _config: [("group1/ice_conc", 0.7), ("group1/temperature", 0.3)],
        )

        output = runner.output(["feature-importance", "--config-name", "sample"])

        assert any("group1/ice_conc" in line and "0.700000" in line for line in output)
        assert any(
            "group1/temperature" in line and "0.300000" in line for line in output
        )
