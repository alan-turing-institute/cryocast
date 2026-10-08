import logging
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from cryocast.checkpoint_config import load_checkpoint_config, merge_checkpoint_config


def _save_run_config(run_dir: Path, config: DictConfig) -> Path:
    """Save a run's model_config.yaml and return the path of a checkpoint in the run."""
    (run_dir / "files").mkdir(parents=True)
    OmegaConf.save(config, run_dir / "files" / "model_config.yaml")
    return run_dir / "checkpoints" / "last.ckpt"


class TestLoadCheckpointConfig:
    def test_loads_saved_config(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Load the model_config.yaml saved alongside a checkpoint."""
        checkpoint_path = _save_run_config(tmp_path, cfg_model_service)

        assert load_checkpoint_config(checkpoint_path) == cfg_model_service

    def test_returns_none_with_warning_when_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn and return None when there is no saved model_config.yaml."""
        with caplog.at_level(logging.WARNING, logger="cryocast.checkpoint_config"):
            result = load_checkpoint_config(tmp_path / "checkpoints" / "last.ckpt")

        assert result is None
        assert "Could not load the checkpoint configuration" in caplog.text

    @pytest.mark.parametrize(
        ("legacy_target", "expected_target"),
        [
            (
                {"group_name": "sic-ssmis", "variables": ["ice_conc"]},
                {"sic-ssmis": ["ice_conc"]},
            ),
            ({"group_name": "sic-ssmis"}, {"sic-ssmis": []}),
            ({"group_name": "sic-ssmis", "variables": []}, {"sic-ssmis": []}),
        ],
        ids=["explicit-variables", "missing-variables", "empty-variables"],
    )
    def test_translates_legacy_predict_config(
        self,
        legacy_target: dict[str, Any],
        expected_target: dict[str, Any],
        cfg_model_service: DictConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Translate configs that predate the 'variables'/'window' split."""
        legacy_config = cfg_model_service.copy()
        del legacy_config["variables"]
        del legacy_config["window"]
        legacy_config["predict"] = {
            "target": legacy_target,
            "n_forecast_steps": 7,
            "n_history_steps": 4,
        }
        checkpoint_path = _save_run_config(tmp_path, legacy_config)

        with caplog.at_level(logging.WARNING, logger="cryocast.checkpoint_config"):
            ckpt_config = load_checkpoint_config(checkpoint_path)

        assert ckpt_config is not None
        assert "uses the legacy 'predict' key" in caplog.text
        assert "predict" not in ckpt_config
        assert OmegaConf.to_container(ckpt_config["variables"]) == {
            "input": {},
            "target": expected_target,
        }
        assert OmegaConf.to_container(ckpt_config["window"]) == {
            "n_forecast_steps": 7,
            "n_history_steps": 4,
        }


class TestMergeCheckpointConfig:
    def test_model_and_window_from_checkpoint_and_the_rest_from_config(
        self, cfg_model_service: DictConfig
    ) -> None:
        """The checkpoint describes the model; everything else is from the config."""
        config = cfg_model_service.copy()
        config["model"]["name"] = "will_not_overwrite"
        config["window"]["n_history_steps"] = 99
        config["reporting"]["loggers"] = "will_overwrite"
        config["train"]["trainer"] = {"max_epochs": 200}

        combined = merge_checkpoint_config(config, cfg_model_service)

        assert combined["model"]["name"] == "mock-model"
        assert combined["window"]["n_history_steps"] == 3
        assert combined["reporting"]["loggers"] == "will_overwrite"
        assert combined["train"]["trainer"]["max_epochs"] == 200

    def test_variables_replaced_not_merged(self, cfg_model_service: DictConfig) -> None:
        """Checkpoint variables win outright rather than being unioned with the config."""
        ckpt_config = cfg_model_service.copy()
        ckpt_config["variables"] = {
            "input": {"sic-ssmis": ["ice_conc"]},
            "target": {"sic-ssmis": ["ice_conc"]},
        }
        config = cfg_model_service.copy()
        config["variables"] = {
            "input": {
                "era5": ["2t"],
                "float-argo": ["TEMP"],
                "sic-osisaf": ["ice_conc"],
            },
            "target": {"sic-osisaf": ["ice_conc"]},
        }

        combined = merge_checkpoint_config(config, ckpt_config)

        assert combined["variables"] == ckpt_config["variables"]

    def test_batch_size_from_config(self, cfg_model_service: DictConfig) -> None:
        """Batch size comes from the config; other window keys do not."""
        config = cfg_model_service.copy()
        config["window"] = {
            "batch_size": 1,
            "n_forecast_steps": 7,
            "n_history_steps": 1,
        }

        combined = merge_checkpoint_config(config, cfg_model_service)

        assert OmegaConf.to_container(combined["window"]) == {
            "batch_size": 1,
            "n_forecast_steps": 2,
            "n_history_steps": 3,
        }

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            (
                {"window": {"n_history_steps": 7}},
                ["window.n_history_steps=3 (configured as 7)"],
            ),
            (
                {"model": {"processor": {"n_blocks": 8}}},
                ["model.processor.n_blocks=6 (configured as 8)"],
            ),
            (
                {"variables": {"input": {"era5": ["2t"]}}},
                ["variables.input.era5='unset' (configured as ['2t'])"],
            ),
            ({"window": {"batch_size": 8}}, []),
            ({"train": {"max_epochs": 8}}, []),
            ({}, []),
        ],
        ids=["window", "model", "variables", "batch-size", "train", "unchanged"],
    )
    def test_warns_about_replaced_values(
        self,
        caplog: pytest.LogCaptureFixture,
        overrides: dict[str, Any],
        expected: list[str],
    ) -> None:
        """Warn about each config value that the checkpoint config replaces."""
        ckpt_config = DictConfig(
            {
                "model": {"name": "dc-gsta-dc", "processor": {"n_blocks": 6}},
                "train": {"max_epochs": 4},
                "variables": {"input": {"sic": ["ice_conc"]}},
                "window": {"batch_size": 2, "n_history_steps": 3},
            }
        )
        config = DictConfig(OmegaConf.merge(ckpt_config, overrides))

        with caplog.at_level(logging.WARNING, logger="cryocast.checkpoint_config"):
            merge_checkpoint_config(config, ckpt_config)

        for message in expected:
            assert message in caplog.text
        if not expected:
            assert not caplog.records

    def test_summarises_different_models_in_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Report a different model in one line rather than every differing value."""
        ckpt_config = DictConfig({"model": {"name": "dc-gsta-dc", "depth": 6}})
        config = DictConfig({"model": {"name": "quick-test", "depth": 2}})

        with caplog.at_level(logging.WARNING, logger="cryocast.checkpoint_config"):
            merge_checkpoint_config(config, ckpt_config)

        assert len(caplog.records) == 1
        assert (
            "Using the 'dc-gsta-dc' model from the checkpoint rather than the "
            "configured 'quick-test' model." in caplog.text
        )
