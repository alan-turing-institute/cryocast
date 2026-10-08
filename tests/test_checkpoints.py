import logging
import pickle
import sys
from enum import StrEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cryocast.model_service.checkpoints import (
    OutdatedCheckpointError,
    find_checkpoint,
    load_checkpoint,
    load_checkpoint_config,
    merge_checkpoint_config,
    upgrade_checkpoint,
    verify_model_matches_data,
)
from cryocast.types import DataSpace


def _save_run_config(run_dir: Path, config: DictConfig) -> Path:
    """Save a run's model_config.yaml and return the path of a checkpoint in the run."""
    (run_dir / "files").mkdir(parents=True)
    OmegaConf.save(config, run_dir / "files" / "model_config.yaml")
    return run_dir / "checkpoints" / "last.ckpt"


class TestFindCheckpoint:
    def test_returns_checkpoint_file(self, tmp_path: Path) -> None:
        """Return a checkpoint file path unchanged."""
        checkpoint_path = tmp_path / "epoch=3-step=10.ckpt"
        checkpoint_path.write_text("checkpoint")

        assert find_checkpoint(checkpoint_path) == checkpoint_path

    def test_finds_last_checkpoint_in_directory(self, tmp_path: Path) -> None:
        """Pick the ``last*.ckpt`` file in a directory, not a best-epoch checkpoint."""
        for name in ("epoch=3-step=10.ckpt", "last.ckpt"):
            (tmp_path / name).write_text("checkpoint")

        assert find_checkpoint(tmp_path) == tmp_path / "last.ckpt"

    def test_raises_without_last_checkpoint_in_directory(self, tmp_path: Path) -> None:
        """Reject a directory with no resumable ``last*.ckpt`` file."""
        (tmp_path / "epoch=3-step=10.ckpt").write_text("checkpoint")

        with pytest.raises(FileNotFoundError, match=r"last\*.ckpt"):
            find_checkpoint(tmp_path)

    def test_raises_when_checkpoint_missing(self, tmp_path: Path) -> None:
        """Reject a checkpoint file that does not exist."""
        with pytest.raises(FileNotFoundError, match="Could not find checkpoint file"):
            find_checkpoint(tmp_path / "missing.ckpt")


class TestLoadCheckpoint:
    def test_loads_plain_checkpoint(self, tmp_path: Path) -> None:
        """Load a checkpoint that contains only plain Python types and tensors."""
        checkpoint_path = tmp_path / "last.ckpt"
        torch.save(
            {"epoch": 3, "state_dict": {"weight": torch.ones(2)}}, checkpoint_path
        )

        checkpoint = load_checkpoint(checkpoint_path)

        assert checkpoint["epoch"] == 3
        assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(2))

    def test_suggests_upgrading_outdated_checkpoint(self, tmp_path: Path) -> None:
        """Refuse to load other objects, suggesting how to upgrade the checkpoint."""
        checkpoint_path = tmp_path / "last.ckpt"
        torch.save(
            {"callbacks": {"dirpath": Path("/run/checkpoints")}}, checkpoint_path
        )

        with pytest.raises(OutdatedCheckpointError) as exc_info:
            load_checkpoint(checkpoint_path)

        assert f"cryocast checkpoint upgrade {checkpoint_path}" in str(exc_info.value)
        assert isinstance(exc_info.value, pickle.UnpicklingError)

    def test_loads_upgraded_checkpoint(self, tmp_path: Path) -> None:
        """Load a checkpoint after upgrading it."""
        checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
        _save_legacy_checkpoint(checkpoint_path)
        with pytest.raises(OutdatedCheckpointError):
            load_checkpoint(checkpoint_path)

        upgrade_checkpoint(checkpoint_path)

        assert load_checkpoint(checkpoint_path)["hyper_parameters"]["hemisphere"] == (
            "north"
        )


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
        with caplog.at_level(
            logging.WARNING, logger="cryocast.model_service.checkpoints"
        ):
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

        with caplog.at_level(
            logging.WARNING, logger="cryocast.model_service.checkpoints"
        ):
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

        with caplog.at_level(
            logging.WARNING, logger="cryocast.model_service.checkpoints"
        ):
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

        with caplog.at_level(
            logging.WARNING, logger="cryocast.model_service.checkpoints"
        ):
            merge_checkpoint_config(config, ckpt_config)

        assert len(caplog.records) == 1
        assert (
            "Using the 'dc-gsta-dc' model from the checkpoint rather than the "
            "configured 'quick-test' model." in caplog.text
        )


def _describe_data(**overrides: object) -> SimpleNamespace:
    """Describe a model or data module with the attributes that are compared."""
    attributes: dict[str, object] = {
        "input_spaces": [DataSpace(5, "input", (20, 20))],
        "n_forecast_steps": 2,
        "n_history_steps": 3,
        "output_space": DataSpace(1, "output", (10, 10)),
    }
    return SimpleNamespace(**(attributes | overrides))


class TestVerifyModelMatchesData:
    def test_accepts_matching_data(self) -> None:
        """Accept data that matches the model."""
        verify_model_matches_data(_describe_data(), _describe_data())  # type: ignore[arg-type]

    def test_ignores_input_order(self) -> None:
        """Inputs are matched by name, so their order does not matter."""
        spaces = [DataSpace(5, "a", (20, 20)), DataSpace(2, "b", (20, 20))]

        verify_model_matches_data(
            _describe_data(input_spaces=spaces),  # type: ignore[arg-type]
            _describe_data(input_spaces=spaces[::-1]),  # type: ignore[arg-type]
        )

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_history_steps": 2}, "n_history_steps is 3 in the model but 2"),
            ({"n_forecast_steps": 5}, "n_forecast_steps is 2 in the model but 5"),
            (
                {"input_spaces": [DataSpace(4, "input", (20, 20))]},
                r"input 'input' is DataSpace\(channels=5.* but DataSpace\(channels=4",
            ),
            (
                {
                    "input_spaces": [
                        DataSpace(5, "input", (20, 20)),
                        DataSpace(2, "extra", (20, 20)),
                    ]
                },
                "input 'extra' is missing in the model",
            ),
            (
                {"output_space": DataSpace(2, "output", (10, 10))},
                r"output is DataSpace\(channels=1.* but DataSpace\(channels=2",
            ),
        ],
        ids=["history", "forecast", "input-channels", "extra-input", "output"],
    )
    def test_raises_on_mismatch(self, overrides: dict[str, object], match: str) -> None:
        """Report how the data differs from the model."""
        with pytest.raises(ValueError, match=match):
            verify_model_matches_data(
                _describe_data(),  # type: ignore[arg-type]
                _describe_data(**overrides),  # type: ignore[arg-type]
            )


def _save_legacy_checkpoint(checkpoint_path: Path) -> None:
    """Save a checkpoint as icenet-mp did, with enums, paths and OmegaConf objects."""

    class Hemisphere(StrEnum):
        NORTH = "north"

    # Pickle Hemisphere as icenet_mp.types.enums.Hemisphere, as icenet-mp did
    Hemisphere.__module__ = "icenet_mp.types.enums"
    Hemisphere.__qualname__ = "Hemisphere"
    module = ModuleType(Hemisphere.__module__)
    module.Hemisphere = Hemisphere  # type: ignore[attr-defined]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        for name in ("icenet_mp", "icenet_mp.types", "icenet_mp.types.enums"):
            mp.setitem(sys.modules, name, module)
        torch.save(
            {
                "callbacks": {"ModelCheckpoint": {"dirpath": Path("/run/checkpoints")}},
                "hyper_parameters": {
                    "hemisphere": Hemisphere.NORTH,
                    "loss": DictConfig({"_target_": "icenet_mp.losses.MAELoss"}),
                },
                "state_dict": {"weight": torch.ones(2)},
            },
            checkpoint_path,
        )


class TestUpgradeCheckpoint:
    def test_upgraded_checkpoint_loads_with_weights_only(self, tmp_path: Path) -> None:
        """Upgrade to plain types that reference cryocast, keeping a backup."""
        checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
        _save_legacy_checkpoint(checkpoint_path)

        upgrade_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert (
            checkpoint["callbacks"]["ModelCheckpoint"]["dirpath"] == "/run/checkpoints"
        )
        assert checkpoint["hyper_parameters"] == {
            "hemisphere": "north",
            "loss": {"_target_": "cryocast.losses.MAELoss"},
        }
        assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(2))
        assert (tmp_path / "checkpoints" / "last.ckpt.bak").is_file()

    def test_upgrades_legacy_model_config(self, tmp_path: Path) -> None:
        """Rewrite icenet_mp references in the run's model config, keeping a backup."""
        checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
        _save_legacy_checkpoint(checkpoint_path)
        legacy_config = "model:\n  _target_: icenet_mp.models.EncodeProcessDecode\n"
        config_path = tmp_path / "files" / "model_config.yaml"
        config_path.parent.mkdir()
        config_path.write_text(legacy_config)

        upgrade_checkpoint(checkpoint_path)

        assert config_path.read_text() == legacy_config.replace("icenet_mp", "cryocast")
        assert (config_path.parent / "model_config.yaml.bak").read_text() == (
            legacy_config
        )

    def test_upgrades_several_checkpoints_from_one_run(self, tmp_path: Path) -> None:
        """Upgrade every checkpoint in a run, with its model config upgraded once."""
        checkpoint_paths = [
            tmp_path / "checkpoints" / name for name in ("epoch=1.ckpt", "last.ckpt")
        ]
        for checkpoint_path in checkpoint_paths:
            _save_legacy_checkpoint(checkpoint_path)
        config_path = tmp_path / "files" / "model_config.yaml"
        config_path.parent.mkdir()
        config_path.write_text("model:\n  _target_: icenet_mp.models.Persistence\n")

        for checkpoint_path in checkpoint_paths:
            upgrade_checkpoint(checkpoint_path)

        for checkpoint_path in checkpoint_paths:
            assert not torch.serialization.get_unsafe_globals_in_checkpoint(
                checkpoint_path
            )
        assert "icenet_mp" not in config_path.read_text()

    def test_refuses_to_overwrite_backup(self, tmp_path: Path) -> None:
        """Refuse to upgrade twice, which would overwrite the original backup."""
        checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
        _save_legacy_checkpoint(checkpoint_path)
        upgrade_checkpoint(checkpoint_path)
        upgraded = checkpoint_path.read_bytes()

        with pytest.raises(FileExistsError, match="already been upgraded"):
            upgrade_checkpoint(checkpoint_path)

        assert checkpoint_path.read_bytes() == upgraded
        assert not list(checkpoint_path.parent.glob("*.tmp"))
