import logging
import pickle
import sys
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cryocast.model_service.checkpoints import (
    CheckpointFile,
    LegacyCheckpointFile,
    OutdatedCheckpointError,
)


def _save_run_config(run_dir: Path, config: DictConfig) -> CheckpointFile:
    """Save a run's model_config.yaml and return a checkpoint file in the run."""
    (run_dir / "files").mkdir(parents=True)
    OmegaConf.save(config, run_dir / "files" / "model_config.yaml")
    path = run_dir / "checkpoints" / "last.ckpt"
    path.parent.mkdir()
    path.write_text("checkpoint")
    return CheckpointFile(path)


class TestInit:
    def test_wraps_existing_file(self, tmp_path: Path) -> None:
        """Wrap a checkpoint file that exists."""
        path = tmp_path / "last.ckpt"
        path.write_text("checkpoint")

        assert CheckpointFile(path).path == path

    def test_raises_when_checkpoint_missing(self, tmp_path: Path) -> None:
        """Reject a checkpoint file that does not exist."""
        with pytest.raises(FileNotFoundError, match="Could not find checkpoint file"):
            CheckpointFile(tmp_path / "missing.ckpt")

    def test_resolves_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Store the absolute path of a checkpoint given by a relative path."""
        (tmp_path / "last.ckpt").write_text("checkpoint")
        monkeypatch.chdir(tmp_path)

        assert (
            CheckpointFile(Path("last.ckpt")).path == tmp_path.resolve() / "last.ckpt"
        )

    def test_config_path_is_in_the_run_files_directory(self, tmp_path: Path) -> None:
        """The run's config is saved in a files directory next to the checkpoints."""
        path = tmp_path / "checkpoints" / "last.ckpt"
        path.parent.mkdir()
        path.write_text("checkpoint")

        assert CheckpointFile(path).config_path == (
            tmp_path / "files" / "model_config.yaml"
        )


class TestFindLast:
    def test_finds_last_checkpoint_in_directory(self, tmp_path: Path) -> None:
        """Pick the ``last*.ckpt`` file in a directory, not a best-epoch checkpoint."""
        checkpoint_dir = tmp_path
        for name in ("epoch=3-step=10.ckpt", "last.ckpt"):
            (checkpoint_dir / name).write_text("checkpoint")

        checkpoint_file = CheckpointFile.find_last(checkpoint_dir)

        assert checkpoint_file.path == checkpoint_dir / "last.ckpt"

    def test_raises_without_last_checkpoint_in_directory(self, tmp_path: Path) -> None:
        """Reject a directory with no resumable ``last*.ckpt`` file."""
        checkpoint_dir = tmp_path
        (checkpoint_dir / "epoch=3-step=10.ckpt").write_text("checkpoint")

        with pytest.raises(FileNotFoundError, match=r"last\*.ckpt"):
            CheckpointFile.find_last(checkpoint_dir)


class TestLoad:
    def test_loads_plain_checkpoint(self, tmp_path: Path) -> None:
        """Load a checkpoint that contains only plain Python types and tensors."""
        path = tmp_path / "last.ckpt"
        torch.save({"epoch": 3, "state_dict": {"weight": torch.ones(2)}}, path)

        checkpoint = CheckpointFile(path).load()

        assert checkpoint["epoch"] == 3
        assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(2))

    def test_suggests_upgrading_outdated_checkpoint(self, tmp_path: Path) -> None:
        """Refuse to load other objects, suggesting how to upgrade the checkpoint."""
        path = tmp_path / "last.ckpt"
        torch.save({"callbacks": {"dirpath": Path("/run/checkpoints")}}, path)

        with pytest.raises(OutdatedCheckpointError) as exc_info:
            CheckpointFile(path).load()

        assert f"cryocast checkpoint upgrade {path}" in str(exc_info.value)
        assert isinstance(exc_info.value, pickle.UnpicklingError)

    def test_loads_upgraded_checkpoint(self, tmp_path: Path) -> None:
        """Load a checkpoint after upgrading it."""
        checkpoint_file = _save_legacy_checkpoint(
            tmp_path / "checkpoints" / "last.ckpt"
        )
        with pytest.raises(OutdatedCheckpointError):
            checkpoint_file.load()

        LegacyCheckpointFile(checkpoint_file).upgrade()

        assert checkpoint_file.load()["hyper_parameters"]["hemisphere"] == "north"


class TestMergeConfig:
    def test_returns_config_with_warning_when_missing(
        self,
        cfg_model_service: DictConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn and use the current config when there is no saved model_config.yaml."""
        path = tmp_path / "checkpoints" / "last.ckpt"
        path.parent.mkdir()
        path.write_text("checkpoint")

        with caplog.at_level(
            logging.WARNING, logger="cryocast.model_service.checkpoints"
        ):
            combined = CheckpointFile(path).merge_config(cfg_model_service)

        assert combined is cfg_model_service
        assert "Could not load the checkpoint configuration" in caplog.text

    def test_refuses_legacy_predict_config(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Suggest upgrading configs that predate the 'variables'/'window' split."""
        legacy_config = cfg_model_service.copy()
        legacy_config["predict"] = {"target": {"group_name": "sic-ssmis"}}
        checkpoint_file = _save_run_config(tmp_path, legacy_config)

        with pytest.raises(
            OutdatedCheckpointError, match="legacy 'predict' key"
        ) as exc:
            checkpoint_file.merge_config(cfg_model_service)

        assert f"cryocast checkpoint upgrade {checkpoint_file.path}" in str(exc.value)

    def test_model_and_window_from_checkpoint_and_the_rest_from_config(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """The checkpoint describes the model; everything else is from the config."""
        config = cfg_model_service.copy()
        config["model"]["name"] = "will_not_overwrite"
        config["window"]["n_history_steps"] = 99
        config["reporting"]["loggers"] = "will_overwrite"
        config["train"]["trainer"] = {"max_epochs": 200}

        checkpoint_file = _save_run_config(tmp_path, cfg_model_service)

        combined = checkpoint_file.merge_config(config)

        assert combined["model"]["name"] == "mock-model"
        assert combined["window"]["n_history_steps"] == 3
        assert combined["reporting"]["loggers"] == "will_overwrite"
        assert combined["train"]["trainer"]["max_epochs"] == 200

    def test_different_model_replaced_not_merged(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """A different configured model's settings do not leak into the checkpoint's."""
        ckpt_config = cfg_model_service.copy()
        ckpt_config["model"] = {"name": "dc-gsta-dc", "depth": 6}
        config = cfg_model_service.copy()
        config["model"] = {"name": "quick-test", "depth": 2, "extra": 1}

        checkpoint_file = _save_run_config(tmp_path, ckpt_config)

        combined = checkpoint_file.merge_config(config)

        assert combined["model"] == ckpt_config["model"]

    def test_same_model_keeps_configured_extra_keys(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """The same model keeps configured keys that the checkpoint config lacks."""
        ckpt_config = cfg_model_service.copy()
        ckpt_config["model"] = {"name": "dc-gsta-dc", "depth": 6}
        config = cfg_model_service.copy()
        config["model"] = {"name": "dc-gsta-dc", "depth": 2, "extra": 1}

        checkpoint_file = _save_run_config(tmp_path, ckpt_config)

        combined = checkpoint_file.merge_config(config)

        assert combined["model"] == {"name": "dc-gsta-dc", "depth": 6, "extra": 1}

    def test_variables_replaced_not_merged(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
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

        checkpoint_file = _save_run_config(tmp_path, ckpt_config)

        combined = checkpoint_file.merge_config(config)

        assert combined["variables"] == ckpt_config["variables"]

    def test_batch_size_from_config(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Batch size comes from the config; other window keys do not."""
        config = cfg_model_service.copy()
        config["window"] = {
            "batch_size": 1,
            "n_forecast_steps": 7,
            "n_history_steps": 1,
        }

        checkpoint_file = _save_run_config(tmp_path, cfg_model_service)

        combined = checkpoint_file.merge_config(config)

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
        tmp_path: Path,
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

        checkpoint_file = _save_run_config(tmp_path, ckpt_config)

        with caplog.at_level(
            logging.WARNING, logger="cryocast.model_service.checkpoints"
        ):
            checkpoint_file.merge_config(config)

        for message in expected:
            assert message in caplog.text
        if not expected:
            assert not caplog.records

    def test_summarises_different_models_in_one_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Report a different model in one line rather than every differing value."""
        ckpt_config = DictConfig({"model": {"name": "dc-gsta-dc", "depth": 6}})
        config = DictConfig({"model": {"name": "quick-test", "depth": 2}})

        checkpoint_file = _save_run_config(tmp_path, ckpt_config)

        with caplog.at_level(
            logging.WARNING, logger="cryocast.model_service.checkpoints"
        ):
            checkpoint_file.merge_config(config)

        assert len(caplog.records) == 1
        assert (
            "Using the 'dc-gsta-dc' model from the checkpoint rather than the "
            "configured 'quick-test' model." in caplog.text
        )


def _save_legacy_checkpoint(path: Path) -> CheckpointFile:
    """Save a legacy checkpoint, with enums, paths, DataSpaces and OmegaConf objects."""

    class Hemisphere(StrEnum):
        NORTH = "north"

    class DataSpace:
        """A DataSpace with public attributes."""

        def __init__(self, channels: int, name: str, shape: tuple[int, int]) -> None:
            self.channels = channels
            self.name = name
            self.shape = shape

    # Pickle these as icenet_mp.types.X classes
    Hemisphere.__module__ = "icenet_mp.types.enums"
    Hemisphere.__qualname__ = "Hemisphere"
    DataSpace.__module__ = "icenet_mp.types.complex_datatypes"
    DataSpace.__qualname__ = "DataSpace"
    module = ModuleType("icenet_mp")
    module.Hemisphere = Hemisphere  # type: ignore[attr-defined]
    module.DataSpace = DataSpace  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        for name in (
            "icenet_mp",
            "icenet_mp.types",
            "icenet_mp.types.enums",
            "icenet_mp.types.complex_datatypes",
        ):
            mp.setitem(sys.modules, name, module)
        torch.save(
            {
                "callbacks": {"ModelCheckpoint": {"dirpath": Path("/run/checkpoints")}},
                "hyper_parameters": {
                    "hemisphere": Hemisphere.NORTH,
                    "loss": DictConfig({"_target_": "icenet_mp.losses.MAELoss"}),
                    "output_space": DataSpace(1, "sic", (432, 432)),
                },
                "state_dict": {"weight": torch.ones(2)},
            },
            path,
        )
    return CheckpointFile(path)


def _save_legacy_config(checkpoint_file: CheckpointFile, text: str) -> None:
    """Save a run's model_config.yaml as text, as an older version would have done."""
    checkpoint_file.config_path.parent.mkdir(exist_ok=True)
    checkpoint_file.config_path.write_text(text)


class TestUpgrade:
    def test_upgraded_checkpoint_loads_with_weights_only(self, tmp_path: Path) -> None:
        """Upgrade to plain types that reference cryocast, keeping a backup."""
        checkpoint_file = _save_legacy_checkpoint(
            tmp_path / "checkpoints" / "last.ckpt"
        )

        LegacyCheckpointFile(checkpoint_file).upgrade()

        checkpoint = torch.load(checkpoint_file.path, weights_only=True)
        assert (
            checkpoint["callbacks"]["ModelCheckpoint"]["dirpath"] == "/run/checkpoints"
        )
        assert checkpoint["hyper_parameters"] == {
            "hemisphere": "north",
            "loss": {"_target_": "cryocast.losses.MAELoss"},
            "output_space": {"channels": 1, "name": "sic", "shape": [432, 432]},
        }
        assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(2))
        assert (tmp_path / "checkpoints" / "last.ckpt.bak").is_file()

    def test_upgrades_legacy_package_in_config(self, tmp_path: Path) -> None:
        """Rewrite icenet_mp references in the run's model config, keeping a backup."""
        checkpoint_file = _save_legacy_checkpoint(
            tmp_path / "checkpoints" / "last.ckpt"
        )
        legacy_config = "model:\n  _target_: icenet_mp.models.EncodeProcessDecode\n"
        _save_legacy_config(checkpoint_file, legacy_config)

        LegacyCheckpointFile(checkpoint_file).upgrade()

        assert OmegaConf.load(checkpoint_file.config_path) == {
            "model": {"_target_": "cryocast.models.EncodeProcessDecode"}
        }
        assert (tmp_path / "files" / "model_config.yaml.bak").read_text() == (
            legacy_config
        )

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
        tmp_path: Path,
    ) -> None:
        """Replace the 'predict' config with 'variables' and 'window' configs."""
        checkpoint_file = _save_legacy_checkpoint(
            tmp_path / "checkpoints" / "last.ckpt"
        )
        legacy_config = {
            "model": {"name": "dc-gsta-dc"},
            "predict": {
                "target": legacy_target,
                "n_forecast_steps": 7,
                "n_history_steps": 4,
            },
        }
        _save_legacy_config(checkpoint_file, OmegaConf.to_yaml(legacy_config))

        LegacyCheckpointFile(checkpoint_file).upgrade()

        assert OmegaConf.load(checkpoint_file.config_path) == {
            "model": {"name": "dc-gsta-dc"},
            # Legacy checkpoints used every variable from every dataset as input
            "variables": {"input": {}, "target": expected_target},
            "window": {"n_forecast_steps": 7, "n_history_steps": 4},
        }

    def test_upgrades_several_checkpoints_from_one_run(self, tmp_path: Path) -> None:
        """Upgrade every checkpoint in a run, with its model config upgraded once."""
        checkpoint_files = [
            _save_legacy_checkpoint(tmp_path / "checkpoints" / name)
            for name in ("epoch=1.ckpt", "last.ckpt")
        ]
        legacy_config = "model:\n  _target_: icenet_mp.models.Persistence\n"
        _save_legacy_config(checkpoint_files[0], legacy_config)

        for checkpoint_file in checkpoint_files:
            LegacyCheckpointFile(checkpoint_file).upgrade()

        for checkpoint_file in checkpoint_files:
            assert not torch.serialization.get_unsafe_globals_in_checkpoint(
                checkpoint_file.path
            )
        assert "icenet_mp" not in checkpoint_files[0].config_path.read_text()
        assert (tmp_path / "files" / "model_config.yaml.bak").read_text() == (
            legacy_config
        )

    def test_upgrading_twice_changes_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leave files that are already up to date unchanged."""
        checkpoint_file = _save_legacy_checkpoint(
            tmp_path / "checkpoints" / "last.ckpt"
        )
        _save_legacy_config(
            checkpoint_file, "predict:\n  target:\n    group_name: sic\n"
        )
        LegacyCheckpointFile(checkpoint_file).upgrade()
        upgraded = [
            path.read_bytes()
            for path in (checkpoint_file.path, checkpoint_file.config_path)
        ]

        with caplog.at_level(logging.INFO, logger="cryocast.model_service.checkpoints"):
            LegacyCheckpointFile(checkpoint_file).upgrade()

        assert [
            path.read_bytes()
            for path in (checkpoint_file.path, checkpoint_file.config_path)
        ] == upgraded
        assert caplog.text.count("is already up to date") == 2
        assert not list(tmp_path.rglob("*.tmp"))

    def test_finishes_partly_upgraded_run_keeping_original_backup(
        self, tmp_path: Path
    ) -> None:
        """Upgrade a config whose checkpoint is current, keeping the earlier backup."""
        path = tmp_path / "checkpoints" / "last.ckpt"
        path.parent.mkdir()
        torch.save({"state_dict": {"weight": torch.ones(2)}}, path)
        checkpoint_file = CheckpointFile(path)
        original_config = "predict:\n  target:\n    group_name: icenet_mp.sic\n"
        partly_upgraded_config = "predict:\n  target:\n    group_name: sic\n"
        _save_legacy_config(checkpoint_file, partly_upgraded_config)
        backup_path = tmp_path / "files" / "model_config.yaml.bak"
        backup_path.write_text(original_config)
        checkpoint = path.read_bytes()

        LegacyCheckpointFile(checkpoint_file).upgrade()

        assert "predict" not in OmegaConf.load(checkpoint_file.config_path)
        assert backup_path.read_text() == original_config
        assert path.read_bytes() == checkpoint
        assert not (tmp_path / "checkpoints" / "last.ckpt.bak").exists()
