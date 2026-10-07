import logging
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from cryocast.callbacks import MediaLoggingCallback, PredictionWriter
from cryocast.model_service import ModelService
from cryocast.models import EncodeProcessDecode
from cryocast.models.multistage import DecoderStage, EncoderStage, ProcessorStage
from cryocast.types import DataSpace

MAE_METRIC_CFG = {"name": "mae", "_target_": "cryocast.metrics.MAEPerForecastDay"}


class FakeCommonDataModule:
    def __init__(self, config: DictConfig) -> None:
        """Mock CommonDataModule."""
        self.config = config
        self.hemisphere = "north"
        self.input_spaces = [DataSpace(5, "input", (20, 20))]
        self.latitudes = {"input": [0.0] * 400}
        self.longitudes = {"input": [0.0] * 400}
        self.mask_directory = Path("nonexistent")
        self.n_forecast_steps = 2
        self.n_history_steps = 3
        self.output_space = DataSpace(1, "output", (10, 10))
        self.target_variable_indices = [0]
        self.target_variables = ["mock_var"]


class FakeModel:
    ignored_hparams: ClassVar[frozenset[str]] = frozenset()
    metrics: list[dict[str, Any]] | None = None
    # Matches the data described by FakeCommonDataModule
    input_spaces: ClassVar[list[DataSpace]] = [DataSpace(5, "input", (20, 20))]
    n_forecast_steps = 2
    n_history_steps = 3
    output_space = DataSpace(1, "output", (10, 10))

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        mask_dir: str | None = None,
        latitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        longitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        map_location: str | None = None,
        metrics: list[dict[str, Any]] | None = None,
        weights_only: bool = False,
    ) -> "FakeModel":
        del (
            mask_dir,
            checkpoint_path,
            latitudes_fn,
            longitudes_fn,
            map_location,
            weights_only,
        )
        model = cls()
        model.metrics = metrics
        return model


def _build_pretrained_processor(mask_dir: Path) -> tuple[ModelService, ProcessorStage]:
    """Build a real full model and a pretrained ProcessorStage to hand over from.

    The CNNEncoder carries BatchNorm buffers, so the target encoder is stateful. As the
    target dataset is also an input, the DiffusionProcessor's latent loss uses the
    input encoder rather than the target encoder, which is only handed over here.
    """
    space = DictConfig({"name": "sic", "channels": 1, "shape": [16, 16]})
    encoder = DictConfig(
        {
            "_target_": "cryocast.models.encoders.CNNEncoder",
            "n_layers": 1,
            "n_subblocks": 1,
            "activation": "LeakyReLU",
        }
    )
    decoder = DictConfig({"_target_": "cryocast.models.decoders.NaiveLinearDecoder"})
    processor = DictConfig(
        {
            "_target_": "cryocast.models.processors.DiffusionProcessor",
            "timesteps": 2,
            "start_out_channels": 8,
            "time_embed_dim": 256,
            "normalization": "none",
            "dropout_rate": 0.0,
            "loss": {"_target_": "torch.nn.MSELoss"},
        }
    )
    model = EncodeProcessDecode(
        encoders=DictConfig({"latent_space": [16, 16], "sic": encoder}),
        processor=processor,
        decoder=decoder,
        target_variable_indices=[0],
        hemisphere="north",
        input_spaces=[space],
        output_space=space,
        mask_dir=str(mask_dir),
        n_history_steps=2,
        n_forecast_steps=2,
        name="finetune-handover",
        metrics=[],
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
    )
    input_stage = EncoderStage.from_template(
        channel_names=["sic"],
        data_space_in=model.input_spaces[0],
        dataset="sic",
        decoder=decoder,
        encoder=encoder,
        template=model,
    )
    target_stage = EncoderStage.from_template(
        channel_names=["sic"],
        data_space_in=model.target_encoder.data_space_in,
        dataset="target",
        decoder=decoder,
        encoder=encoder,
        template=model,
    )
    decoder_stage = DecoderStage.from_template(
        decoder=decoder,
        encoders=[input_stage],
        output_space=model.output_space,
        target_dataset_name="sic",
        target_variable_indices=[0],
    )
    pretrained = ProcessorStage.from_template(
        processor=processor,
        decoder_model=decoder_stage,
        target_encoder=target_stage,
    )
    # Stand in for checkpoint contents without running an optimisation step
    with torch.no_grad():
        for parameter in pretrained.target_encoder.parameters():
            parameter.add_(0.5)
        for buffer in pretrained.target_encoder.buffers():
            buffer.add_(2)
    service = ModelService.__new__(ModelService)
    service.model_ = model
    return service, pretrained


class TestModelService:
    def test_init_seeds_when_random_seed_configured(self) -> None:
        """Seed global RNGs and set reproducibility env vars when a seed is set."""
        config = DictConfig({"random": {"seed": 42}})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "environ", os.environ.copy())
            mock_seed_everything = MagicMock()
            mp.setattr("cryocast.model_service.seed_everything", mock_seed_everything)
            mp.setattr("cryocast.model_service.patch_open_file_limit", MagicMock())
            mp.setattr(
                "cryocast.model_service.patch_interpolate_antialias", MagicMock()
            )
            service = ModelService(config)

            assert os.environ["PYTHONHASHSEED"] == "42"
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

        mock_seed_everything.assert_called_once_with(42, workers=True)
        assert service.config == config

    def test_from_config_loads_model(self, cfg_model_service: DictConfig) -> None:
        mock_instantiate = MagicMock()
        mock_instantiate.return_value = FakeModel()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.instantiate", mock_instantiate
            )
            service = ModelService.from_config(cfg_model_service)
            assert isinstance(service.model, FakeModel)

        args, kwargs = mock_instantiate.call_args
        assert args[0] is cfg_model_service["model"]
        assert kwargs["input_spaces"] == [DataSpace(5, "input", (20, 20)).to_dict()]
        assert kwargs["output_space"] == DataSpace(1, "output", (10, 10)).to_dict()
        assert kwargs["n_forecast_steps"] == 2
        assert kwargs["n_history_steps"] == 3
        assert kwargs["metrics"] == cfg_model_service["reporting"]["metrics"]
        assert kwargs["optimizer"] is cfg_model_service["train"]["optimizer"]
        assert kwargs["scheduler"] is cfg_model_service["train"]["scheduler"]
        assert kwargs["lr_scheduler"] is cfg_model_service["train"]["lr_scheduler"]
        assert kwargs["_recursive_"] is False
        assert kwargs["_convert_"] == "object"

    def test_from_checkpoint_loads_model(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        # Generate a checkpoint file and corresponding model_config.yaml
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(cfg_model_service, files_dir / "model_config.yaml")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr(
                "cryocast.model_service.torch.load", lambda *_a, **_k: {"epoch": 3}
            )
            service = ModelService.from_checkpoint(DictConfig({}), checkpoint_path)
            assert isinstance(service.model, FakeModel)
            assert service.config == cfg_model_service
            assert service.model.checkpoint_epoch == 3

    def test_from_checkpoint_sets_checkpoint_epoch_to_none_when_absent(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Don't crash when the raw checkpoint dict has no 'epoch' key."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(cfg_model_service, files_dir / "model_config.yaml")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr("cryocast.model_service.torch.load", lambda *_a, **_k: {})
            service = ModelService.from_checkpoint(DictConfig({}), checkpoint_path)

        assert service.model.checkpoint_epoch is None

    def test_from_checkpoint_config_overloads(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        # Generate a checkpoint file and corresponding model_config.yaml
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(cfg_model_service, files_dir / "model_config.yaml")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr(
                "cryocast.model_service.torch.load", lambda *_a, **_k: {"epoch": 3}
            )
            service = ModelService.from_checkpoint(
                DictConfig(
                    {
                        "reporting": {"loggers": "will_overwrite"},
                        "model": {"name": "will_not_overwrite"},
                    }
                ),
                checkpoint_path,
            )
            assert isinstance(service.model, FakeModel)

            expected_config = cfg_model_service.copy()
            expected_config["reporting"]["loggers"] = "will_overwrite"
            assert service.config == expected_config
            assert service.config["model"]["name"] != "will_not_overwrite"

    @pytest.mark.parametrize("test_from_dir", [False, True], ids=["file", "dir"])
    def test_from_checkpoint_config_precedence(
        self, cfg_model_service: DictConfig, tmp_path: Path, *, test_from_dir: bool
    ) -> None:
        """Take the model description from the checkpoint and the rest from config."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "last.ckpt"
        checkpoint_path.write_text("checkpoint")
        (checkpoints_dir / "epoch=3-step=10.ckpt").write_text("checkpoint")

        ckpt_config = cfg_model_service.copy()
        ckpt_config["train"]["trainer"] = {"max_epochs": 10}
        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(ckpt_config, files_dir / "model_config.yaml")

        current_config = cfg_model_service.copy()
        current_config["train"]["trainer"] = {"max_epochs": 200}
        current_config["model"]["name"] = "will_not_overwrite"
        current_config["window"]["n_history_steps"] = 99

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr("cryocast.model_service.torch.load", lambda *_a, **_k: {})
            service = ModelService.from_checkpoint(
                current_config, checkpoints_dir if test_from_dir else checkpoint_path
            )

        assert service.config["train"]["trainer"]["max_epochs"] == 200
        assert service.config["model"]["name"] == "mock-model"
        assert service.config["window"]["n_history_steps"] == 3
        assert service.checkpoint_path == checkpoint_path

    def test_from_checkpoint_raises_without_last_checkpoint_in_dir(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Reject a checkpoint directory with no resumable ``last*.ckpt`` file."""
        (tmp_path / "epoch=3-step=10.ckpt").write_text("checkpoint")

        with pytest.raises(FileNotFoundError, match=r"last\*.ckpt"):
            ModelService.from_checkpoint(cfg_model_service, tmp_path)

    def test_from_checkpoint_variables_replaced_not_merged(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Checkpoint variables win outright rather than being unioned with the CLI."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        ckpt_variables = {
            "input": {"sic-ssmis": ["ice_conc"]},
            "target": {"sic-ssmis": ["ice_conc"]},
        }
        ckpt_config = cfg_model_service.copy()
        ckpt_config["variables"] = ckpt_variables
        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(ckpt_config, files_dir / "model_config.yaml")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr(
                "cryocast.model_service.torch.load", lambda *_a, **_k: {"epoch": 3}
            )
            service = ModelService.from_checkpoint(
                DictConfig(
                    {
                        "variables": {
                            "input": {
                                "era5": ["2t"],
                                "float-argo": ["TEMP"],
                                "sic-osisaf": ["ice_conc"],
                            },
                            "target": {"sic-osisaf": ["ice_conc"]},
                        }
                    }
                ),
                checkpoint_path,
            )

        assert OmegaConf.to_container(service.config["variables"]) == ckpt_variables

    def test_from_checkpoint_batch_size_from_current_config(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Batch size comes from the current config; other window keys do not."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(cfg_model_service, files_dir / "model_config.yaml")

        cli_config = cfg_model_service.copy()
        cli_config["window"] = {
            "batch_size": 1,
            "n_forecast_steps": 7,
            "n_history_steps": 1,
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr(
                "cryocast.model_service.torch.load", lambda *_a, **_k: {"epoch": 3}
            )
            service = ModelService.from_checkpoint(cli_config, checkpoint_path)

        assert OmegaConf.to_container(service.config["window"]) == {
            "batch_size": 1,
            "n_forecast_steps": 2,
            "n_history_steps": 3,
        }

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
    def test_from_checkpoint_translates_legacy_predict_config(
        self,
        legacy_target: dict[str, Any],
        expected_target: dict[str, Any],
        cfg_model_service: DictConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Translate checkpoints whose config predates the 'variables'/'window' split."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        ckpt_config = cfg_model_service.copy()
        del ckpt_config["variables"]
        del ckpt_config["window"]
        ckpt_config["predict"] = {
            "target": legacy_target,
            "n_forecast_steps": 7,
            "n_history_steps": 4,
        }
        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(ckpt_config, files_dir / "model_config.yaml")

        with (
            pytest.MonkeyPatch.context() as mp,
            caplog.at_level(logging.WARNING, logger="cryocast.model_service"),
        ):
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr(
                "cryocast.model_service.torch.load", lambda *_a, **_k: {"epoch": 3}
            )
            service = ModelService.from_checkpoint(cfg_model_service, checkpoint_path)

        assert "uses the legacy 'predict' key" in caplog.text
        assert "predict" not in service.config
        assert OmegaConf.to_container(service.config["variables"]) == {
            "input": {},
            "target": expected_target,
        }
        assert service.config["window"]["n_forecast_steps"] == 7
        assert service.config["window"]["n_history_steps"] == 4
        assert (
            service.config["window"]["batch_size"]
            == cfg_model_service["window"]["batch_size"]
        )

    def test_from_checkpoint_raises_when_checkpoint_missing(
        self, tmp_path: Path
    ) -> None:
        missing_path = tmp_path / "missing.ckpt"

        with pytest.raises(FileNotFoundError, match="Could not find checkpoint file"):
            ModelService.from_checkpoint(DictConfig({}), missing_path)

    @pytest.mark.parametrize(
        ("attribute", "value", "match"),
        [
            ("n_history_steps", 2, "n_history_steps is 3 in the model but 2"),
            ("n_forecast_steps", 5, "n_forecast_steps is 2 in the model but 5"),
            (
                "input_spaces",
                [DataSpace(4, "input", (20, 20))],
                "input 'input' is DataSpace\\(channels=5.* but DataSpace\\(channels=4",
            ),
            (
                "input_spaces",
                [DataSpace(5, "input", (20, 20)), DataSpace(2, "extra", (20, 20))],
                "input 'extra' is missing in the model",
            ),
            (
                "output_space",
                DataSpace(2, "output", (10, 10)),
                "output is DataSpace\\(channels=1.* but DataSpace\\(channels=2",
            ),
        ],
        ids=["history", "forecast", "input-channels", "extra-input", "output"],
    )
    def test_verify_model_matches_data_raises_on_mismatch(
        self, attribute: str, value: object, match: str
    ) -> None:
        """Report how the configured data differs from the checkpointed model."""
        data_module = FakeCommonDataModule(DictConfig({}))
        setattr(data_module, attribute, value)
        service = ModelService.__new__(ModelService)
        service.model_ = FakeModel()  # type: ignore[assignment]
        service.data_module_ = data_module  # type: ignore[assignment]

        with pytest.raises(ValueError, match=match):
            service._verify_model_matches_data()

    def test_verify_model_matches_data_ignores_input_order(self) -> None:
        """Inputs are matched by name, so their order does not matter."""
        spaces = [DataSpace(5, "a", (20, 20)), DataSpace(2, "b", (20, 20))]
        model = FakeModel()
        model.input_spaces = spaces  # type: ignore[misc]
        data_module = FakeCommonDataModule(DictConfig({}))
        data_module.input_spaces = spaces[::-1]
        service = ModelService.__new__(ModelService)
        service.model_ = model  # type: ignore[assignment]
        service.data_module_ = data_module  # type: ignore[assignment]

        service._verify_model_matches_data()

    def test_from_checkpoint_falls_back_to_provided_config_when_ckpt_config_missing(
        self,
        cfg_model_service: DictConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn and use the current config when there is no saved model_config.yaml."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")
        # Deliberately do not create a "files/model_config.yaml" alongside it.

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.CommonDataModule", FakeCommonDataModule)
            mp.setattr(
                "cryocast.model_service.hydra.utils.get_class",
                lambda _target: FakeModel,
            )
            mp.setattr(
                "cryocast.model_service.torch.load", lambda *_a, **_k: {"epoch": 3}
            )
            with caplog.at_level(logging.WARNING, logger="cryocast.model_service"):
                service = ModelService.from_checkpoint(
                    cfg_model_service, checkpoint_path
                )

        assert isinstance(service.model, FakeModel)
        assert service.config == cfg_model_service
        assert "Could not load the checkpoint configuration" in caplog.text

    def test_build_run_directory_uses_wandb_sync_dir(self, tmp_path: Path) -> None:
        service = ModelService.__new__(ModelService)
        trainer = MagicMock()
        wandb_run = SimpleNamespace(
            _settings=SimpleNamespace(sync_dir=str(tmp_path / "wandb-run"))
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "cryocast.model_service.get_wandb_run", lambda _trainer: wandb_run
            )
            result = service.build_run_directory(trainer)

        assert result == tmp_path / "wandb-run"

    def test_build_run_directory_generates_local_path_without_wandb(
        self, tmp_path: Path
    ) -> None:
        service = ModelService.__new__(ModelService)
        service.data_module_ = MagicMock()
        service.data_module_.base_path = tmp_path
        trainer = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.model_service.get_wandb_run", lambda _trainer: None)
            mp.setattr(
                "cryocast.model_service.get_timestamp", lambda: "20260101-000000"
            )
            mp.setattr("cryocast.model_service.generate_id", lambda: "abc123")
            result = service.build_run_directory(trainer)

        expected = tmp_path / "training" / "local" / "run-20260101-000000-abc123"
        assert result == expected

    @pytest.mark.parametrize(
        ("deterministic_enabled", "warn_only_enabled", "match"),
        [(False, True, "deterministic"), (True, False, "warn_only")],
        ids=["deterministic-mismatch", "warn-only-missing"],
    )
    def test_build_trainer_raises_on_deterministic_config_errors(
        self, *, deterministic_enabled: bool, warn_only_enabled: bool, match: str
    ) -> None:
        service = ModelService.__new__(ModelService)
        service.fully_deterministic = True
        service.config_ = DictConfig({"unrelated": True})
        config = DictConfig({"trainer": {}})
        fake_trainer = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "cryocast.model_service.hydra.utils.instantiate",
                lambda *_a, **_k: fake_trainer,
            )
            mp.setattr(
                "cryocast.model_service.torch.are_deterministic_algorithms_enabled",
                lambda: deterministic_enabled,
            )
            mp.setattr(
                "cryocast.model_service.torch.is_deterministic_algorithms_warn_only_enabled",
                lambda: warn_only_enabled,
            )
            with pytest.raises(ValueError, match=match):
                service.build_trainer(config=config, project="train")

    def test_build_trainer_warns_when_no_callbacks_or_loggers(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = ModelService.__new__(ModelService)
        service.fully_deterministic = False
        service.model_ = MagicMock()
        service.data_module_ = MagicMock()
        service.config_ = DictConfig({"model": {"name": "test_model"}})
        config = DictConfig({"trainer": {}})

        fake_trainer = MagicMock()
        fake_trainer.callbacks = []
        fake_trainer.is_global_zero = True

        with (
            pytest.MonkeyPatch.context() as mp,
            caplog.at_level(logging.WARNING, logger="cryocast.model_service"),
        ):
            mp.setattr(
                "cryocast.model_service.hydra.utils.instantiate",
                lambda *_a, **_k: fake_trainer,
            )
            mp.setattr(
                "cryocast.model_service.torch.are_deterministic_algorithms_enabled",
                lambda: False,
            )
            mp.setattr("cryocast.model_service.suggested_max_num_workers", lambda _n: 1)
            mp.setattr(
                service, "build_run_directory", lambda _trainer: tmp_path / "run"
            )
            mp.setattr("cryocast.model_service.get_wandb_run", lambda _trainer: None)
            service.build_trainer(config=config, project="train")

        assert "No callbacks have been set" in caplog.text
        assert "No loggers have been set" in caplog.text

    def test_build_trainer_configures_run_directory_and_callbacks(
        self, tmp_path: Path
    ) -> None:
        """Wire up workers, the run directory, and per-callback prefix/dirpath."""
        service = ModelService.__new__(ModelService)
        service.fully_deterministic = False
        service.model_ = MagicMock()
        service.data_module_ = MagicMock()
        service.config_ = DictConfig({"model": {"name": "test_model"}})
        config = DictConfig({"trainer": {}})

        plotting_callback = MagicMock(spec=MediaLoggingCallback)
        checkpoint_callback = MagicMock(spec=ModelCheckpoint)
        enabled_prediction_writer = PredictionWriter(enabled=True)
        disabled_prediction_writer = PredictionWriter(enabled=False)

        fake_trainer = MagicMock()
        fake_trainer.callbacks = [
            plotting_callback,
            checkpoint_callback,
            enabled_prediction_writer,
            disabled_prediction_writer,
        ]
        fake_trainer.num_devices = 1
        fake_trainer.is_global_zero = True

        run_dir = tmp_path / "run"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "cryocast.model_service.hydra.utils.instantiate",
                lambda *_a, **_k: fake_trainer,
            )
            mp.setattr(
                "cryocast.model_service.torch.are_deterministic_algorithms_enabled",
                lambda: False,
            )
            mp.setattr("cryocast.model_service.suggested_max_num_workers", lambda _n: 4)
            mp.setattr(service, "build_run_directory", lambda _trainer: run_dir)
            mp.setattr("cryocast.model_service.get_wandb_run", lambda _trainer: None)
            result = service.build_trainer(
                config=config, project="train", job_stage="processor"
            )

        service.data_module_.assign_workers.assert_called_once_with(4)
        assert (run_dir / "files" / "model_config.yaml").exists()
        assert plotting_callback.prefix == "processor"
        assert checkpoint_callback.dirpath == str(run_dir / "checkpoints")
        assert (
            enabled_prediction_writer.output_path
            == run_dir / "files" / "predictions.nc"
        )
        assert enabled_prediction_writer.mask_dir == service.data_module_.mask_directory
        assert disabled_prediction_writer.output_path is None
        assert disabled_prediction_writer.mask_dir is None
        assert result is fake_trainer

    def test_build_trainer_wires_wandb_logger_and_saves_config_to_wandb(
        self, tmp_path: Path
    ) -> None:
        """Instantiate a WandbLogger with job_type/project and push the config to W&B."""
        service = ModelService.__new__(ModelService)
        service.fully_deterministic = False
        service.model_ = MagicMock()
        service.data_module_ = MagicMock()
        service.config_ = DictConfig(
            {
                "model": {"name": "test_model"},
                "reporting": {
                    "loggers": {
                        "wandb": {"_target_": "lightning.pytorch.loggers.WandbLogger"},
                        "csv": {"_target_": "lightning.pytorch.loggers.CSVLogger"},
                    },
                },
            }
        )
        config = DictConfig({"trainer": {}})

        fake_trainer = MagicMock()
        fake_trainer.callbacks = []
        fake_trainer.num_devices = 1
        fake_trainer.is_global_zero = True

        run_dir = tmp_path / "run"
        wandb_logger = MagicMock()
        csv_logger = MagicMock()
        wandb_run = MagicMock()

        def fake_instantiate(cfg: DictConfig, **_kwargs: object) -> MagicMock:
            target = cfg.get("_target_", "")
            if target.endswith("WandbLogger"):
                return wandb_logger
            if target.endswith("CSVLogger"):
                return csv_logger
            return fake_trainer

        with pytest.MonkeyPatch.context() as mp:
            mock_instantiate = MagicMock(side_effect=fake_instantiate)
            mp.setattr(
                "cryocast.model_service.hydra.utils.instantiate", mock_instantiate
            )
            mp.setattr(
                "cryocast.model_service.torch.are_deterministic_algorithms_enabled",
                lambda: False,
            )
            mp.setattr("cryocast.model_service.suggested_max_num_workers", lambda _n: 1)
            mp.setattr(service, "build_run_directory", lambda _trainer: run_dir)
            mp.setattr(
                "cryocast.model_service.get_wandb_run", lambda _trainer: wandb_run
            )
            service.build_trainer(config=config, project="train")

        wandb_call = next(
            call
            for call in mock_instantiate.call_args_list
            if call.args[0].get("_target_", "").endswith("WandbLogger")
        )
        assert wandb_call.kwargs == {
            "job_type": "single-stage",
            "project": "train",
            "_convert_": "all",
        }

        csv_call = next(
            call
            for call in mock_instantiate.call_args_list
            if call.args[0].get("_target_", "").endswith("CSVLogger")
        )
        assert csv_call.kwargs == {}

        model_config_path = run_dir / "files" / "model_config.yaml"
        wandb_run.save.assert_called_once_with(
            model_config_path, base_path=model_config_path.parent, policy="now"
        )
        defined_metrics = {c.args[0] for c in wandb_run.define_metric.call_args_list}
        assert defined_metrics == {"train_loss", "validation_loss", "test_loss"}

    def test_config_merging_applies_stage_overrides(self) -> None:
        """Merge stage-specific values over the common training configuration."""
        service = ModelService.__new__(ModelService)
        service.config_ = OmegaConf.create(
            {
                "train": {
                    "optimizer": {"lr": 0.001, "weight_decay": 0.01},
                    "trainer": {"max_epochs": 20, "accelerator": "auto"},
                    "multistage": {
                        "processor": {
                            "optimizer": {"lr": 0.01},
                            "trainer": {"max_epochs": 3},
                        }
                    },
                }
            }
        )

        merged = service._merged_config("processor")

        assert merged["optimizer"]["lr"] == 0.01
        assert merged["optimizer"]["weight_decay"] == 0.01
        assert merged["trainer"]["max_epochs"] == 3
        assert merged["trainer"]["accelerator"] == "auto"

    def test_config_raises_when_not_initialised(self) -> None:
        service = ModelService.__new__(ModelService)
        service.config_ = DictConfig({})

        with pytest.raises(AttributeError, match="config"):
            _ = service.config

    def test_data_module_lazily_constructs_and_caches(
        self, cfg_model_service: DictConfig
    ) -> None:
        """Build the data module once from config, then reuse the cached instance."""
        service = ModelService.__new__(ModelService)
        service.config_ = cfg_model_service
        service.data_module_ = None

        with pytest.MonkeyPatch.context() as mp:
            mock_data_module_cls = MagicMock(
                return_value=FakeCommonDataModule(cfg_model_service)
            )
            mp.setattr("cryocast.model_service.CommonDataModule", mock_data_module_cls)
            first = service.data_module
            second = service.data_module

        assert first is second
        mock_data_module_cls.assert_called_once_with(cfg_model_service)

    def test_evaluate_runs_trainer_test(self) -> None:
        service = ModelService.__new__(ModelService)
        service.config_ = DictConfig({"evaluate": {"trainer": {}}})
        service.model_ = MagicMock()
        service.data_module_ = MagicMock()
        trainer = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mock_build_trainer = MagicMock(return_value=trainer)
            mp.setattr(service, "build_trainer", mock_build_trainer)
            service.evaluate()

        mock_build_trainer.assert_called_once_with(
            config=service.config_["evaluate"], project="evaluate"
        )
        trainer.test.assert_called_once_with(
            model=service.model_, datamodule=service.data_module_
        )

    @pytest.mark.parametrize(
        "include_loss", [True, False], ids=["with-loss", "no-loss"]
    )
    def test_fit_configures_model_and_runs_trainer_fit(
        self, *, include_loss: bool
    ) -> None:
        """Apply the training config to the model and delegate to trainer.fit()."""
        service = ModelService.__new__(ModelService)
        model = MagicMock()
        service.model_ = model
        service.data_module_ = MagicMock()
        config_dict = {
            "optimizer": "optimizer_cfg",
            "scheduler": "scheduler_cfg",
            "lr_scheduler": "lr_scheduler_cfg",
        }
        if include_loss:
            config_dict["loss"] = "loss_cfg"
        config = DictConfig(config_dict)

        trainer = MagicMock()
        trainer.max_epochs = 5
        trainer.num_devices = 1
        ckpt_path = Path("ckpt.ckpt")

        with pytest.MonkeyPatch.context() as mp:
            mock_build_trainer = MagicMock(return_value=trainer)
            mp.setattr(service, "build_trainer", mock_build_trainer)
            mp.setattr("cryocast.model_service.torch.cuda.is_available", lambda: False)
            mp.setattr("cryocast.model_service.torch.mps.is_available", lambda: False)
            mp.setattr("cryocast.model_service.torch.xpu.is_available", lambda: False)
            result = service._fit(
                config=config, job_stage="processor", ckpt_path=ckpt_path
            )

        assert model.optimizer_cfg == "optimizer_cfg"
        assert model.scheduler_cfg == "scheduler_cfg"
        assert model.lr_scheduler_cfg == "lr_scheduler_cfg"
        if include_loss:
            assert model.loss_cfg == "loss_cfg"
        mock_build_trainer.assert_called_once_with(
            config=config, job_stage="processor", project="train"
        )
        trainer.fit.assert_called_once_with(
            model=model,
            datamodule=service.data_module_,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        assert result is trainer

    def test_fit_clears_device_caches_when_available(self) -> None:
        """Release cached device memory on every backend that reports itself available."""
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.data_module_ = MagicMock()
        config = DictConfig({"optimizer": "o", "scheduler": "s", "lr_scheduler": "l"})
        trainer = MagicMock()
        trainer.max_epochs = 1
        trainer.num_devices = 1

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "build_trainer", MagicMock(return_value=trainer))
            mp.setattr("cryocast.model_service.torch.cuda.is_available", lambda: True)
            mp.setattr("cryocast.model_service.torch.mps.is_available", lambda: True)
            mp.setattr("cryocast.model_service.torch.xpu.is_available", lambda: True)
            mock_cuda_empty = MagicMock()
            mock_mps_empty = MagicMock()
            mock_xpu_empty = MagicMock()
            mp.setattr("cryocast.model_service.torch.cuda.empty_cache", mock_cuda_empty)
            mp.setattr("cryocast.model_service.torch.mps.empty_cache", mock_mps_empty)
            mp.setattr("cryocast.model_service.torch.xpu.empty_cache", mock_xpu_empty)
            service._fit(config=config)

        mock_cuda_empty.assert_called_once_with()
        mock_mps_empty.assert_called_once_with()
        mock_xpu_empty.assert_called_once_with()

    def test_train_standard_mode_rejects_model_requiring_multistage(self) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.model_.multistage_only = True

        with pytest.raises(ValueError, match="multistage"):
            service.train()

    def test_train_standard_mode_allows_model_not_requiring_multistage(
        self,
    ) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.model_.multistage_only = False
        service.config_ = DictConfig({"train": "train_config"})
        service.checkpoint_path = None

        with pytest.MonkeyPatch.context() as mp:
            mock_fit = MagicMock()
            mp.setattr(service, "_fit", mock_fit)
            service.train()

        mock_fit.assert_called_once_with(config="train_config", ckpt_path=None)

    def test_save_stage_checkpoint_saves_when_no_best_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Save a deterministic stage checkpoint when callbacks have no best path."""
        service = ModelService.__new__(ModelService)
        trainer = MagicMock()
        trainer.current_epoch = 4
        trainer.global_step = 17
        trainer.checkpoint_callbacks = []
        trainer.is_global_zero = True
        run_dir = tmp_path / "run"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "build_run_directory", lambda _trainer: run_dir)
            result = service._save_stage_checkpoint(trainer, "processor")

        expected = run_dir / "checkpoints" / "processor.epoch=4-step=17.ckpt"
        trainer.save_checkpoint.assert_called_once_with(expected, weights_only=False)
        assert result == expected

    def test_save_stage_checkpoint_moves_single_best_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Move one callback-selected best checkpoint to the stage checkpoint path."""
        service = ModelService.__new__(ModelService)
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        best_path = tmp_path / "best.ckpt"
        best_path.write_text("checkpoint")

        trainer = MagicMock()
        trainer.current_epoch = 2
        trainer.global_step = 8
        trainer.is_global_zero = True
        trainer.checkpoint_callbacks = [SimpleNamespace(best_model_path=str(best_path))]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "build_run_directory", lambda _trainer: run_dir)
            result = service._save_stage_checkpoint(trainer, "decoder")

        expected = run_dir / "checkpoints" / "decoder.best.ckpt"
        assert result == expected
        assert expected.read_text() == "checkpoint"
        trainer.save_checkpoint.assert_not_called()
        trainer.strategy.barrier.assert_called_once_with()

    def test_save_stage_checkpoint_rejects_multiple_best_paths(
        self, tmp_path: Path
    ) -> None:
        """Reject ambiguous checkpoint selection from multiple callbacks."""
        service = ModelService.__new__(ModelService)
        trainer = MagicMock()
        trainer.current_epoch = 1
        trainer.global_step = 2
        trainer.checkpoint_callbacks = [
            SimpleNamespace(best_model_path=str(tmp_path / "a.ckpt")),
            SimpleNamespace(best_model_path=str(tmp_path / "b.ckpt")),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "build_run_directory", lambda _trainer: tmp_path)
            with pytest.raises(ValueError, match="2 checkpoints"):
                service._save_stage_checkpoint(trainer, "encoder")

    def test_train_resumes_from_loaded_checkpoint(self, tmp_path: Path) -> None:
        """Resume training from the checkpoint that the model was loaded from."""
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.model_.multistage_only = False
        service.config_ = DictConfig({"train": "train_config"})
        ckpt_path = tmp_path / "last.ckpt"
        service.checkpoint_path = ckpt_path

        with pytest.MonkeyPatch.context() as mp:
            mock_fit = MagicMock()
            mp.setattr(service, "_fit", mock_fit)
            service.train()

        mock_fit.assert_called_once_with(config="train_config", ckpt_path=ckpt_path)

    def test_train_multistage_rejects_non_encode_process_decode_model(self) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()

        with pytest.raises(TypeError, match="EncodeProcessDecode"):
            service.train_multistage()

    def test_train_multistage_orchestrates_all_stages(self, tmp_path: Path) -> None:
        """Chain encoders -> decoder -> processor -> finetune with merged configs."""
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock(spec=EncodeProcessDecode)
        target_encoder = MagicMock()
        trained_decoder = MagicMock()
        processor_model = MagicMock()
        final_trainer = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "_merged_config", lambda name: f"merged_{name}")
            mock_encoders = MagicMock(
                return_value=[MagicMock(), MagicMock(), target_encoder]
            )
            mock_decoder = MagicMock(return_value=trained_decoder)
            mock_processor = MagicMock(return_value=processor_model)
            mock_finetune = MagicMock(return_value=final_trainer)
            mp.setattr(service, "train_stage_encoders", mock_encoders)
            mp.setattr(service, "train_stage_decoder", mock_decoder)
            mp.setattr(service, "train_stage_processor", mock_processor)
            mp.setattr(service, "train_stage_finetune", mock_finetune)
            result = service.train_multistage(checkpoint_dir=tmp_path)

        mock_encoders.assert_called_once_with(
            train_cfg="merged_encoders", checkpoint_dir=tmp_path
        )
        trained_encoders = mock_encoders.return_value
        mock_decoder.assert_called_once_with(
            trained_encoders, train_cfg="merged_decoder", checkpoint_dir=tmp_path
        )
        mock_processor.assert_called_once_with(
            trained_decoder,
            train_cfg="merged_processor",
            checkpoint_dir=tmp_path,
            target_encoder=target_encoder,
        )
        mock_finetune.assert_called_once_with(
            processor_model=processor_model, train_cfg="merged_finetune"
        )
        assert result is final_trainer

    def test_train_stage_decoder_rejects_non_encode_process_decode_model(self) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()

        with pytest.raises(TypeError, match="EncodeProcessDecode"):
            service.train_stage_decoder([], train_cfg=DictConfig({}))

    def test_train_stage_decoder_trains_new_decoder(self, tmp_path: Path) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock(spec=EncodeProcessDecode)
        service.config_ = DictConfig({"model": {"decoder": {"foo": "bar"}}})
        service.data_module_ = MagicMock()
        service.data_module_.target_group_name = "target"
        service.data_module_.target_variable_indices = [0]
        service.data_module_.mask_directory = tmp_path
        encoder_models: list[EncoderStage] = [MagicMock()]

        decoder_model = MagicMock()

        trainer = MagicMock()
        ckpt_path = tmp_path / "decoder.ckpt"

        with pytest.MonkeyPatch.context() as mp:
            mock_from_template = MagicMock(return_value=decoder_model)
            mp.setattr(DecoderStage, "from_template", mock_from_template)
            mp.setattr(service, "_fit", MagicMock(return_value=trainer))
            mp.setattr(
                service, "_save_stage_checkpoint", MagicMock(return_value=ckpt_path)
            )
            mp.setattr(
                "cryocast.model_service.torch.load",
                lambda *_a, **_k: {"state_dict": "decoder_state"},
            )
            result = service.train_stage_decoder(
                encoder_models, train_cfg=DictConfig({"lr": 1})
            )

        mock_from_template.assert_called_once_with(
            decoder=service.config_["model"]["decoder"],
            encoders=encoder_models,
            output_space=service.data_module_.output_space,
            target_dataset_name="target",
            target_variable_indices=[0],
            mask_dir=str(tmp_path),
        )
        decoder_model.load_state_dict.assert_called_once_with("decoder_state")
        assert result is decoder_model

    def test_train_stage_decoder_loads_existing_checkpoint(
        self, tmp_path: Path
    ) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock(spec=EncodeProcessDecode)
        service.config_ = DictConfig(
            {
                "model": {"decoder": {"foo": "bar"}},
                "reporting": {"metrics": [MAE_METRIC_CFG]},
            }
        )
        service.data_module_ = MagicMock()
        service.data_module_.target_group_name = "target"
        service.data_module_.target_variable_indices = [0]
        service.data_module_.mask_directory = tmp_path
        encoder_models: list[EncoderStage] = [MagicMock()]
        checkpoint_path = tmp_path / "decoder.epoch=2-step=10.ckpt"
        checkpoint_path.write_text("checkpoint")
        loaded_decoder = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mock_load = MagicMock(return_value=loaded_decoder)
            mp.setattr(DecoderStage, "load_from_checkpoint", mock_load)
            result = service.train_stage_decoder(
                encoder_models, train_cfg=DictConfig({}), checkpoint_dir=tmp_path
            )

        mock_load.assert_called_once_with(
            checkpoint_path,
            map_location="cpu",
            metrics=[MAE_METRIC_CFG],
            weights_only=False,
            decoder=service.config_["model"]["decoder"],
            encoders=encoder_models,
            target_dataset_name="target",
            target_variable_indices=[0],
            mask_dir=str(tmp_path),
        )
        assert result is loaded_decoder

    def test_train_stage_encoders_rejects_non_encode_process_decode_model(self) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()

        with pytest.raises(TypeError, match="EncodeProcessDecode"):
            service.train_stage_encoders(train_cfg=DictConfig({}))

    def test_train_stage_encoders_trains_new_and_skips_via_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Train a fresh encoder while reusing an existing checkpoint for another."""
        service = ModelService.__new__(ModelService)
        encoder_era5 = SimpleNamespace(
            name="era5", data_space_in=DataSpace(3, "era5", (10, 10))
        )
        target_encoder = SimpleNamespace(name="target")
        service.model_ = MagicMock(spec=EncodeProcessDecode)
        service.model_.encoders = [encoder_era5]
        service.model_.target_encoder = target_encoder
        service.config_ = DictConfig(
            {
                "model": {
                    "decoder": {"foo": "bar"},
                    "encoders": {"era5": {"baz": "qux"}},
                },
                "reporting": {"metrics": [MAE_METRIC_CFG]},
            }
        )
        service.data_module_ = MagicMock()
        service.data_module_.target_group_name = "target"
        service.data_module_.target_variables = ["sic"]
        service.data_module_.datasets = {
            "era5": SimpleNamespace(variable_names=["t2m"])
        }
        service.data_module_.latitudes = {"input": [0.0]}
        service.data_module_.longitudes = {"input": [0.0]}

        checkpoint_path = tmp_path / "encoder-target.epoch=3-step=9.ckpt"
        checkpoint_path.write_text("checkpoint")

        trained_encoder_model = MagicMock()
        loaded_target_model = MagicMock()
        trainer = MagicMock()
        ckpt_path = tmp_path / "encoder-era5.ckpt"

        with pytest.MonkeyPatch.context() as mp:
            mock_from_template = MagicMock(return_value=trained_encoder_model)
            mock_load_from_checkpoint = MagicMock(return_value=loaded_target_model)
            mp.setattr(EncoderStage, "from_template", mock_from_template)
            mp.setattr(EncoderStage, "load_from_checkpoint", mock_load_from_checkpoint)
            mp.setattr(service, "_fit", MagicMock(return_value=trainer))
            mp.setattr(
                service, "_save_stage_checkpoint", MagicMock(return_value=ckpt_path)
            )
            mp.setattr(
                "cryocast.model_service.torch.load",
                lambda *_a, **_k: {"state_dict": "era5_state"},
            )
            result = service.train_stage_encoders(
                train_cfg=DictConfig({"foo": "bar"}), checkpoint_dir=tmp_path
            )

        assert result == [trained_encoder_model, loaded_target_model]
        mock_from_template.assert_called_once_with(
            channel_names=["t2m"],
            data_space_in=encoder_era5.data_space_in,
            dataset="era5",
            decoder=service.config_["model"]["decoder"],
            encoder=service.config_["model"]["encoders"]["era5"],
            template=service.model_,
        )
        trained_encoder_model.load_state_dict.assert_called_once_with("era5_state")

        assert mock_load_from_checkpoint.call_args.args == (checkpoint_path,)
        load_kwargs = mock_load_from_checkpoint.call_args.kwargs
        assert load_kwargs["map_location"] == "cpu"
        assert load_kwargs["metrics"] == [MAE_METRIC_CFG]
        assert load_kwargs["weights_only"] is False
        assert load_kwargs["latitudes_fn"]() == service.data_module_.latitudes
        assert load_kwargs["longitudes_fn"]() == service.data_module_.longitudes

    def test_train_stage_finetune_loads_pretrained_weights_and_fits(self) -> None:
        service = ModelService.__new__(ModelService)
        encoder = MagicMock()
        encoder.name = "era5"
        service.model_ = MagicMock(spec=EncodeProcessDecode)
        service.model_.encoders = [encoder]
        service.model_.processor = MagicMock()
        service.model_.decoder = MagicMock()
        service.model_.target_encoder = MagicMock()

        pretrained_encoder = MagicMock()
        pretrained_encoder.name = "era5"
        pretrained_encoder.state_dict.return_value = "encoder_state"
        processor_model = MagicMock()
        processor_model.encoders = [pretrained_encoder]
        processor_model.processor.state_dict.return_value = "processor_state"
        processor_model.decoder.state_dict.return_value = "decoder_state"
        processor_model.target_encoder.state_dict.return_value = "target_encoder_state"

        trainer = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mock_fit = MagicMock(return_value=trainer)
            mock_save = MagicMock()
            mp.setattr(service, "_fit", mock_fit)
            mp.setattr(service, "_save_stage_checkpoint", mock_save)
            result = service.train_stage_finetune(
                train_cfg=DictConfig({"lr": 1}),
                processor_model=cast("ProcessorStage", processor_model),
            )

        encoder.load_state_dict.assert_called_once_with("encoder_state")
        service.model_.processor.load_state_dict.assert_called_once_with(
            "processor_state"
        )
        service.model_.decoder.load_state_dict.assert_called_once_with("decoder_state")
        service.model_.target_encoder.load_state_dict.assert_called_once_with(
            "target_encoder_state"
        )
        mock_fit.assert_called_once_with(
            config=DictConfig({"lr": 1}), job_stage="finetune"
        )
        mock_save.assert_called_once_with(trainer, "finetune")
        assert result is trainer

    def test_train_stage_finetune_transfers_target_encoder_state(
        self, tmp_path: Path
    ) -> None:
        """Copy parameters and buffers without swapping in the frozen pretrained module."""
        service, pretrained = _build_pretrained_processor(tmp_path)
        model = cast("EncodeProcessDecode", service.model)
        target_encoder = model.target_encoder
        buffer_names = dict(pretrained.target_encoder.named_buffers())
        assert any(name.endswith("running_mean") for name in buffer_names)
        assert any(name.endswith("running_var") for name in buffer_names)
        assert any(name.endswith("num_batches_tracked") for name in buffer_names)
        requires_grad = [p.requires_grad for p in target_encoder.parameters()]
        training_mode = target_encoder.training
        trainer = MagicMock(spec=Trainer)

        def check_handover(*, config: DictConfig, job_stage: str) -> Trainer:
            del config, job_stage
            for actual, expected in zip(
                (*model.encoders, model.processor, model.decoder, target_encoder),
                (
                    *pretrained.encoders,
                    pretrained.processor,
                    pretrained.decoder,
                    pretrained.target_encoder,
                ),
                strict=True,
            ):
                torch.testing.assert_close(
                    actual.state_dict(), expected.state_dict(), rtol=0, atol=0
                )
            assert model.target_encoder is target_encoder
            assert [p.requires_grad for p in target_encoder.parameters()] == (
                requires_grad
            )
            assert target_encoder.training == training_mode
            for actual_tensor, expected_tensor in zip(
                (*target_encoder.parameters(), *target_encoder.buffers()),
                (
                    *pretrained.target_encoder.parameters(),
                    *pretrained.target_encoder.buffers(),
                ),
                strict=True,
            ):
                assert actual_tensor.data_ptr() != expected_tensor.data_ptr()
            return trainer

        with pytest.MonkeyPatch.context() as mp:
            mock_fit = MagicMock(side_effect=check_handover)
            mock_save = MagicMock()
            mp.setattr(service, "_fit", mock_fit)
            mp.setattr(service, "_save_stage_checkpoint", mock_save)
            result = service.train_stage_finetune(
                train_cfg=DictConfig({}), processor_model=pretrained
            )

        mock_fit.assert_called_once_with(config=DictConfig({}), job_stage="finetune")
        mock_save.assert_called_once_with(trainer, "finetune")
        assert result is trainer

    @pytest.mark.parametrize("seed", [0, 99, 123], ids=lambda s: f"seed-{s}")
    def test_train_stage_finetune_preserves_latent_loss(
        self, tmp_path: Path, seed: int
    ) -> None:
        """The same diffusion randomness must give the same loss across the handover."""
        service, pretrained = _build_pretrained_processor(tmp_path)
        model = cast("EncodeProcessDecode", service.model)
        model.eval()
        pretrained.eval()
        batch = {
            "sic": torch.rand(2, 2, 1, 16, 16),
            "target": torch.rand(2, 2, 1, 16, 16),
        }

        def loss(module: EncodeProcessDecode) -> torch.Tensor:
            with (
                pytest.MonkeyPatch.context() as mp,
                torch.random.fork_rng(devices=[]),
                torch.no_grad(),
            ):
                mp.setattr(module, "log", MagicMock())
                torch.manual_seed(seed)
                return module.training_step(batch, 0).loss

        expected = loss(pretrained)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "_fit", MagicMock())
            mp.setattr(service, "_save_stage_checkpoint", MagicMock())
            service.train_stage_finetune(
                train_cfg=DictConfig({}), processor_model=pretrained
            )

        assert torch.isfinite(expected)
        torch.testing.assert_close(loss(model), expected, rtol=0, atol=0)

    def test_train_stage_processor_trains_new_processor(self, tmp_path: Path) -> None:
        service = ModelService.__new__(ModelService)
        service.config_ = DictConfig({"model": {"processor": {"foo": "bar"}}})
        service.data_module_ = MagicMock()
        service.data_module_.mask_directory = tmp_path
        decoder_model = MagicMock()
        target_encoder = MagicMock()
        processor_model = MagicMock()
        processor_model.processor.data_space.chw = (4, 8, 8)
        trainer = MagicMock()
        ckpt_path = tmp_path / "processor.ckpt"

        with pytest.MonkeyPatch.context() as mp:
            mock_from_template = MagicMock(return_value=processor_model)
            mp.setattr(ProcessorStage, "from_template", mock_from_template)
            mp.setattr(service, "_fit", MagicMock(return_value=trainer))
            mp.setattr(
                service, "_save_stage_checkpoint", MagicMock(return_value=ckpt_path)
            )
            mp.setattr(
                "cryocast.model_service.torch.load",
                lambda *_a, **_k: {"state_dict": "processor_state"},
            )
            result = service.train_stage_processor(
                decoder_model, target_encoder, train_cfg=DictConfig({"lr": 1})
            )

        mock_from_template.assert_called_once_with(
            processor=service.config_["model"]["processor"],
            decoder_model=decoder_model,
            target_encoder=target_encoder,
            mask_dir=str(tmp_path),
        )
        processor_model.load_state_dict.assert_called_once_with("processor_state")
        assert result is processor_model

    def test_train_stage_processor_loads_existing_checkpoint(
        self, tmp_path: Path
    ) -> None:
        service = ModelService.__new__(ModelService)
        service.config_ = DictConfig(
            {
                "model": {"processor": {"foo": "bar"}},
                "reporting": {"metrics": [MAE_METRIC_CFG]},
            }
        )
        service.data_module_ = MagicMock()
        service.data_module_.mask_directory = tmp_path
        decoder_model = MagicMock()
        target_encoder = MagicMock()
        checkpoint_path = tmp_path / "processor.epoch=1-step=5.ckpt"
        checkpoint_path.write_text("checkpoint")
        loaded_processor = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mock_load = MagicMock(return_value=loaded_processor)
            mp.setattr(ProcessorStage, "load_from_checkpoint", mock_load)
            result = service.train_stage_processor(
                decoder_model,
                target_encoder,
                train_cfg=DictConfig({}),
                checkpoint_dir=tmp_path,
            )

        mock_load.assert_called_once_with(
            checkpoint_path,
            map_location="cpu",
            metrics=[MAE_METRIC_CFG],
            weights_only=False,
            processor=service.config_["model"]["processor"],
            decoder_model=decoder_model,
            target_encoder=target_encoder,
            mask_dir=str(tmp_path),
        )
        assert result is loaded_processor
