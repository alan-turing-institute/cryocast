from pathlib import Path
from typing import Any

import pytest
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from icenet_mp.sweep import OptunaSweep
from icenet_mp.sweep.parameters import CategoricalParameter, FloatParameter

pytestmark = pytest.mark.filterwarnings(
    "ignore:QMCSampler is experimental:optuna.exceptions.ExperimentalWarning"
)

CONFIG: dict[str, Any] = {
    "name": "example",
    "n_trials": 6,
    "sampler": "qmc",
    "seed": 0,
    "parameters": {
        "train.optimizer.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "loss.delta": {"type": "float", "low": 0.1, "high": 2.0},
        "model.name": {"type": "categorical", "choices": ["unet", "cnn_unet_cnn"]},
    },
}


def build_sampler(config: dict[str, Any], study_path: Path) -> OptunaSweep:
    """Build an OptunaSweep at `study_path`.

    The persisted model config has a default for every path in `CONFIG`'s
    `parameters`, mirroring a real composed Hydra config where a sweep only ever
    overrides an already-present leaf.
    """
    sampler = OptunaSweep(config)
    sampler._study_path = study_path
    sampler._study_name = study_path.name
    OmegaConf.save(
        OmegaConf.create(
            {
                "train": {"optimizer": {"lr": 0.001}},
                "loss": {"delta": 1.0},
                "model": {"name": "unet"},
            }
        ),
        study_path / "model_config.yaml",
    )
    return sampler


class TestOptunaSweepInit:
    """Tests for OptunaSweep.__init__."""

    def test_attributes_set_from_config(self) -> None:
        sampler = OptunaSweep(CONFIG)
        assert sampler.name == CONFIG["name"]
        assert sampler.n_trials == CONFIG["n_trials"]
        assert sampler.seed == CONFIG["seed"]
        assert sampler.metric == {"name": "validation_loss.min", "goal": "minimize"}
        assert sampler.parameters == CONFIG["parameters"]
        assert sampler.sampler_cls == CONFIG["sampler"]

    def test_seed_defaults_to_zero(self) -> None:
        config = {k: v for k, v in CONFIG.items() if k != "seed"}
        assert OptunaSweep(config).seed == 0


class TestOptunaSweepFromYaml:
    """Tests for OptunaSweep.from_yaml."""

    def test_round_trips_config(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "example.sweep.yaml"
        yaml_path.write_text(yaml.safe_dump(CONFIG))

        sampler = OptunaSweep.from_yaml(yaml_path)

        assert sampler.name == CONFIG["name"]
        assert sampler.parameters == CONFIG["parameters"]


class TestOptunaSweepFromPath:
    """Tests for OptunaSweep.from_path."""

    def test_sets_study_name_and_path(self, tmp_path: Path) -> None:
        study_path = tmp_path / "my-study"
        study_path.mkdir()
        (study_path / "optuna.yaml").write_text(yaml.safe_dump(CONFIG))

        sampler = OptunaSweep.from_path(study_path)

        assert sampler.study_name == "my-study"
        assert sampler.study_path == study_path
        assert sampler.name == CONFIG["name"]


class TestUnsetProperties:
    """Tests for the ValueError guards on entity/study_name/study_path."""

    @pytest.mark.parametrize(
        ("attr", "message"),
        [
            ("entity", "entity has not been set"),
            ("study_name", "Study name has not been set"),
            ("study_path", "Study path has not been set"),
        ],
    )
    def test_raises_when_unset(self, attr: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            getattr(OptunaSweep(CONFIG), attr)


class TestOptunaSweepStudy:
    """Tests for the OptunaSweep.study property."""

    def test_unknown_sampler_raises(self, tmp_path: Path) -> None:
        sampler = build_sampler({**CONFIG, "sampler": "not-a-sampler"}, tmp_path)
        with pytest.raises(ValueError, match="Unknown sampler"):
            _ = sampler.study

    def test_creates_sqlite_backed_study(self, tmp_path: Path) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        study = sampler.study
        assert study.study_name == tmp_path.name
        assert (tmp_path / "optuna.db").exists()


class TestOptunaSweepAsk:
    """Tests for OptunaSweep.ask."""

    def test_one_override_per_parameter(self, tmp_path: Path) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        _, overrides = sampler.ask()
        assert {parameter.name for parameter, _ in overrides} == set(
            CONFIG["parameters"]
        )

    def test_override_generation_mutates_state(self, tmp_path: Path) -> None:
        """Multiple runs in the same study should use different trials."""
        sampler1 = build_sampler(CONFIG, tmp_path)
        _, overrides1 = sampler1.ask()

        sampler2 = build_sampler(CONFIG, tmp_path)
        _, overrides2 = sampler2.ask()

        assert [value for _, value in overrides1] != [value for _, value in overrides2]

    def test_unknown_parameter_type_raises(self, tmp_path: Path) -> None:
        bad_config = {**CONFIG, "parameters": {"x": {"type": "not-a-type"}}}
        sampler = build_sampler(bad_config, tmp_path)
        with pytest.raises(ValueError, match="Unknown parameter type"):
            sampler.ask()

    def test_raises_when_two_parameters_sanitise_to_the_same_name(
        self, tmp_path: Path
    ) -> None:
        """A collision must raise, not silently drop one parameter from the sweep.

        'train.optimizer.lr' and 'optimizer.lr' would otherwise both collapse to the
        W&B config key 'optimizer.lr'.
        """
        config = {
            **CONFIG,
            "parameters": {
                "train.optimizer.lr": {"type": "float", "low": 1e-5, "high": 1e-2},
                "optimizer.lr": {"type": "float", "low": 0.0, "high": 1.0},
            },
        }
        sampler = build_sampler(config, tmp_path)
        with pytest.raises(ValueError, match="both sanitise"):
            sampler.ask()

    def test_does_not_duplicate_a_concurrent_trials_samples(
        self, tmp_path: Path
    ) -> None:
        """Concurrent trials must not clobber each other's sampler state.

        A slow process must re-read `sampler.pkl` before its next ask(), not
        overwrite a faster concurrent process's already-persisted RNG advance with
        its own stale in-memory copy.
        """
        config = {**CONFIG, "sampler": "random"}
        slow = build_sampler(config, tmp_path)
        slow.ask()

        concurrent = build_sampler(config, tmp_path)
        _, concurrent_overrides = concurrent.ask()

        _, slow_overrides = slow.ask()

        assert [v for _, v in slow_overrides] != [v for _, v in concurrent_overrides]


class TestOptunaSweepGenerateTrialConfig:
    """Tests for OptunaSweep.generate_trial_config."""

    def test_parameter_values_match_search_space(self, tmp_path: Path) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        _, overrides = sampler.ask()
        trial_cfg = sampler.generate_trial_config(overrides)
        assert 1e-5 <= OmegaConf.select(trial_cfg, "train.optimizer.lr") <= 1e-2
        assert 0.1 <= OmegaConf.select(trial_cfg, "loss.delta") <= 2.0
        assert OmegaConf.select(trial_cfg, "model.name") in {"unet", "cnn_unet_cnn"}

    def test_wandb_config_mirrors_overrides_by_sanitised_name(
        self, tmp_path: Path
    ) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        _, overrides = sampler.ask()
        trial_cfg = sampler.generate_trial_config(overrides)
        wandb_config = OmegaConf.select(trial_cfg, "reporting.loggers.wandb.config")
        assert set(wandb_config) == {"optimizer.lr", "loss.delta", "model.name"}
        assert wandb_config["optimizer.lr"] == OmegaConf.select(
            trial_cfg, "train.optimizer.lr"
        )
        assert wandb_config["loss.delta"] == OmegaConf.select(trial_cfg, "loss.delta")
        assert wandb_config["model.name"] == OmegaConf.select(trial_cfg, "model.name")

    def test_categorical_value_resembling_a_yaml_keyword_is_kept_as_a_string(
        self, tmp_path: Path
    ) -> None:
        """A choice like 'off' or 'null' must not be YAML-coerced to bool/None."""
        sampler = build_sampler(CONFIG, tmp_path)
        parameter = CategoricalParameter("model.name", ["off", "null", "unet"])
        trial_cfg = sampler.generate_trial_config([(parameter, "off")])
        assert OmegaConf.select(trial_cfg, "model.name") == "off"
        wandb_config = OmegaConf.select(trial_cfg, "reporting.loggers.wandb.config")
        assert wandb_config["model.name"] == "off"

    def test_unknown_parameter_path_raises(self, tmp_path: Path) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        parameter = FloatParameter("train.optimzer.lr", 1e-5, 1e-2)
        with pytest.raises(OmegaConfBaseException, match="optimzer"):
            sampler.generate_trial_config([(parameter, 0.001)])


class TestOptunaSweepValidateParameters:
    """Tests for OptunaSweep.validate_parameters."""

    def test_accepts_paths_present_in_the_base_config(self) -> None:
        sampler = OptunaSweep(CONFIG)
        model_cfg = OmegaConf.create(
            {
                "train": {"optimizer": {"lr": 0.001}},
                "loss": {"delta": 1.0},
                "model": {"name": "unet"},
            }
        )
        sampler.validate_parameters(model_cfg)  # must not raise

    def test_raises_for_an_unknown_parameter_path(self) -> None:
        config = {
            **CONFIG,
            "parameters": {
                "train.optimzer.lr": {"type": "float", "low": 1e-5, "high": 1e-2}
            },
        }
        sampler = OptunaSweep(config)
        model_cfg = OmegaConf.create({"train": {"optimizer": {"lr": 0.001}}})
        with pytest.raises(ValueError, match=r"train\.optimzer\.lr"):
            sampler.validate_parameters(model_cfg)

    def test_reports_every_invalid_parameter_not_just_the_first(self) -> None:
        config = {
            **CONFIG,
            "parameters": {
                "train.optimzer.lr": {"type": "float", "low": 1e-5, "high": 1e-2},
                "model.decoder.made_up": {"type": "categorical", "choices": ["a"]},
            },
        }
        sampler = OptunaSweep(config)
        model_cfg = OmegaConf.create(
            {"train": {"optimizer": {"lr": 0.001}}, "model": {"decoder": {}}}
        )
        with pytest.raises(ValueError, match="Invalid sweep parameter") as exc_info:
            sampler.validate_parameters(model_cfg)
        assert "train.optimzer.lr" in str(exc_info.value)
        assert "model.decoder.made_up" in str(exc_info.value)

    def test_does_not_mutate_the_passed_in_config(self) -> None:
        sampler = OptunaSweep(CONFIG)
        model_cfg = OmegaConf.create(
            {
                "train": {"optimizer": {"lr": 0.001}},
                "loss": {"delta": 1.0},
                "model": {"name": "unet"},
            }
        )
        sampler.validate_parameters(model_cfg)
        assert OmegaConf.select(model_cfg, "train.optimizer.lr") == 0.001


class TestOptunaSweepAskAndTell:
    """Tests for OptunaSweep.ask and OptunaSweep.tell."""

    def test_ask_persists_sampler_state(self, tmp_path: Path) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        sampler.ask()
        assert (tmp_path / "sampler.pkl").exists()

    def test_successive_asks_return_different_trials(self, tmp_path: Path) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        first, _ = sampler.ask()
        second, _ = sampler.ask()
        assert first.number != second.number

    def test_tell_records_value_on_best_trial(self, tmp_path: Path) -> None:
        sampler = build_sampler(CONFIG, tmp_path)
        trial, _ = sampler.ask()
        result = sampler.tell(trial, 0.5)
        assert result.number == trial.number
        assert result.value == 0.5
        assert sampler.study.best_trial.number == trial.number


class TestOptunaSweepInitialiseStudy:
    """Tests for OptunaSweep.initialise_study."""

    def test_writes_config_files(self, tmp_path: Path) -> None:
        sampler = OptunaSweep({**CONFIG, "entity": "test-entity"})
        model_cfg = OmegaConf.create({"base_path": str(tmp_path)})

        sampler.initialise_study(model_cfg, "sweep123")

        study_path = tmp_path / "sweeps" / "sweep123"
        assert sampler.study_path == study_path
        assert sampler.study_name == "sweep123"

        optuna_cfg = yaml.safe_load((study_path / "optuna.yaml").read_text())
        assert optuna_cfg["name"] == CONFIG["name"]
        assert optuna_cfg["entity"] == "test-entity"
        assert optuna_cfg["parameters"] == CONFIG["parameters"]

        saved_model_cfg = OmegaConf.load(study_path / "model_config.yaml")
        assert saved_model_cfg.base_path == str(tmp_path)


class TestOptunaSweepInitialiseSweep:
    """Tests for OptunaSweep.initialise_sweep."""

    def test_builds_sweep_config_and_calls_wandb_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_sweep(
            sweep_config: dict[str, Any],
            entity: str | None = None,
            project: str | None = None,
        ) -> str:
            captured["sweep_config"] = sweep_config
            captured["entity"] = entity
            captured["project"] = project
            return "fake-sweep-id"

        monkeypatch.setattr(wandb, "sweep", fake_sweep)
        sampler = OptunaSweep(CONFIG)
        model_cfg = DictConfig(
            {"reporting": {"loggers": {"wandb": {"entity": "my-entity"}}}}
        )

        sweep_id = sampler.initialise_sweep(model_cfg)

        assert sweep_id == "fake-sweep-id"
        assert sampler.entity == "my-entity"
        assert captured["entity"] == "my-entity"
        assert captured["project"] == "train"
        assert captured["sweep_config"]["method"] == "random"
        assert captured["sweep_config"]["metric"] == {
            "name": "validation_loss.min",
            "goal": "minimize",
        }
        assert captured["sweep_config"]["parameters"]["model.name"] == {
            "values": ["unet", "cnn_unet_cnn"]
        }

    def test_falls_back_to_wandb_entity_env_var_when_entity_is_explicitly_null(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            wandb, "sweep", lambda *_, **kwargs: captured.update(kwargs) or "sweep-id"
        )
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        sampler = OptunaSweep(CONFIG)
        model_cfg = DictConfig({"reporting": {"loggers": {"wandb": {"entity": None}}}})

        sampler.initialise_sweep(model_cfg)

        assert sampler.entity == "env-entity"
        assert captured["entity"] == "env-entity"
