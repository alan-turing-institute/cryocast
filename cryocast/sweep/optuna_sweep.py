import os
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from optuna import Study, create_study
from optuna.importance import get_param_importances
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, Trial, TrialState

from .parameters import Parameter, build_parameter
from .sampler_store import SamplerStore


class OptunaSweep:
    # The goal is always to minimise the validation loss
    metric: ClassVar[dict[str, str]] = {
        "name": "validation_loss.min",
        "goal": "minimize",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize an OptunaSweep from a parsed YAML dict."""
        self.name: str = config["name"]
        self.n_trials: int = config["n_trials"]
        self.seed: int = config.get("seed", 0)
        self.parameters: dict[str, Any] = config["parameters"]
        self.sampler_cls = config["sampler"]
        self._entity: str | None = config.get("entity")
        self._study: Study | None = None
        self._study_name: str | None = None
        self._study_path: Path | None = None

    @classmethod
    def from_path(cls, study_path: Path) -> "OptunaSweep":
        """Load a sweep config from a study path."""
        yaml_path = study_path / "optuna.yaml"
        instance = cls.from_yaml(yaml_path)
        instance._study_path = study_path
        instance._study_name = study_path.name
        return instance

    @classmethod
    def from_yaml(cls, path: Path) -> "OptunaSweep":
        """Load a sweep config from a YAML file."""
        return cls(yaml.safe_load(path.read_text()))

    @property
    def entity(self) -> str:
        if self._entity is None:
            msg = "W&B entity has not been set."
            raise ValueError(msg)
        return self._entity

    @property
    def study_name(self) -> str:
        if self._study_name is None:
            msg = "Study name has not been set."
            raise ValueError(msg)
        return self._study_name

    @property
    def study_path(self) -> Path:
        if self._study_path is None:
            msg = "Study path has not been set."
            raise ValueError(msg)
        return self._study_path

    @property
    def study(self) -> Study:
        if self._study is None:
            study_direction = {
                "maximize": StudyDirection.MAXIMIZE,
                "minimize": StudyDirection.MINIMIZE,
            }.get(self.metric["goal"], StudyDirection.NOT_SET)
            # Use an in-memory placeholder sampler to avoid unnecessary disk I/O
            self._study = create_study(
                direction=study_direction,
                load_if_exists=True,
                sampler=self.sampler.temporary(),
                storage=f"sqlite:///{self.study_path / 'optuna.db'}",
                study_name=self.study_name,
            )
        return self._study

    @cached_property
    def sampler(self) -> SamplerStore:
        return SamplerStore(self.study_path, self.sampler_cls, self.seed)

    @cached_property
    def _built_parameters(self) -> list[Parameter]:
        """Build and validate every configured Parameter.

        Raises if two parameters sanitise to the same W&B config key, since the sweep
        or trial config would otherwise silently drop one of them.
        """
        parameters = [
            build_parameter(name, param_spec)
            for name, param_spec in self.parameters.items()
        ]
        seen: dict[str, str] = {}
        for parameter in parameters:
            collision = seen.get(parameter.sanitised_name)
            if collision is not None:
                msg = (
                    f"Parameters '{collision}' and '{parameter.name}' both sanitise "
                    f"to the W&B config key '{parameter.sanitised_name}'. If no "
                    "changes are made, one will be silently dropped from the sweep."
                )
                raise ValueError(msg)
            seen[parameter.sanitised_name] = parameter.name
        return parameters

    def ask(self) -> tuple[Trial, list[tuple[Parameter, int | float | str]]]:
        """Ask the study for a new trial and suggest a value for each parameter.

        ask() and suggest() calls both mutate the sampler, so we use a single lock
        acquisition to keep the pair synchronised.
        """
        with self.sampler.lock(self.study):
            trial = self.study.ask()
            overrides = [
                (parameter, parameter.suggest(trial))
                for parameter in self._built_parameters
            ]
            return trial, overrides

    def generate_trial_config(
        self, overrides: list[tuple[Parameter, int | float | str]]
    ) -> DictConfig:
        """Generate a trial config from a list of parameter overrides."""
        base_config = DictConfig(OmegaConf.load(self.study_path / "model_config.yaml"))
        # Set each override directly, so that categorical values cannot be interpreted
        # as YAML keywords (e.g. "off", "null") and re-parsed into a bool/None. We set
        # struct-mode so that typos will raise instead of silently creating new keys.
        OmegaConf.set_struct(base_config, value=True)
        for parameter, value in overrides:
            OmegaConf.update(base_config, parameter.name, value, merge=True)
        # Disable struct mode before merging in wandb config, which is a new key
        OmegaConf.set_struct(base_config, value=False)
        wandb_overrides = OmegaConf.create(
            {
                "reporting": {
                    "loggers": {
                        "wandb": {
                            "config": {
                                parameter.sanitised_name: value
                                for parameter, value in overrides
                            }
                        }
                    }
                }
            }
        )
        return DictConfig(OmegaConf.merge(base_config, wandb_overrides))

    def validate_parameters(self, model_cfg: DictConfig) -> None:
        """Check that every configured parameter resolves against `model_cfg`."""
        probe_config = model_cfg.copy()
        OmegaConf.set_struct(probe_config, value=True)
        errors: list[str] = []
        for parameter in self._built_parameters:
            try:
                OmegaConf.update(
                    probe_config,
                    parameter.name,
                    parameter.probe_value(),
                    merge=True,
                )
            except OmegaConfBaseException as exc:
                errors.append(f"'{parameter.name}': {exc}")
        if errors:
            msg = "Invalid sweep parameter(s):\n" + "\n".join(errors)
            raise ValueError(msg)

    def initialise_study(self, model_cfg: DictConfig, sweep_id: str) -> None:
        """Create the Optuna study directory and save the model and sweep configs."""
        # Generate study and storage paths
        sweep_base = (Path(model_cfg.get("base_path"))).resolve() / "sweeps"
        self._study_name = sweep_id
        self._study_path = (sweep_base / self.study_name).resolve()
        self.study_path.mkdir(parents=True, exist_ok=True)

        # Write the Optuna config to disk
        optuna_cfg = {
            "entity": self.entity,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "name": self.name,
            "parameters": self.parameters,
            "sampler": self.sampler_cls,
            "seed": self.seed,
        }
        with (self.study_path / "optuna.yaml").open("w") as f_yaml:
            yaml.safe_dump(optuna_cfg, f_yaml, default_flow_style=False)

        # Save the model config to the study path
        OmegaConf.save(model_cfg, self.study_path / "model_config.yaml")

    def initialise_sweep(self, model_cfg: DictConfig) -> str:
        """Generate a new W&B sweep."""
        # Generate the W&B sweep config
        sweep_config = {
            "program": "imp",
            "method": "random",
            "metric": self.metric,
            "parameters": {
                k: v
                for parameter in self._built_parameters
                for k, v in parameter.sweep_cfg().items()
            },
        }
        wandb_cfg = model_cfg.get("reporting", {}).get("loggers", {}).get("wandb", {})
        self._entity = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
        # Start the sweep
        return wandb.sweep(sweep_config, entity=self.entity, project="train")

    def parameter_importances(self) -> dict[str, float]:
        """Use Optuna to estimate parameter importance.

        Requires at least two completed trials with varying parameters.

        Returns:
            A dict mapping parameter names to their importance, from most to least.

        """
        trial_values = {
            t.value for t in self.study.get_trials() if t.state == TrialState.COMPLETE
        }
        if len(trial_values) < 2:  # noqa: PLR2004
            return {}
        try:
            return get_param_importances(self.study)
        except (ValueError, RuntimeError):
            return {}

    def tell(
        self,
        trial: Trial | int,
        values: float | Sequence[float] | None = None,
        *,
        state: TrialState | None = None,
        skip_if_finished: bool = False,
    ) -> FrozenTrial:
        """Tell the study the result of a completed trial.

        Since tell() calls mutate the sampler, we need to perform this under a lock.
        """
        with self.sampler.lock(self.study):
            return self.study.tell(trial, values, state, skip_if_finished)
