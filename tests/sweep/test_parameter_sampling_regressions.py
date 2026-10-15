"""Exercise actual Optuna sampling, invalid domains, and W&B preflight checks."""

from pathlib import Path
from typing import Any

import pytest
import wandb
from omegaconf import OmegaConf
from optuna import create_study
from optuna.distributions import FloatDistribution
from optuna.samplers import RandomSampler

from cryocast.sweep import OptunaSweep, optuna_sweep
from cryocast.sweep.parameters import CategoricalParameter, build_parameter


@pytest.mark.parametrize(
    ("spec", "allowed"),
    [
        ({"type": "int", "low": 4, "high": 12, "step": 4}, {4, 8, 12}),
        ({"type": "int", "low": 1, "high": 100, "log": True}, range(1, 101)),
        (
            {"type": "float", "low": 0.25, "high": 1.0, "step": 0.25},
            {0.25, 0.5, 0.75, 1.0},
        ),
        (
            {"type": "categorical", "choices": ["off", "on", "null"]},
            {"off", "on", "null"},
        ),
    ],
    ids=["quantised-int", "log-int", "quantised-float", "string-categorical"],
)
def test_valid_search_space_samples_from_its_real_optuna_distribution(
    spec: dict[str, Any], allowed: object
) -> None:
    """A parameter's declared range must be reflected in trial storage."""
    parameter = build_parameter("model.test_setting", spec)
    study = create_study(sampler=RandomSampler(seed=708))
    trial = study.ask()

    result = parameter.suggest(trial)

    assert result in allowed  # type: ignore[operator]
    assert trial.params == {"model.test_setting": result}
    assert "model.test_setting" in trial.distributions


def test_log_float_sampling_uses_a_positive_real_domain() -> None:
    """Regression for float log-uniform suggestions (not just W&B config)."""
    parameter = build_parameter(
        "train.optimizer.lr",
        {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    )
    study = create_study(sampler=RandomSampler(seed=82))
    trial = study.ask()
    result = parameter.suggest(trial)

    assert isinstance(result, float)
    assert 1e-5 <= result <= 1e-2
    assert trial.params["train.optimizer.lr"] == result
    distribution = trial.distributions["train.optimizer.lr"]
    assert isinstance(distribution, FloatDistribution)
    assert distribution.log


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "int", "low": 3, "high": 1},
        {"type": "int", "low": 0, "high": 100, "log": True},
        {"type": "int", "low": 1, "high": 10, "step": 0},
        {"type": "float", "low": 3.0, "high": 1.0},
        {"type": "float", "low": 0.0, "high": 10.0, "log": True},
        {"type": "float", "low": 0.1, "high": 10.0, "step": 0.0},
        {"type": "categorical", "choices": []},
    ],
    ids=[
        "inverted-int",
        "nonpositive-log-int",
        "zero-int-step",
        "inverted-float",
        "nonpositive-log-float",
        "zero-float-step",
        "empty-choices",
    ],
)
def test_invalid_search_space_fails_at_optuna_suggestion(
    spec: dict[str, Any],
) -> None:
    """Never silently turn invalid sweep domains into successful trials."""
    parameter = build_parameter("model.test_setting", spec)
    trial = create_study(sampler=RandomSampler(seed=45)).ask()
    with pytest.raises(ValueError, match=r"."):
        parameter.suggest(trial)


@pytest.mark.parametrize("value", ["off", "null", "true", "false", "yes", "no"])
def test_yaml_keyword_like_categorical_values_keep_string_identity(
    tmp_path: Path, value: str
) -> None:
    """A categorical override must not be reparsed as YAML bool or null."""
    model_cfg = OmegaConf.create({"model": {"kind": "initial"}})
    sampler = OptunaSweep(
        {
            "name": "keyword-values",
            "sampler": "random",
            "parameters": {},
        }
    )
    # generate_trial_config reads the persisted base config.
    sampler._study_path = tmp_path
    OmegaConf.save(model_cfg, tmp_path / "model_config.yaml")
    param = CategoricalParameter("model.kind", [value])
    config = sampler.generate_trial_config([(param, value)])

    assert OmegaConf.select(config, "model.kind") == value
    assert (
        OmegaConf.select(config, "reporting.loggers.wandb.config")["model.kind"]
        == value
    )
    assert OmegaConf.select(model_cfg, "model.kind") == "initial"


def test_missing_wandb_entity_stops_before_external_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset entity must fail locally, not call W&B with bad parameters."""
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    requests: list[object] = []
    monkeypatch.setattr(
        wandb, "sweep", lambda *args, **kwargs: requests.append((args, kwargs))
    )
    sampler = OptunaSweep(
        {
            "name": "needs-entity",
            "sampler": "random",
            "parameters": {
                "model.kind": {"type": "categorical", "choices": ["x"]},
            },
        }
    )
    config = OmegaConf.create({"reporting": {"loggers": {"wandb": {}}}})

    with pytest.raises(ValueError, match="entity has not been set"):
        sampler.initialise_sweep(config)

    assert requests == []


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
def test_parameter_importance_backend_failure_returns_empty_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    """Numerically degenerate Optuna importance must not crash summaries."""
    sweep = OptunaSweep(
        {
            "name": "importance-recovery",
            "sampler": "random",
            "seed": 7,
            "entity": "offline",
            "parameters": {
                "model.width": {"type": "int", "low": 1, "high": 4},
            },
        }
    )
    sweep.initialise_study(
        OmegaConf.create({"base_path": str(tmp_path), "model": {"width": 2}}),
        "importance-check",
    )
    for score in (0.1, 0.3):
        trial, _ = sweep.ask()
        sweep.tell(trial, score)

    def fail_importances(_study: object) -> None:
        msg = "simulated Optuna importance failure"
        raise exception_type(msg)

    monkeypatch.setattr(optuna_sweep, "get_param_importances", fail_importances)
    assert sweep.parameter_importances() == {}
