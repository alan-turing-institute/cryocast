from abc import ABC, abstractmethod
from typing import Any, cast

from optuna import Trial


class Parameter(ABC):
    """A single hyperparameter in a sweep search space."""

    def __init__(self, name: str) -> None:
        """Initialise a Parameter with a name."""
        self.name = name

    @property
    def sanitised_name(self) -> str:
        """Return a sanitised version of the parameter name for use in W&B sweeps."""
        prefixes = {
            "model.decoder": "decoder",
            "model.encoders": "encoders",
            "model.processor": "processor",
            "train.optimizer": "optimizer",
            "train.scheduler": "scheduler",
        }
        for old, new in prefixes.items():
            if self.name == old:
                return new
            if self.name.startswith(f"{old}."):
                return new + self.name[len(old) :]
        return self.name

    @abstractmethod
    def suggest(self, trial: Trial) -> int | float | str:
        """Sample a value for this parameter from an Optuna trial."""

    @abstractmethod
    def probe_value(self) -> int | float | str:
        """Return a representative value, without needing an Optuna trial."""

    @classmethod
    @abstractmethod
    def from_spec(cls, name: str, spec: dict[str, Any]) -> "Parameter":
        """Build this parameter from a search-space spec dict."""

    @abstractmethod
    def sweep_cfg(self) -> dict[str, dict[str, Any]]:
        """Build a W&B sweep config for this parameter."""


class IntParameter(Parameter):
    """An integer-valued parameter, sampled uniformly or log-uniformly over a range."""

    def __init__(
        self, name: str, low: int, high: int, *, log: bool = False, step: int = 1
    ) -> None:
        """Initialise an IntParameter with a name, range, and sampling options."""
        if log and step != 1:
            msg = f"Parameter '{name}': `step` must be 1 when `log` is true."
            raise ValueError(msg)
        super().__init__(name)
        self.low = low
        self.high = high
        self.log = log
        self.step = step

    @classmethod
    def from_spec(cls, name: str, spec: dict[str, Any]) -> "IntParameter":
        return cls(
            name,
            spec["low"],
            spec["high"],
            log=spec.get("log", False),
            step=spec.get("step", 1),
        )

    def suggest(self, trial: Trial) -> int:
        return trial.suggest_int(
            self.name, self.low, self.high, log=self.log, step=self.step
        )

    def probe_value(self) -> int:
        return self.low

    def sweep_cfg(self) -> dict[str, dict[str, Any]]:
        cfg: dict[str, Any] = {"min": self.low, "max": self.high}
        if self.log:
            cfg["distribution"] = "q_log_uniform_values"
            cfg["q"] = self.step
        elif self.step != 1:
            cfg["distribution"] = "q_uniform"
            cfg["q"] = self.step
        else:
            cfg["distribution"] = "int_uniform"
        return {self.sanitised_name: cfg}


class FloatParameter(Parameter):
    """A float-valued parameter, sampled uniformly or log-uniformly over a range."""

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> None:
        """Initialise a FloatParameter with a name, range, and sampling options."""
        if log and step is not None:
            msg = f"Parameter '{name}': `step` is not supported when `log` is true."
            raise ValueError(msg)
        super().__init__(name)
        self.low = low
        self.high = high
        self.log = log
        self.step = step

    @classmethod
    def from_spec(cls, name: str, spec: dict[str, Any]) -> "FloatParameter":
        return cls(
            name,
            spec["low"],
            spec["high"],
            log=spec.get("log", False),
            step=spec.get("step"),
        )

    def suggest(self, trial: Trial) -> float:
        return trial.suggest_float(
            self.name, self.low, self.high, log=self.log, step=self.step
        )

    def probe_value(self) -> float:
        return self.low

    def sweep_cfg(self) -> dict[str, dict[str, Any]]:
        cfg: dict[str, Any] = {"min": self.low, "max": self.high}
        if self.log:
            cfg["distribution"] = "log_uniform_values"
        elif self.step is not None:
            cfg["distribution"] = "q_uniform"
            cfg["q"] = self.step
        else:
            cfg["distribution"] = "uniform"
        return {self.sanitised_name: cfg}


class CategoricalParameter(Parameter):
    """A parameter sampled from a fixed set of choices."""

    def __init__(self, name: str, choices: list[int | float | str]) -> None:
        """Initialise a CategoricalParameter with a name and a list of choices."""
        super().__init__(name)
        self.choices = choices

    @classmethod
    def from_spec(cls, name: str, spec: dict[str, Any]) -> "CategoricalParameter":
        return cls(name, spec["choices"])

    def suggest(self, trial: Trial) -> int | float | str:
        return cast(
            "int | float | str", trial.suggest_categorical(self.name, self.choices)
        )

    def probe_value(self) -> int | float | str:
        return self.choices[0]

    def sweep_cfg(self) -> dict[str, dict[str, Any]]:
        return {self.sanitised_name: {"values": self.choices}}


PARAMETER_TYPES: dict[str, type[Parameter]] = {
    "categorical": CategoricalParameter,
    "float": FloatParameter,
    "int": IntParameter,
}


def build_parameter(name: str, spec: dict[str, Any]) -> Parameter:
    """Build the `Parameter` subclass matching a search-space spec's `type`."""
    param_type = spec["type"]
    param_cls = PARAMETER_TYPES.get(param_type)
    if param_cls is None:
        msg = f"Unknown parameter type '{param_type}' for parameter '{name}'"
        raise ValueError(msg)
    return param_cls.from_spec(name, spec)
