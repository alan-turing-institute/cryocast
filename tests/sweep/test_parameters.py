import pytest

from icenet_mp.sweep.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    build_parameter,
)


class TestSanitisedName:
    """Tests for Parameter.sanitised_name."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("model.decoder.channels", "decoder.channels"),
            ("model.encoders.sic.channels", "encoders.sic.channels"),
            ("model.processor.depth", "processor.depth"),
            ("train.optimizer.lr", "optimizer.lr"),
            ("train.scheduler.T_max", "scheduler.T_max"),
            ("train.optimizer", "optimizer"),
            ("loss.delta", "loss.delta"),
            ("window.n_forecast_steps", "window.n_forecast_steps"),
        ],
        ids=[
            "model-decoder",
            "model-encoders",
            "model-processor",
            "train-optimizer-nested",
            "train-scheduler",
            "train-optimizer-leaf",
            "loss-unprefixed",
            "window-unprefixed",
        ],
    )
    def test_strips_known_prefixes(self, name: str, expected: str) -> None:
        assert CategoricalParameter(name, [1]).sanitised_name == expected

    def test_does_not_match_a_longer_name_sharing_the_same_prefix_text(self) -> None:
        """A name sharing a prefix's characters is not the same as sharing its prefix.

        'model.decoders_extra...' must not be mistaken for the 'model.decoder' prefix
        just because it starts with the same characters.
        """
        name = "model.decoders_extra.channels"
        assert CategoricalParameter(name, [1]).sanitised_name == name


class TestIntParameter:
    """Tests for IntParameter."""

    def test_accepts_log_with_default_step(self) -> None:
        parameter = IntParameter("x", 1, 10, log=True)
        assert parameter.step == 1

    def test_accepts_a_non_default_step_without_log(self) -> None:
        parameter = IntParameter("x", 1, 10, step=2)
        assert parameter.step == 2

    def test_rejects_log_with_a_non_default_step(self) -> None:
        with pytest.raises(ValueError, match=r"step.*must be 1"):
            IntParameter("x", 1, 10, log=True, step=2)

    def test_from_spec_rejects_log_with_a_non_default_step(self) -> None:
        spec = {"low": 1, "high": 10, "log": True, "step": 2}
        with pytest.raises(ValueError, match=r"step.*must be 1"):
            IntParameter.from_spec("x", spec)

    def test_sweep_cfg_uses_int_uniform_by_default(self) -> None:
        cfg = IntParameter("x", 1, 10).sweep_cfg()
        assert cfg["x"] == {"distribution": "int_uniform", "min": 1, "max": 10}

    def test_sweep_cfg_quantizes_a_non_default_step(self) -> None:
        cfg = IntParameter("x", 1, 10, step=2).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "q_uniform",
            "min": 1,
            "max": 10,
            "q": 2,
        }

    def test_sweep_cfg_sets_q_to_one_when_log(self) -> None:
        cfg = IntParameter("x", 1, 10, log=True).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "q_log_uniform_values",
            "min": 1,
            "max": 10,
            "q": 1,
        }

    def test_probe_value_returns_low(self) -> None:
        assert IntParameter("x", 1, 10).probe_value() == 1


class TestFloatParameter:
    """Tests for FloatParameter."""

    def test_accepts_log_without_a_step(self) -> None:
        parameter = FloatParameter("x", 0.1, 10.0, log=True)
        assert parameter.step is None

    def test_accepts_a_step_without_log(self) -> None:
        parameter = FloatParameter("x", 0.1, 10.0, step=0.5)
        assert parameter.step == 0.5

    def test_rejects_log_with_a_step(self) -> None:
        with pytest.raises(ValueError, match=r"step.*not supported"):
            FloatParameter("x", 0.1, 10.0, log=True, step=0.5)

    def test_from_spec_rejects_log_with_a_step(self) -> None:
        spec = {"low": 0.1, "high": 10.0, "log": True, "step": 0.5}
        with pytest.raises(ValueError, match=r"step.*not supported"):
            FloatParameter.from_spec("x", spec)

    def test_sweep_cfg_uses_uniform_by_default(self) -> None:
        cfg = FloatParameter("x", 0.1, 10.0).sweep_cfg()
        assert cfg["x"] == {"distribution": "uniform", "min": 0.1, "max": 10.0}

    def test_sweep_cfg_quantizes_a_step(self) -> None:
        cfg = FloatParameter("x", 0.1, 10.0, step=0.5).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "q_uniform",
            "min": 0.1,
            "max": 10.0,
            "q": 0.5,
        }

    def test_sweep_cfg_stays_unquantized_when_log(self) -> None:
        cfg = FloatParameter("x", 0.1, 10.0, log=True).sweep_cfg()
        assert cfg["x"] == {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10.0,
        }

    def test_probe_value_returns_low(self) -> None:
        assert FloatParameter("x", 0.1, 10.0).probe_value() == 0.1


class TestCategoricalParameter:
    """Tests for CategoricalParameter."""

    def test_probe_value_returns_first_choice(self) -> None:
        assert CategoricalParameter("x", ["b", "a"]).probe_value() == "b"


class TestBuildParameter:
    """Tests for build_parameter."""

    def test_builds_an_int_parameter(self) -> None:
        parameter = build_parameter("x", {"type": "int", "low": 1, "high": 10})
        assert isinstance(parameter, IntParameter)

    def test_builds_a_float_parameter(self) -> None:
        parameter = build_parameter("x", {"type": "float", "low": 0.1, "high": 10.0})
        assert isinstance(parameter, FloatParameter)

    def test_builds_a_categorical_parameter(self) -> None:
        parameter = build_parameter("x", {"type": "categorical", "choices": [1, 2]})
        assert isinstance(parameter, CategoricalParameter)

    def test_raises_for_an_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter type"):
            build_parameter("x", {"type": "not-a-type"})
