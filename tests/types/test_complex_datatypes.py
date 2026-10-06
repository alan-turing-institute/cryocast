from datetime import datetime, timedelta

import pytest
import torch
from omegaconf import DictConfig

from cryocast.types import (
    ColourScale,
    DataSpace,
    Hemisphere,
    ModelStepOutput,
    PlotSpec,
    Timespan,
)


class TestDataSpace:
    """Tests for DataSpace."""

    def test_properties(self) -> None:
        """Expose the expected DataSpace helper properties."""
        space = DataSpace(channels=3, name="sic", shape=(16, 24))

        assert space.channels == 3
        assert space.name == "sic"
        assert space.shape == (16, 24)
        assert space.area == 384
        assert space.chw == (3, 16, 24)

    def test_round_trip_from_dictconfig(self) -> None:
        """Round-trip DataSpace values through DictConfig."""
        config = DictConfig({"channels": 4, "name": "weather", "shape": [32, 48]})

        space = DataSpace.from_dict(config)
        result = space.to_dict()

        assert space == DataSpace(channels=4, name="weather", shape=(32, 48))
        assert isinstance(result, DictConfig)
        assert DataSpace.from_dict(result) == space

    def test_equal_by_value(self) -> None:
        """Treat distinct DataSpaces with the same values as equal, with equal hashes."""
        space = DataSpace(channels=3, name="sic", shape=(16, 24))
        other = DataSpace(channels="3", name="sic", shape=("16", "24"))  # type: ignore[arg-type]

        assert space is not other
        assert space == other
        assert hash(space) == hash(other)
        assert len({space, other}) == 1

    @pytest.mark.parametrize("field", ["channels", "name", "shape"])
    def test_immutable(self, field: str) -> None:
        """Reject assignment to any field, so that the hash cannot change."""
        space = DataSpace(channels=3, name="sic", shape=(16, 24))

        with pytest.raises(AttributeError):
            setattr(space, field, getattr(space, field))

    @pytest.mark.parametrize(
        ("channels", "name", "shape"),
        [(4, "sic", (16, 24)), (3, "era5", (16, 24)), (3, "sic", (24, 16))],
        ids=["channels", "name", "shape"],
    )
    def test_not_equal_when_any_field_differs(
        self, channels: int, name: str, shape: tuple[int, int]
    ) -> None:
        """Treat DataSpaces that differ in any one field as unequal."""
        space = DataSpace(channels=3, name="sic", shape=(16, 24))

        assert space != DataSpace(channels=channels, name=name, shape=shape)

    def test_not_equal_to_other_types(self) -> None:
        """Do not treat a DataSpace as equal to an object of another type."""
        space = DataSpace(channels=3, name="sic", shape=(16, 24))

        assert space != (3, "sic", (16, 24))
        assert space != space.to_dict()


class TestColourScale:
    """Tests for ColourScale."""

    def test_preserves_bounds_and_colourmap(self) -> None:
        """Preserve bounds and colourmap configuration."""
        spec = ColourScale(vmin=-1.0, vmax=1.0, cmap="coolwarm")

        assert spec.vmin == pytest.approx(-1.0)
        assert spec.vmax == pytest.approx(1.0)
        assert spec.cmap == "coolwarm"

    def test_defaults_to_no_units(self) -> None:
        """A ColourScale built without units (e.g. for a difference panel) defaults to None."""
        spec = ColourScale(cmap="viridis")

        assert spec.units is None


class TestPlotSpec:
    """Tests for PlotSpec."""

    def test_accepts_dictconfig_override(self) -> None:
        """Apply PlotSpec overrides supplied as DictConfig."""
        spec = PlotSpec(hemisphere=Hemisphere.NORTH)
        override = DictConfig(
            {
                "include_difference": False,
                "dpi": 200,
            }
        )

        result = spec + override

        assert result.hemisphere == "north"
        assert result.include_difference is False
        assert result.dpi == 200

    def test_add_none_returns_same_spec(self) -> None:
        """Return the same PlotSpec when merging with None."""
        spec = PlotSpec(hemisphere=Hemisphere.NORTH)

        assert spec + None is spec

    def test_add_plot_spec_override(self) -> None:
        """Apply overrides supplied as another PlotSpec instance."""
        spec = PlotSpec(hemisphere=Hemisphere.NORTH, colourmap="viridis")
        override = PlotSpec(colourmap="magma", video_fps=5)

        result = spec + override

        assert result.colourmap == "magma"
        assert result.video_fps == 5

    def test_default_styles_are_not_shared(self) -> None:
        """Keep default per-variable style dictionaries independent."""
        first = PlotSpec()
        second = PlotSpec()

        first.per_variable_styles["sic-ssmis:ice_conc"]["cmap"] = "magma"

        assert second.per_variable_styles["sic-ssmis:ice_conc"]["cmap"] == "Blues_r"

    def test_default_uncertainty_variables(self) -> None:
        """Default uncertainty_variables maps ice_conc to its reported uncertainty."""
        spec = PlotSpec()

        assert spec.uncertainty_variables == {"ice_conc": "total_standard_uncertainty"}

    def test_default_uncertainty_variables_are_not_shared(self) -> None:
        """Keep default uncertainty_variables dictionaries independent across instances."""
        first = PlotSpec()
        second = PlotSpec()

        first.uncertainty_variables["ice_conc"] = "other_uncertainty"

        assert second.uncertainty_variables["ice_conc"] == "total_standard_uncertainty"

    def test_uncertainty_variables_is_overridable(self) -> None:
        """Allow callers to configure a different uncertainty-variable mapping."""
        spec = PlotSpec(uncertainty_variables={"sic": "sic_uncertainty"})

        assert spec.uncertainty_variables == {"sic": "sic_uncertainty"}

    def test_dict_override_preserves_other_values(self) -> None:
        """Apply dict overrides without changing unspecified PlotSpec values."""
        spec = PlotSpec(hemisphere=Hemisphere.NORTH, colourmap="viridis", video_fps=2)

        result = spec + {"colourmap": "magma", "video_fps": 5}

        assert result.hemisphere == "north"
        assert result.colourmap == "magma"
        assert result.video_fps == 5
        assert result.include_difference is True


class TestModelStepOutput:
    """Tests for ModelStepOutput."""

    def _make_output(self) -> ModelStepOutput:
        return ModelStepOutput(
            prediction=torch.zeros(1, 1, 1, 4, 4),
            target=torch.ones(1, 1, 1, 4, 4),
            loss=torch.tensor(0.5),
        )

    def test_copy_contains_all_keys(self) -> None:
        """copy() dict has exactly the three expected keys."""
        result = self._make_output().copy()
        assert set(result.keys()) == {"prediction", "target", "loss"}

    def test_copy_returns_plain_dict(self) -> None:
        """copy() returns a plain dict, not a ModelStepOutput or other Mapping."""
        result = self._make_output().copy()
        assert type(result) is dict

    def test_copy_values_are_original_tensors(self) -> None:
        """copy() values are the same tensor objects as the original fields."""
        output = self._make_output()
        result = output.copy()
        assert result["prediction"] is output.prediction
        assert result["target"] is output.target
        assert result["loss"] is output.loss

    def test_getitem_raises_key_error_for_unknown_key(self) -> None:
        """__getitem__ raises KeyError for a key that is not one of the three fields."""
        output = self._make_output()
        with pytest.raises(KeyError, match="unknown"):
            output["unknown"]

    def test_len_returns_three(self) -> None:
        """ModelStepOutput always reports a length of three."""
        assert len(self._make_output()) == 3


class TestTimespan:
    """Tests for Timespan."""

    def test_frequency_daily(self) -> None:
        """Derive a one-day frequency from consecutive daily dates."""
        span = Timespan(
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        )

        assert span.frequency == timedelta(days=1)

    def test_frequency_sub_daily(self) -> None:
        """Derive an hourly frequency from consecutive hourly dates."""
        span = Timespan(
            [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 1, 18),
            ]
        )

        assert span.frequency == timedelta(hours=6)

    def test_frequency_none_with_single_date(self) -> None:
        """Return None when there aren't enough dates to derive a spacing."""
        span = Timespan([datetime(2020, 1, 1)])

        assert span.frequency is None
