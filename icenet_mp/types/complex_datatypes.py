from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Literal, Self, cast

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .annotations import TensorNTCHW
from .constants import SEA_ICE_THRESHOLD
from .enums import DiffMode, Hemisphere
from .protocols import SupportsMetadataFromDataset


class DataSpace:
    """Description of a CHW data space."""

    channels: int
    name: str
    shape: tuple[int, int]

    def __init__(self, channels: int, name: str, shape: Sequence[int]) -> None:
        """Initialise a DataSpace from channels, name and shape."""
        self.channels = int(channels)
        self.name = name
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def area(self) -> int:
        """Return the area of the data space."""
        return self.shape[0] * self.shape[1]

    @property
    def chw(self) -> tuple[int, int, int]:
        """Return a tuple of [channels, height, width]."""
        return (self.channels, *self.shape)

    @classmethod
    def from_dict(cls, config: DictConfig | dict[str, Any]) -> Self:
        return cls(
            channels=config["channels"], name=config["name"], shape=config["shape"]
        )

    def to_dict(self) -> DictConfig:
        """Return the DataSpace as a DictConfig."""
        return DictConfig(
            {"channels": self.channels, "name": self.name, "shape": self.shape}
        )


@dataclass(frozen=True)
class ColourScale:
    """Specify how to colour a rendered panel, for a variable or a difference.

    Attributes:
        cmap: Matplotlib colourmap name (e.g., "viridis", "RdBu_r").
        vmin: Lower bound for the colour scale.
        vmax: Upper bound for the colour scale.
        units: Display units for the variable (e.g., "K", "m/s").

    """

    cmap: str
    vmin: float | None = None
    vmax: float | None = None
    units: str | None = None


@dataclass(frozen=True)
class Metadata:
    """Structured metadata extracted from training configuration.

    Attributes:
        model: Model name (if available).
        n_history_steps: Number of history steps used as model input window (days).
        n_samples: Number of training points per epoch (if available).
        trained_epochs: Number of epochs the model has been trained for.
        training_start: Training start date string (if available).
        training_end: Training end date string (if available).
        vars_by_source: Dictionary mapping dataset source names to lists of variable names.

    """

    model: str | None = None
    n_history_steps: int | None = None
    n_samples: int | None = None
    trained_epochs: int | None = None
    training_end: str | None = None
    training_start: str | None = None
    vars_by_source: dict[str, list[str]] | None = None

    @classmethod
    def from_dataset(
        cls,
        dataset: SupportsMetadataFromDataset,
        *,
        model_name: str | None = None,
        trained_epochs: int | None = None,
    ) -> "Metadata":
        """Build structured metadata from a dataset-like source."""
        vars_by_source = {ds.name: sorted(ds.variable_names) for ds in dataset.inputs}

        return cls(
            model=model_name,
            trained_epochs=trained_epochs,
            n_history_steps=dataset.n_history_steps,
            n_samples=len(dataset),
            training_end=str(dataset.end_date.astype("datetime64[D]")),
            training_start=str(dataset.start_date.astype("datetime64[D]")),
            vars_by_source=vars_by_source or None,
        )


@dataclass(frozen=True)
class ModelStepOutput(Mapping[str, Tensor]):
    """Output of a model step: prediction, target, and loss."""

    prediction: TensorNTCHW
    target: TensorNTCHW
    loss: Tensor

    def __getitem__(self, key: str) -> Tensor:
        """Get a tensor by key."""
        if key == "prediction":
            return self.prediction
        if key == "target":
            return self.target
        if key == "loss":
            return self.loss
        msg = f"Key {key} not found in ModelStepOutput"
        raise KeyError(msg)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of ModelStepOutput."""
        yield "prediction"
        yield "target"
        yield "loss"

    def __len__(self) -> int:
        """Return ModelStepOutput length."""
        return 3

    def copy(self) -> dict[str, Tensor]:
        """Return a plain dict copy, required by Lightning's manual optimization loop."""
        return dict(self)


@dataclass(frozen=True)
class PlotSpec:
    """Configure how sea-ice plots are rendered.

    Attributes:
        title_groundtruth: Title above the ground-truth panel.
        title_prediction: Title above the prediction panel.
        title_difference: Title above the difference panel.
        colourmap: colourmap used for GT/prediction panels.
        dpi: Dots per inch for figure rendering (default 300).
        include_difference: Whether to draw a difference panel.
        diff_mode: Difference definition (e.g. "signed", "absolute", "smape").
        selected_timestep: Slice index when a single timestep is needed.
        vmin: Lower bound for GT/prediction colour scale (None = infer).
        vmax: Upper bound for GT/prediction colour scale (None = infer).
        include_ice_edge: Whether to overlay the sea ice edge contour in red.
        ice_edge_threshold: Concentration value defining the sea ice edge contour.
        uncertainty_variables: Maps each target variable to the input variable
            holding its reported standard uncertainty (used for the z-score panel).

    """

    title_groundtruth: str = "Ground Truth"
    title_prediction: str = "Prediction"
    title_difference: str = "Difference"

    colourmap: str = "viridis"
    dpi: int = 300

    # Difference pane
    include_difference: bool = True
    diff_mode: DiffMode = DiffMode.SIGNED
    selected_timestep: int = 0

    # Colourscale ranges: defaults to [0,1]
    vmin: float | None = 0.0
    vmax: float | None = 1.0

    # Sea ice edge overlay
    include_ice_edge: bool = False
    ice_edge_threshold: float = SEA_ICE_THRESHOLD

    # Optional metadata for titling
    hemisphere: Hemisphere | None = None

    # Video settings
    video_fps: int = 2
    video_format: Literal["mp4", "gif"] = "mp4"

    # Per-variable styles
    per_variable_styles: dict[str, dict[str, str | float | bool]] = field(
        default_factory=lambda: {
            # Sea ice concentration
            "sic-osisaf:ice_conc": {"cmap": "Blues_r"},
            "sic-ssmis:ice_conc": {"cmap": "Blues_r"},
        }
    )

    # Uncertainty variable lookup, for the z-score panel
    uncertainty_variables: dict[str, str] = field(
        default_factory=lambda: {"ice_conc": "total_standard_uncertainty"}
    )

    def __add__(
        self, other: "PlotSpec | DictConfig | dict[str, Any] | None"
    ) -> "PlotSpec":
        """Combine two PlotSpec instances or a PlotSpec with a dictionary."""
        if other is None:
            return self
        if isinstance(other, PlotSpec):
            dict_other = asdict(other)
        elif isinstance(other, DictConfig):
            dict_other = cast("dict[str, Any]", OmegaConf.to_container(other))
        else:
            dict_other = dict(other)
        return PlotSpec(**(asdict(self) | dict_other))


class Timespan:
    """A span of time."""

    MIN_DATES_FOR_FREQUENCY = 2

    def __init__(self, dates: Iterable[datetime]) -> None:
        """Initialise a timespan with a series of dates."""
        self._dates = list(dates)

    @cached_property
    def end(self) -> datetime:
        """Return the end date of the timespan."""
        return self._dates[-1]

    @cached_property
    def frequency(self) -> timedelta | None:
        """Return the spacing between consecutive steps, or None if unknown.

        Assumes uniform spacing derived from the first and last dates.
        """
        if len(self._dates) < self.MIN_DATES_FOR_FREQUENCY:
            return None
        return (self.end - self.start) / (self.steps - 1)

    @cached_property
    def start(self) -> datetime:
        """Return the start date of the timespan."""
        return self._dates[0]

    @cached_property
    def steps(self) -> int:
        """Return the number of steps (dates) in the timespan."""
        return len(self._dates)

    def __getitem__(self, step: int) -> "datetime":
        """Return the date at a given step index from the start of the timespan."""
        return self._dates[step]
