from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Literal, Self, cast

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .annotations import TensorNTCHW
from .constants import SEA_ICE_THRESHOLD
from .enums import DiffMode, Hemisphere, RolloutSpace, SkipAnchorType
from .protocols import SupportsMetadataFromDataset


class DataSpace:
    """Immutable description of a CHW data space, compared and hashed by value."""

    def __init__(self, channels: int, name: str, shape: Sequence[int]) -> None:
        """Initialise a DataSpace from channels, name and shape."""
        self._channels = int(channels)
        self._name = name
        self._shape = (int(shape[0]), int(shape[1]))

    def __eq__(self, other: object) -> bool:
        """Compare DataSpaces by value rather than by identity."""
        if not isinstance(other, DataSpace):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        """Hash consistently with __eq__."""
        return hash(self._key)

    def __repr__(self) -> str:
        """Show the DataSpace values."""
        return (
            f"DataSpace(channels={self.channels}, name={self.name!r}, "
            f"shape={self.shape})"
        )

    @property
    def _key(self) -> tuple[int, str, tuple[int, int]]:
        return (self._channels, self._name, self._shape)

    @property
    def area(self) -> int:
        """Return the area of the data space."""
        return self.shape[0] * self.shape[1]

    @property
    def channels(self) -> int:
        """Return the number of channels."""
        return self._channels

    @property
    def chw(self) -> tuple[int, int, int]:
        """Return a tuple of [channels, height, width]."""
        return (self.channels, *self.shape)

    @property
    def name(self) -> str:
        """Return the name."""
        return self._name

    @property
    def shape(self) -> tuple[int, int]:
        """Return the (height, width) shape."""
        return self._shape

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
        show_first_lead: Whether static plots show the first lead time.
        show_last_lead: Whether static plots show the last lead time.
        vmin: Lower bound for GT/prediction colour scale (None = infer).
        vmax: Upper bound for GT/prediction colour scale (None = infer).
        include_ice_edge: Whether to overlay the sea ice edge contour in red.
        ice_edge_threshold: Concentration value defining the sea ice edge contour.
        uncertainty_variables: Maps each target variable to the variable in the same
            dataset group holding its reported standard uncertainty (used for the
            z-score panel). This does not need to be selected as an input variable.

    """

    title_groundtruth: str = "Ground Truth"
    title_prediction: str = "Prediction"
    title_difference: str = "Difference"

    colourmap: str = "viridis"
    dpi: int = 300

    # Difference pane
    include_difference: bool = True
    diff_mode: DiffMode = DiffMode.SIGNED

    # Lead times for static plots
    show_first_lead: bool = True
    show_last_lead: bool = True

    # Colourscale ranges: defaults to [0,1]
    vmin: float | None = 0.0
    vmax: float | None = 1.0

    # Sea ice edge overlay
    include_ice_edge: bool = True
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

    def static_lead_indices(self, n_leads: int) -> list[int]:
        """Return the indices of the lead times that static plots show.

        Args:
            n_leads: The number of lead times in the forecast.

        Returns:
            The selected indices, in order.

        """
        return sorted(
            set([0] * self.show_first_lead + [n_leads - 1] * self.show_last_lead)
        )


@dataclass(frozen=True)
class RolloutStrategy:
    """Describe how EncodeProcessDecode advances its forecast loop.

    Attributes:
        space: Where the forecast loop runs: in latent space, or in physical space
            where each step is decoded and re-encoded.
        anchor: The frame that a finaliser skip connection adds to each forecast step:
            the last observed frame (INITIAL) or the previous forecast step (PREVIOUS).
            A physical rollout needs a PREVIOUS anchor.
        relaxation_tau: The folding time, in forecast steps, of the relaxation of each
            forecast step towards climatology, or None for no relaxation. Relaxation
            needs a PREVIOUS anchor, so that it carries over between steps.

    """

    space: RolloutSpace
    anchor: SkipAnchorType
    relaxation_tau: float | None = None

    def __post_init__(self) -> None:
        """Validate the anchor and the relaxation."""
        if (
            self.space == RolloutSpace.PHYSICAL
            and self.anchor != SkipAnchorType.PREVIOUS
        ):
            msg = (
                f"rollout_space='physical' requires skip_anchor='previous', but it is "
                f"'{self.anchor.value}'. A physical rollout anchors each step on the "
                f"previous forecast step."
            )
            raise ValueError(msg)
        if self.relaxation_tau is not None and self.anchor != SkipAnchorType.PREVIOUS:
            msg = (
                f"relaxation requires skip_anchor='previous', but it is "
                f"'{self.anchor.value}'. With a fixed anchor, the relaxation does not "
                f"carry over from one step to the next."
            )
            raise ValueError(msg)

    @classmethod
    def from_options(
        cls,
        space: RolloutSpace | str = RolloutSpace.LATENT,
        anchor: SkipAnchorType | str | None = None,
        relaxation_tau: float | None = None,
    ) -> Self:
        """Create a RolloutStrategy from enums or strings.

        If the anchor is unset, use the default anchor for the space. A latent rollout
        anchors every step on the last observed frame. A physical rollout anchors each
        step on the previous forecast step.
        """
        space = RolloutSpace(space)
        if anchor is None:
            anchor = (
                SkipAnchorType.PREVIOUS
                if space == RolloutSpace.PHYSICAL
                else SkipAnchorType.INITIAL
            )
        return cls(
            space=space, anchor=SkipAnchorType(anchor), relaxation_tau=relaxation_tau
        )

    @property
    def is_physical(self) -> bool:
        """Return True if the forecast loop runs in physical space."""
        return self.space == RolloutSpace.PHYSICAL


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
