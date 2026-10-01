import logging
from datetime import date, datetime

from icenet_mp.types import Metadata, PlotSpec, Timespan
from icenet_mp.utils import iso_from_date

logger = logging.getLogger(__name__)


class MediaAnnotator:
    """Compose titles and footers for use in media."""

    def __init__(self, metadata: Metadata, plot_spec: PlotSpec) -> None:
        """Bind the metadata and plot spec shared by every title/footer."""
        self.plot_spec = plot_spec
        self.metadata = metadata

    @staticmethod
    def multiline_string(*parts: str | None) -> str:
        """Join strings with newlines, dropping None values."""
        return "\n".join(part for part in parts if part is not None)

    def describe_datasets(self) -> str | None:
        """Describe the input datasets.

        Returns:
            Formatted string like:

            Input datasets: <source> (<num vars>) <source> (<num vars>)

        """
        parts = [
            f"{source} ({len(variables)} variables)" if variables else source
            for source, variables in sorted(
                (self.metadata.vars_by_source or {}).items()
            )
        ]
        if parts:
            return " ".join(["Input datasets:", *parts])
        return None

    def describe_dates(
        self, forecast_date: datetime, history_ctx: Timespan | None
    ) -> str:
        """Describe the history window and leadtime for a forecast date.

        Returns:
            Formatted string like:

            History: YYYY-MM-DD - YYYY-MM-DD (<num steps> steps)   Leadtime (+<leadtime> steps): YYYY-MM-DD

            OR

            Date: YYYY-MM-DD

        """
        if not history_ctx:
            return f"Date: {iso_from_date(forecast_date)}"
        elapsed = forecast_date - history_ctx.end
        leadtime = (
            round(elapsed / history_ctx.frequency)
            if history_ctx.frequency is not None
            else elapsed.days
        )
        hspan = f"{iso_from_date(history_ctx.start)} - {iso_from_date(history_ctx.end)}"
        return (
            f"History: {hspan} ({history_ctx.steps} steps)"
            "   "
            f"Leadtime (+{leadtime} steps): {iso_from_date(forecast_date)}"
        )

    def describe_model(self) -> str | None:
        """Describe the model.

        Returns:
            Formatted string like:

            Model: <model>

        """
        if self.metadata.model:
            return f"Model: {self.metadata.model}"
        return None

    def describe_training(self) -> str | None:
        """Describe the training process.

        Returns:
            Formatted string like:

            Trained: <start> — <end> (<num> epoch(s), <num> samples/epoch)

        """
        parts: list[str] = []
        if self.metadata.training_start or self.metadata.training_end:
            parts.append(
                f"Trained: {self.metadata.training_start or '?'} — {self.metadata.training_end or '?'}"
            )
        epochs = (
            f"{self.metadata.trained_epochs} epoch(s)"
            if self.metadata.trained_epochs is not None
            else None
        )
        samples = (
            f"{self.metadata.n_samples} samples/epoch"
            if self.metadata.n_samples is not None
            else None
        )
        if epochs or samples:
            parts.append(f"({', '.join(part for part in (epochs, samples) if part)})")
        if parts:
            return " ".join(parts)
        return None

    def describe_variable(self, variable_name: str, units: str | None) -> str:
        """Describe a single variable.

        Args:
            variable_name: Variable name.
            units: Display units for the variable.

        Returns:
            Formatted string like:

            <Variable> [<units>] (<Hemisphere>)

        """
        variable = variable_name.replace("_", " ").strip()
        units_s = f" [{units}]" if units else ""
        hemisphere = (
            f" ({self.plot_spec.hemisphere})" if self.plot_spec.hemisphere else ""
        )
        return f"{variable}{units_s}{hemisphere}"

    def footer(self) -> str:
        """Compose footer text for static and video figures.

        Returns:
            Formatted string like:

            Model: <model>
            Trained: <start> — <end> (<num> epoch(s), <num> samples/epoch)
            Input datasets: <source> (<num vars>) <source> (<num vars>)

        """
        return self.multiline_string(
            self.describe_model(), self.describe_training(), self.describe_datasets()
        )

    def header(
        self,
        *,
        forecast_date: datetime,
        history_ctx: Timespan | None,
        variable_name: str,
    ) -> str:
        """Compose header text for a forecast.

        Args:
            history_ctx: The history context for the model.
            forecast_date: The date of the forecast.
            variable_name: Variable name.

        Returns:
            Formatted string like:

            <Variable> (<Hemisphere>)
            History: YYYY-MM-DD - YYYY-MM-DD (<num steps> steps)   Leadtime (+<leadtime> steps): YYYY-MM-DD

        """
        return self.multiline_string(
            self.describe_variable(variable_name, units=None),
            self.describe_dates(forecast_date, history_ctx),
        )

    def header_for_variable(
        self,
        *,
        units: str | None,
        variable_name: str,
        when: date | datetime,
    ) -> str:
        """Compose header text for a standalone variable plot.

        Args:
            units: Display units for the variable.
            variable_name: Variable name.
            when: Date or datetime of the data.

        Returns:
            Formatted string like:

            <Variable> (<Hemisphere>) on YYYY-MM-DD

        """
        return f"{self.describe_variable(variable_name, units=units)} on {iso_from_date(when)}"
