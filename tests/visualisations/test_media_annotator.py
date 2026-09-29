from datetime import date, datetime

from icenet_mp.types import Hemisphere, Metadata, PlotSpec, Timespan
from icenet_mp.visualisations.media_annotator import MediaAnnotator


class TestFormatTitle:
    def test_with_hemisphere_and_units(self) -> None:
        """Include hemisphere and units when both are given."""
        annotator = MediaAnnotator(Metadata(), PlotSpec(hemisphere=Hemisphere.NORTH))

        result = annotator.header_for_variable(
            units="K", variable_name="2t", when=date(2020, 1, 1)
        )

        assert result == "2t [K] (north) on 2020-01-01"

    def test_without_hemisphere_or_units(self) -> None:
        """Omit hemisphere and units segments when neither is given."""
        annotator = MediaAnnotator(Metadata(), PlotSpec(hemisphere=None))

        result = annotator.header_for_variable(
            units=None, variable_name="2t", when=date(2020, 1, 1)
        )

        assert result == "2t on 2020-01-01"

    def test_accepts_datetime(self) -> None:
        """Accept a datetime and format only its date portion."""
        annotator = MediaAnnotator(Metadata(), PlotSpec(hemisphere=None))

        result = annotator.header_for_variable(
            units=None, variable_name="2t", when=datetime(2020, 1, 1, 12, 30)
        )

        assert result == "2t on 2020-01-01"


class TestDescribeVariable:
    def test_replaces_underscores_with_spaces(self) -> None:
        """Turn a snake_case variable name into a space-separated name."""
        result = MediaAnnotator(Metadata(), PlotSpec()).describe_variable(
            "sea_ice_concentration", units=None
        )

        assert result == "sea ice concentration"

    def test_empty_string_stays_empty(self) -> None:
        """Return an empty string unchanged."""
        result = MediaAnnotator(Metadata(), PlotSpec()).describe_variable(
            "", units=None
        )

        assert result == ""

    def test_includes_units_and_hemisphere(self) -> None:
        """Append units and hemisphere suffixes when both are given."""
        annotator = MediaAnnotator(Metadata(), PlotSpec(hemisphere=Hemisphere.SOUTH))

        result = annotator.describe_variable("ice_conc", units="%")

        assert result == "ice conc [%] (south)"


class TestHeader:
    def test_combines_variable_and_dates(self) -> None:
        """Combine the variable/hemisphere line and the dates line into the header."""
        history_ctx = Timespan(
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        )
        annotator = MediaAnnotator(Metadata(), PlotSpec(hemisphere=Hemisphere.NORTH))

        result = annotator.header(
            forecast_date=datetime(2020, 1, 6),
            history_ctx=history_ctx,
            variable_name="sea_ice_concentration",
        )

        assert result == (
            "sea ice concentration (north)\n"
            "History: 2020-01-01 - 2020-01-03 (3 steps)   "
            "Leadtime (+3 steps): 2020-01-06"
        )


class TestDescribeDates:
    def test_formats_history_window_and_leadtime(self) -> None:
        """Combine the history window and the leadtime into one line."""
        history_ctx = Timespan(
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        )
        annotator = MediaAnnotator(Metadata(), PlotSpec())

        result = annotator.describe_dates(datetime(2020, 1, 6), history_ctx)

        assert result == (
            "History: 2020-01-01 - 2020-01-03 (3 steps)   "
            "Leadtime (+3 steps): 2020-01-06"
        )

    def test_counts_leadtime_in_steps_for_sub_daily_cadence(self) -> None:
        """Count the leadtime in steps, not calendar days, for sub-daily data.

        A history window spaced 6 hours apart with a forecast one step past
        the end spans less than a full calendar day, so `.days` would floor
        it to 0 steps even though it is genuinely +1 step.
        """
        history_ctx = Timespan(
            [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 1, 18),
            ]
        )
        annotator = MediaAnnotator(Metadata(), PlotSpec())

        result = annotator.describe_dates(datetime(2020, 1, 2, 0), history_ctx)

        assert result is not None
        assert "Leadtime (+1 steps): 2020-01-02" in result

    def test_falls_back_to_plain_date_without_history_context(self) -> None:
        """Without a history context, just state the date rather than a leadtime."""
        annotator = MediaAnnotator(Metadata(), PlotSpec())

        result = annotator.describe_dates(datetime(2020, 1, 6), None)

        assert result == "Date: 2020-01-06"


class TestDescribeModel:
    def test_formats_model(self) -> None:
        """Format the model name on its own line."""
        metadata = Metadata(model="unet")

        result = MediaAnnotator(metadata, PlotSpec()).describe_model()

        assert result == "Model: unet"

    def test_returns_none_when_absent(self) -> None:
        """Return None when model is not set."""
        assert MediaAnnotator(Metadata(), PlotSpec()).describe_model() is None


class TestDescribeTraining:
    def test_formats_dates_samples_and_epochs(self) -> None:
        """Format the training date range, sample count and epoch count."""
        metadata = Metadata(
            training_start="2020-01-01",
            training_end="2020-01-10",
            n_samples=10,
            trained_epochs=16,
        )

        result = MediaAnnotator(metadata, PlotSpec()).describe_training()

        assert (
            result == "Trained: 2020-01-01 — 2020-01-10 (16 epoch(s), 10 samples/epoch)"
        )

    def test_samples_without_epochs(self) -> None:
        """Omit the epoch count when trained_epochs is unset."""
        metadata = Metadata(n_samples=10)

        result = MediaAnnotator(metadata, PlotSpec()).describe_training()

        assert result == "(10 samples/epoch)"

    def test_epochs_without_samples(self) -> None:
        """Omit the sample count when n_samples is unset."""
        metadata = Metadata(trained_epochs=16)

        result = MediaAnnotator(metadata, PlotSpec()).describe_training()

        assert result == "(16 epoch(s))"

    def test_dates_without_samples(self) -> None:
        """Omit the parenthetical entirely when n_samples is unset."""
        metadata = Metadata(training_start="2020-01-01", training_end="2020-01-10")

        result = MediaAnnotator(metadata, PlotSpec()).describe_training()

        assert result == "Trained: 2020-01-01 — 2020-01-10"

    def test_returns_none_when_absent(self) -> None:
        """Return None when no training facts are set."""
        assert MediaAnnotator(Metadata(), PlotSpec()).describe_training() is None


class TestDescribeDatasets:
    def test_lists_sources_with_variable_counts(self) -> None:
        """List each source with its variable count, sorted by source name."""
        metadata = Metadata(
            vars_by_source={"osisaf": ["ice_conc"], "era5": ["2t", "sp"]}
        )

        result = MediaAnnotator(metadata, PlotSpec()).describe_datasets()

        assert result == "Input datasets: era5 (2 variables) osisaf (1 variables)"

    def test_lists_source_with_no_variables(self) -> None:
        """List a source with an empty variable list without parentheses."""
        metadata = Metadata(vars_by_source={"era5": []})

        result = MediaAnnotator(metadata, PlotSpec()).describe_datasets()

        assert result == "Input datasets: era5"

    def test_returns_none_when_absent(self) -> None:
        """Return None when no dataset metadata is set."""
        assert MediaAnnotator(Metadata(), PlotSpec()).describe_datasets() is None


class TestFooter:
    def test_includes_model_training_and_datasets(self) -> None:
        """Combine model, training and dataset descriptions into the footer."""
        metadata = Metadata(
            model="test_model",
            trained_epochs=5,
            training_start="2020-01-01",
            training_end="2020-01-10",
            n_samples=10,
            vars_by_source={"era5": ["2t", "sp"]},
        )

        footer = MediaAnnotator(metadata, PlotSpec()).footer()

        assert "Model: test_model" in footer
        assert "Trained: 2020-01-01 — 2020-01-10" in footer
        assert "10 samples/epoch" in footer
        assert "5 epoch(s)" in footer
        assert "Input datasets: era5 (2 variables)" in footer

    def test_omits_missing_fields(self) -> None:
        """Only include lines for the metadata facts that are actually set."""
        footer = MediaAnnotator(Metadata(model="unet"), PlotSpec()).footer()

        assert footer == "Model: unet"

    def test_empty_when_metadata_has_no_facts(self) -> None:
        """Return an empty string when the bound metadata formats to nothing."""
        footer = MediaAnnotator(Metadata(), PlotSpec()).footer()

        assert footer == ""
