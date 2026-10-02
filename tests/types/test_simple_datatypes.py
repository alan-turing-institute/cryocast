from unittest.mock import MagicMock

import numpy as np
import torch
from anemoi.datasets.create.recipe import Recipe

from icenet_mp.types import (
    AnemoiCleanupArgs,
    AnemoiDatasetStatus,
    AnemoiFinaliseArgs,
    AnemoiInitArgs,
    AnemoiInspectArgs,
    AnemoiLoadArgs,
    Metadata,
    ProcessorOutput,
)
from icenet_mp.types.protocols import (
    SupportsMetadataFromDataset,
    SupportsMetadataInput,
)


def fake_metadata_source_input(
    name: str, variable_names: list[str]
) -> SupportsMetadataInput:
    """Return a duck-typed `SupportsMetadataInput` stand-in."""

    class FakeMetadataSourceInput:
        def __init__(self, name: str, variable_names: list[str]) -> None:
            self.name = name
            self.variable_names = variable_names

    return FakeMetadataSourceInput(name, variable_names)


def fake_metadata_source(
    *,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-10",
    frequency: np.timedelta64 | None = None,
    length: int = 10,
    n_history_steps: int = 0,
    inputs: list[SupportsMetadataInput] | None = None,
) -> SupportsMetadataFromDataset:
    """Return a duck-typed `SupportsMetadataFromDataset` stand-in."""

    class FakeMetadataSource:
        def __init__(self) -> None:
            self.start_date = np.datetime64(start_date)
            self.end_date = np.datetime64(end_date)
            self.frequency = (
                frequency if frequency is not None else np.timedelta64(1, "D")
            )
            self.n_history_steps = n_history_steps
            self.inputs = inputs if inputs is not None else []

        def __len__(self) -> int:
            return length

    return FakeMetadataSource()


class TestAnemoiCommandArgs:
    """Tests for the Anemoi CLI command argument dataclasses."""

    def test_cleanup_defaults(self) -> None:
        """Default AnemoiCleanupArgs command when omitted."""
        args = AnemoiCleanupArgs(path="dataset.zarr")

        assert args.command == "unused"

    def test_finalise_defaults(self) -> None:
        """Default AnemoiFinaliseArgs command while preserving the recipe."""
        recipe = MagicMock(spec=Recipe)

        args = AnemoiFinaliseArgs(path="dataset.zarr", recipe=recipe)

        assert args.command == "unused"
        assert args.recipe is recipe

    def test_init_defaults(self) -> None:
        """Default AnemoiInitArgs command and overwrite while preserving the recipe."""
        recipe = MagicMock(spec=Recipe)

        args = AnemoiInitArgs(path="dataset.zarr", recipe=recipe)

        assert args.command == "unused"
        assert args.overwrite is False
        assert args.recipe is recipe

    def test_inspect_preserves_requested_flags(self) -> None:
        """Preserve every explicit inspect flag without hidden defaults."""
        args = AnemoiInspectArgs(
            detailed=True,
            path="dataset.zarr",
            progress=False,
            size=True,
            statistics=False,
        )

        assert args.detailed is True
        assert args.path == "dataset.zarr"
        assert args.progress is False
        assert args.size is True
        assert args.statistics is False

    def test_load_defaults(self) -> None:
        """Default AnemoiLoadArgs command while preserving the recipe."""
        recipe = MagicMock(spec=Recipe)

        args = AnemoiLoadArgs(path="dataset.zarr", recipe=recipe)

        assert args.command == "unused"
        assert args.recipe is recipe


class TestAnemoiDatasetStatus:
    """Tests for AnemoiDatasetStatus."""

    def test_is_tuple_compatible(self) -> None:
        """Preserve tuple compatibility for Anemoi dataset status values."""
        status = AnemoiDatasetStatus(
            copy_in_progress=False,
            download_complete=True,
            is_finalised=True,
        )

        assert tuple(status) == (False, True, True)
        assert status.download_complete is True


class TestMetadata:
    """Tests for Metadata."""

    def test_accepts_training_summary_fields(self) -> None:
        """Accept and preserve training-summary metadata fields."""
        metadata = Metadata(
            model="cnn-vit-cnn",
            trained_epochs=7,
            training_start="2017-01-01",
            training_end="2019-12-31",
            n_samples=1095,
            n_history_steps=3,
            vars_by_source={"sic-ssmis": ["ice_conc"]},
        )

        assert metadata.model == "cnn-vit-cnn"
        assert metadata.trained_epochs == 7
        assert metadata.n_history_steps == 3
        assert metadata.vars_by_source == {"sic-ssmis": ["ice_conc"]}

    def test_defaults_are_independent_and_optional(self) -> None:
        """Keep Metadata defaults optional and independent across instances."""
        first = Metadata(vars_by_source={"era5": ["2t"]})
        second = Metadata()

        assert first.model is None
        assert first.n_samples is None
        assert first.vars_by_source == {"era5": ["2t"]}
        assert second.vars_by_source is None


class TestMetadataFromDataset:
    """Tests for Metadata.from_dataset."""

    def test_derives_dates_and_length_from_dataset(self) -> None:
        """Metadata fields come from the dataset's realised state, not from config."""
        dataset = fake_metadata_source(
            start_date="2020-01-01T12:30:00",
            end_date="2020-01-10T00:00:00",
            length=10,
            n_history_steps=3,
        )

        metadata = Metadata.from_dataset(dataset, model_name="unet", trained_epochs=5)

        assert metadata.model == "unet"
        assert metadata.trained_epochs == 5
        assert metadata.training_start == "2020-01-01"
        assert metadata.training_end == "2020-01-10"
        assert metadata.n_samples == 10
        assert metadata.n_history_steps == 3

    def test_defaults_model_and_epoch_to_none(self) -> None:
        """Omitted model_name/trained_epochs fall back to None."""
        dataset = fake_metadata_source()

        metadata = Metadata.from_dataset(dataset)

        assert metadata.model is None
        assert metadata.trained_epochs is None

    def test_collects_sorted_variable_names_by_source(self) -> None:
        """vars_by_source maps each input dataset's name to its sorted variable names."""
        dataset = fake_metadata_source(
            inputs=[
                fake_metadata_source_input("era5", ["sp", "2t"]),
                fake_metadata_source_input("osisaf-south", ["sic"]),
            ]
        )

        metadata = Metadata.from_dataset(dataset)

        assert metadata.vars_by_source == {
            "era5": ["2t", "sp"],
            "osisaf-south": ["sic"],
        }

    def test_empty_inputs_yields_none_vars_by_source(self) -> None:
        """No input datasets means no variable-by-source mapping."""
        dataset = fake_metadata_source(inputs=[])

        metadata = Metadata.from_dataset(dataset)

        assert metadata.vars_by_source is None


class TestProcessorOutput:
    """Tests for ProcessorOutput."""

    def test_defaults_loss_to_none(self) -> None:
        """Default ProcessorOutput loss to None when omitted."""
        prediction = torch.randn(2, 3, 4, 5, 6)

        output = ProcessorOutput(prediction=prediction)

        assert output.prediction is prediction
        assert output.loss is None

    def test_keeps_custom_loss_tensor(self) -> None:
        """Preserve an explicitly supplied ProcessorOutput loss tensor."""
        prediction = torch.randn(1, 2, 3, 4, 5)
        loss = torch.tensor(0.25)

        output = ProcessorOutput(prediction=prediction, loss=loss)

        assert output.loss is loss
