import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from icenet_mp.callbacks.media_logging_callback import MediaLoggingCallback
from icenet_mp.data import CombinedDataset
from icenet_mp.models import BaseModel
from icenet_mp.types import ModelStepOutput, PlotSpec


@pytest.fixture
def make_plots_args(mock_trainer: MagicMock) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Configure mock_trainer and build a matching pl_module/dataset pair for make_plots tests."""
    mock_trainer.current_epoch = 0
    mock_trainer.testing = False
    mock_trainer.loggers = []
    mock_trainer.datamodule = None
    pl_module = MagicMock(spec=BaseModel)
    pl_module.hemisphere = "south"
    dataset = MagicMock(spec=CombinedDataset)
    dataset.dates = [np.datetime64("2020-01-01T12:00:00")]
    dataset.get_forecast_steps.return_value = [np.datetime64("2020-01-01T12:00:00")]
    dataset.inputs = []
    dataset.start_date = np.datetime64("2020-01-01")
    dataset.end_date = np.datetime64("2020-01-10")
    dataset.frequency = np.timedelta64(1, "D")
    dataset.n_history_steps = 1
    dataset.__len__.return_value = 10
    return mock_trainer, pl_module, dataset


def _stub_media_publisher(
    callback: MediaLoggingCallback, monkeypatch: pytest.MonkeyPatch
) -> dict[str, MagicMock]:
    """Replace load_target_uncertainties and MediaPublisher (as make_plots constructs it) with spies.

    `make_plots` now builds a fresh `MediaPublisher(...)` locally on every call rather than
    holding one on the callback, so the class itself is replaced with a MagicMock: its
    `call_args_list` records each construction's kwargs (dataset/hemisphere/land_mask/
    model_name/trained_epochs), and `.return_value` is the stub instance whose
    log_* methods `make_plots` calls.
    """
    publisher_class = MagicMock()
    publisher = publisher_class.return_value
    monkeypatch.setattr(
        "icenet_mp.callbacks.media_logging_callback.MediaPublisher", publisher_class
    )
    mocks = {
        "load_target_uncertainties": MagicMock(return_value={}),
        "log_static_outputs": publisher.log_static_outputs,
        "log_static_inputs": publisher.log_static_inputs,
        "log_video_outputs": publisher.log_video_outputs,
        "log_video_inputs": publisher.log_video_inputs,
        "media_publisher_class": publisher_class,
    }
    monkeypatch.setattr(
        callback, "load_target_uncertainties", mocks["load_target_uncertainties"]
    )
    return mocks


@pytest.fixture
def dataset_with_uncertainty() -> tuple[MagicMock, MagicMock]:
    """A dataset double with one target/uncertainty variable pair (ice_conc)."""
    dataset = MagicMock()
    target = MagicMock()
    target.name = "target"
    target.variable_names = ["ice_conc"]
    target.statistics = {"minimum": [0.0], "maximum": [2.0]}
    dataset.target = target

    source = MagicMock()
    source.name = "target"
    source.variable_names = ["ice_conc", "total_standard_uncertainty"]
    uncertainty_ds = MagicMock()
    uncertainty_ds.get_tchw.return_value = np.array(
        [[[[0.1, 0.2], [0.3, 1.1]]]], dtype=np.float32
    )
    source.subset.return_value = uncertainty_ds
    dataset.inputs = [source]
    return dataset, uncertainty_ds


class TestLoadTargetUncertainties:
    def test_scales_and_masks(
        self, dataset_with_uncertainty: tuple[MagicMock, MagicMock]
    ) -> None:
        """Scale source uncertainty to target space and mask invalid values."""
        dataset, _ = dataset_with_uncertainty

        result = MediaLoggingCallback().load_target_uncertainties(
            dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
        )

        assert set(result) == {0}
        np.testing.assert_allclose(
            result[0],
            np.array([[[0.05, 0.1], [0.15, np.nan]]]),
            equal_nan=True,
        )
        dataset.inputs[0].subset.assert_called_once_with(
            variables=["total_standard_uncertainty"], normalise=False
        )

    def test_skips_missing_source(
        self, dataset_with_uncertainty: tuple[MagicMock, MagicMock]
    ) -> None:
        """Return no uncertainty when the matching target input is unavailable."""
        dataset, _ = dataset_with_uncertainty
        dataset.inputs[0].name = "other"

        result = MediaLoggingCallback().load_target_uncertainties(
            dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
        )

        assert result == {}
        dataset.inputs[0].subset.assert_not_called()

    def test_skips_when_target_variable_not_present(
        self, dataset_with_uncertainty: tuple[MagicMock, MagicMock]
    ) -> None:
        """Skip uncertainty loading when the target variable itself isn't in the dataset."""
        dataset, _ = dataset_with_uncertainty
        dataset.target.variable_names = ["other_variable"]

        result = MediaLoggingCallback().load_target_uncertainties(
            dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
        )

        assert result == {}
        dataset.inputs[0].subset.assert_not_called()

    def test_handles_data_error(
        self,
        dataset_with_uncertainty: tuple[MagicMock, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Skip uncertainty plotting when the source read fails."""
        dataset, uncertainty_ds = dataset_with_uncertainty
        uncertainty_ds.get_tchw.side_effect = ValueError("missing uncertainty")

        with caplog.at_level(logging.WARNING):
            result = MediaLoggingCallback().load_target_uncertainties(
                dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
            )

        assert result == {}
        assert "Could not load target uncertainty" in caplog.text

    def test_handles_missing_statistics(
        self,
        dataset_with_uncertainty: tuple[MagicMock, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Skip uncertainty plotting when target statistics are missing a key."""
        dataset, _ = dataset_with_uncertainty
        dataset.target.statistics = {}

        with caplog.at_level(logging.WARNING):
            result = MediaLoggingCallback().load_target_uncertainties(
                dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
            )

        assert result == {}
        assert "Could not load target uncertainty" in caplog.text

    def test_rejects_invalid_target_range(
        self,
        dataset_with_uncertainty: tuple[MagicMock, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Skip scaling when the target normalisation range is invalid."""
        dataset, _ = dataset_with_uncertainty
        dataset.target.statistics = {"minimum": [1.0], "maximum": [1.0]}

        with caplog.at_level(logging.WARNING):
            result = MediaLoggingCallback().load_target_uncertainties(
                dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
            )

        assert result == {}
        assert "Could not scale target uncertainty" in caplog.text

    def test_uses_uncertainty_variables_from_plot_spec(
        self, dataset_with_uncertainty: tuple[MagicMock, MagicMock]
    ) -> None:
        """The target/uncertainty variable mapping comes from plot_spec, not a hardcoded default."""
        dataset, _ = dataset_with_uncertainty
        callback = MediaLoggingCallback(plot_spec=PlotSpec(uncertainty_variables={}))

        result = callback.load_target_uncertainties(
            dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
        )

        assert result == {}


class TestInit:
    """Tests for MediaLoggingCallback construction."""

    def test_defaults_frequency_to_negative_one_when_not_given(self) -> None:
        """Disable all frequency-based triggers when no frequency dict is given."""
        callback = MediaLoggingCallback()

        assert callback.frequency_batch == -1
        assert callback.frequency_epoch == -1
        assert callback.frequency_number == -1

    def test_parses_frequency_dict(self) -> None:
        """Read batch/epoch/number frequencies from the given dict."""
        callback = MediaLoggingCallback(frequency={"batch": 5, "epoch": 2, "number": 3})

        assert callback.frequency_batch == 5
        assert callback.frequency_epoch == 2
        assert callback.frequency_number == 3

    def test_stores_plot_toggles_and_prefix(self) -> None:
        """Store the plot-type toggles and key prefix as configured."""
        callback = MediaLoggingCallback(
            make_input_plots=True,
            make_static_plots=False,
            make_video_plots=False,
            prefix="eval",
        )

        assert callback.make_input_plots is True
        assert callback.make_static_plots is False
        assert callback.make_video_plots is False
        assert callback.prefix == "eval"


class TestCacheBatch:
    """Tests for cache_batch."""

    def test_caches_when_outputs_is_a_mapping(self) -> None:
        """Cache batch index, dataloader index, and outputs when given a mapping."""
        callback = MediaLoggingCallback()
        outputs = {
            "prediction": torch.zeros(1, 1, 1, 2, 2),
            "target": torch.ones(1, 1, 1, 2, 2),
            "loss": torch.tensor(0.0),
        }

        callback.cache_batch(3, 1, outputs)

        assert callback.cached_batch_idx_ == 3
        assert callback.cached_dataloader_idx_ == 1
        assert isinstance(callback.cached_outputs_, ModelStepOutput)

    def test_does_not_cache_when_outputs_is_not_a_mapping(self) -> None:
        """Leave the cache untouched when outputs is not a mapping."""
        callback = MediaLoggingCallback()

        callback.cache_batch(3, 1, torch.tensor(0.0))

        assert callback.cached_batch_idx_ is None
        assert callback.cached_dataloader_idx_ is None
        assert callback.cached_outputs_ is None


class TestIsSampleBatch:
    """Tests for is_sample_batch."""

    def test_returns_false_when_frequency_number_not_positive(self) -> None:
        """Never select a batch when sampling is disabled."""
        callback = MediaLoggingCallback(frequency={"number": 0})

        assert callback.is_sample_batch(0, 10) is False

    @pytest.mark.parametrize(
        "total_batches", [float("inf"), float("nan")], ids=["inf", "nan"]
    )
    def test_returns_false_when_total_batches_not_finite(
        self, total_batches: float
    ) -> None:
        """Never select a batch when the total batch count is not finite."""
        callback = MediaLoggingCallback(frequency={"number": 3})

        assert callback.is_sample_batch(0, total_batches) is False

    def test_returns_false_when_total_batches_not_positive(self) -> None:
        """Never select a batch when there are no batches to sample from."""
        callback = MediaLoggingCallback(frequency={"number": 3})

        assert callback.is_sample_batch(0, 0) is False

    def test_single_target_selects_last_batch(self) -> None:
        """Sample only the final batch when frequency_number resolves to one target."""
        callback = MediaLoggingCallback(frequency={"number": 1})

        assert callback.is_sample_batch(4, 5) is True
        assert callback.is_sample_batch(0, 5) is False

    @pytest.mark.parametrize(
        ("batch_idx", "expected"),
        [(0, True), (4, True), (9, True), (1, False), (5, False)],
        ids=["first", "middle", "last", "not-first", "not-middle"],
    )
    def test_selects_evenly_spaced_targets(
        self, batch_idx: int, *, expected: bool
    ) -> None:
        """Sample evenly-spaced batch indices across the epoch."""
        callback = MediaLoggingCallback(frequency={"number": 3})

        assert callback.is_sample_batch(batch_idx, 10) is expected


class TestLoadDataset:
    """Tests for load_dataset."""

    def test_returns_none_when_dataloader_is_none(self) -> None:
        """Return None when there is no dataloader to inspect."""
        callback = MediaLoggingCallback()
        callback.cached_dataloader_idx_ = 0

        assert callback.load_dataset(None) is None

    def test_returns_none_when_cached_dataloader_idx_is_none(self) -> None:
        """Return None when no batch has been cached yet."""
        callback = MediaLoggingCallback()

        assert callback.load_dataset(MagicMock(spec=DataLoader)) is None

    def test_indexes_into_sequence_of_dataloaders(self) -> None:
        """Select the dataloader at cached_dataloader_idx_ from a sequence."""
        callback = MediaLoggingCallback()
        callback.cached_dataloader_idx_ = 1
        dataset = MagicMock(spec=CombinedDataset)
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = dataset
        dataloader.batch_size = 4
        other_dataloader = MagicMock(spec=DataLoader)

        result = callback.load_dataset([other_dataloader, dataloader])

        assert result == (dataset, 4)

    def test_uses_single_dataloader_directly(self) -> None:
        """Use the dataloader directly when it is not a sequence."""
        callback = MediaLoggingCallback()
        callback.cached_dataloader_idx_ = 0
        dataset = MagicMock(spec=CombinedDataset)
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = dataset
        dataloader.batch_size = 2

        assert callback.load_dataset(dataloader) == (dataset, 2)

    def test_returns_none_and_warns_when_dataset_is_not_combined_dataset(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn and return None when the dataloader's dataset is the wrong type."""
        callback = MediaLoggingCallback()
        callback.cached_dataloader_idx_ = 0
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = object()
        dataloader.batch_size = 2

        with caplog.at_level(logging.WARNING):
            result = callback.load_dataset(dataloader)

        assert result is None
        assert "not CombinedDataset" in caplog.text

    def test_returns_none_and_warns_when_batch_size_is_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn and return None when the dataloader has no batch size."""
        callback = MediaLoggingCallback()
        callback.cached_dataloader_idx_ = 0
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = MagicMock(spec=CombinedDataset)
        dataloader.batch_size = None

        with caplog.at_level(logging.WARNING):
            result = callback.load_dataset(dataloader)

        assert result is None
        assert "does not have a batch size" in caplog.text


class TestMakePlots:
    """Tests for make_plots."""

    def test_warns_and_returns_when_no_cached_outputs(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Skip plotting entirely when no batch has been cached."""
        callback = MediaLoggingCallback()
        trainer, pl_module, dataset = make_plots_args

        with caplog.at_level(logging.WARNING):
            callback.make_plots(trainer, pl_module, dataset, 1)

        assert "Could not load outputs" in caplog.text

    def test_warns_and_returns_when_pl_module_is_not_base_model(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        mock_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Skip plotting when the module is not a BaseModel (no hemisphere info)."""
        callback = MediaLoggingCallback()
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, _pl_module, dataset = make_plots_args

        with caplog.at_level(logging.WARNING):
            callback.make_plots(trainer, mock_module, dataset, 1)

        assert "skipping plotting" in caplog.text

    def test_pushes_current_epoch_and_hemisphere_to_media_publisher(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """During training/validation, show the (1-indexed) count of completed epochs."""
        callback = MediaLoggingCallback()
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args
        trainer.current_epoch = 7

        callback.make_plots(trainer, pl_module, dataset, 1)

        construction_calls = stubs["media_publisher_class"].call_args_list
        assert any(c.kwargs.get("trained_epochs") == 8 for c in construction_calls)
        assert any(
            c.kwargs.get("plot_spec") is not None
            and c.kwargs["plot_spec"].hemisphere == "south"
            for c in construction_calls
        )

    def test_pushes_model_trained_epochs_to_media_publisher_during_evaluation(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """During evaluation, show how many epochs the loaded model was trained for.

        A fresh evaluate-only trainer's current_epoch is always 0, so this must come
        from the model instead (set by ModelService.from_checkpoint).
        """
        callback = MediaLoggingCallback()
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args
        trainer.current_epoch = 0
        trainer.testing = True
        pl_module.checkpoint_epoch = 15

        callback.make_plots(trainer, pl_module, dataset, 1)

        construction_calls = stubs["media_publisher_class"].call_args_list
        assert any(c.kwargs.get("trained_epochs") == 16 for c in construction_calls)

    def test_trained_epochs_is_none_when_checkpoint_epoch_is_unset(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass None rather than crashing when checkpoint_epoch was never set.

        This happens when evaluating a model that was never loaded from a checkpoint
        (checkpoint_epoch defaults to None on BaseModel).
        """
        callback = MediaLoggingCallback()
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args
        trainer.testing = True
        pl_module.checkpoint_epoch = None

        callback.make_plots(trainer, pl_module, dataset, 1)

        construction_calls = stubs["media_publisher_class"].call_args_list
        assert any(c.kwargs.get("trained_epochs") is None for c in construction_calls)

    def test_uses_datamodule_training_dataset_for_media_publisher_metadata(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Source the footer's "Trained:" metadata from the datamodule's training split, not the active dataset."""
        callback = MediaLoggingCallback()
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args
        training_dataset = MagicMock(spec=CombinedDataset)
        trainer.datamodule = MagicMock(
            training_dataset=training_dataset, mask_directory=None
        )

        callback.make_plots(trainer, pl_module, dataset, 1)

        construction_calls = stubs["media_publisher_class"].call_args_list
        assert any(
            c.kwargs.get("dataset") is training_dataset for c in construction_calls
        )

    def test_falls_back_to_active_dataset_without_datamodule_training_dataset(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fall back to the active split's dataset when no datamodule is available."""
        callback = MediaLoggingCallback()
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        callback.make_plots(trainer, pl_module, dataset, 1)

        construction_calls = stubs["media_publisher_class"].call_args_list
        assert any(c.kwargs.get("dataset") is dataset for c in construction_calls)

    def test_selects_start_date_using_batch_size_and_cached_batch_idx(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Index into dataset.dates using batch_size * cached_batch_idx_, not just 0."""
        callback = MediaLoggingCallback()
        _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 2
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args
        dataset.dates = [
            np.datetime64(f"2020-01-{day:02d}T12:00:00") for day in range(1, 10)
        ]

        callback.make_plots(trainer, pl_module, dataset, 3)

        dataset.get_forecast_steps.assert_called_once_with(dataset.dates[6])

    def test_caches_land_mask_per_path(
        self,
        tmp_path: Path,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Build a LandMask once per mask directory and reuse it on repeat calls."""
        callback = MediaLoggingCallback()
        _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args
        trainer.datamodule = MagicMock(mask_directory=tmp_path)

        callback.make_plots(trainer, pl_module, dataset, 1)
        land_mask_path = tmp_path / "land_mask.npy"
        first_land_mask = callback._land_mask_cache[land_mask_path]
        callback.make_plots(trainer, pl_module, dataset, 1)

        assert callback._land_mask_cache[land_mask_path] is first_land_mask
        assert len(callback._land_mask_cache) == 1

    def test_passes_land_mask_to_media_publisher(
        self,
        tmp_path: Path,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass the cached LandMask to MediaPublisher at construction."""
        callback = MediaLoggingCallback()
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args
        trainer.datamodule = MagicMock(mask_directory=tmp_path)

        callback.make_plots(trainer, pl_module, dataset, 1)

        land_mask_path = tmp_path / "land_mask.npy"
        expected_land_mask = callback._land_mask_cache[land_mask_path]
        construction_calls = stubs["media_publisher_class"].call_args_list
        assert any(
            c.kwargs.get("land_mask") is expected_land_mask for c in construction_calls
        )

    def test_skips_static_and_video_plots_when_disabled(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Do not call any image logger output methods when both toggles are disabled."""
        callback = MediaLoggingCallback(make_static_plots=False, make_video_plots=False)
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        callback.make_plots(trainer, pl_module, dataset, 1)

        stubs["log_static_outputs"].assert_not_called()
        stubs["log_video_outputs"].assert_not_called()

    def test_makes_input_plots_when_enabled(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Also log static and video input plots when make_input_plots is enabled."""
        callback = MediaLoggingCallback(make_input_plots=True)
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        callback.make_plots(trainer, pl_module, dataset, 1)

        stubs["log_static_inputs"].assert_called_once()
        stubs["log_video_inputs"].assert_called_once()

    def test_only_logs_input_plots_once_per_dataloader_and_date(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Do not repeat static/video input plots for a dataloader/date already logged."""
        callback = MediaLoggingCallback(make_input_plots=True)
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        callback.make_plots(trainer, pl_module, dataset, 1)
        callback.make_plots(trainer, pl_module, dataset, 1)

        stubs["log_static_inputs"].assert_called_once()
        stubs["log_video_inputs"].assert_called_once()

    def test_logs_input_plots_again_for_a_different_dataloader(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Log input plots again when the same date recurs on a different dataloader."""
        callback = MediaLoggingCallback(make_input_plots=True)
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        callback.cached_dataloader_idx_ = 0
        callback.make_plots(trainer, pl_module, dataset, 1)
        callback.cached_dataloader_idx_ = 1
        callback.make_plots(trainer, pl_module, dataset, 1)

        assert stubs["log_static_inputs"].call_count == 2
        assert stubs["log_video_inputs"].call_count == 2

    def test_compare_truth_climatology_only_true_once_per_date(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass compare_truth_climatology=True only on the first call for a date."""
        callback = MediaLoggingCallback(make_input_plots=True)
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        callback.make_plots(trainer, pl_module, dataset, 1)
        callback.make_plots(trainer, pl_module, dataset, 1)

        static_calls = stubs["log_static_outputs"].call_args_list
        video_calls = stubs["log_video_outputs"].call_args_list
        assert [c.kwargs["compare_truth_climatology"] for c in static_calls] == [
            True,
            False,
        ]
        assert [c.kwargs["compare_truth_climatology"] for c in video_calls] == [
            True,
            False,
        ]

    def test_compare_truth_climatology_false_when_input_plots_disabled(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Never pass compare_truth_climatology=True when make_input_plots is off."""
        callback = MediaLoggingCallback(make_input_plots=False)
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        callback.make_plots(trainer, pl_module, dataset, 1)

        static_call = stubs["log_static_outputs"].call_args
        video_call = stubs["log_video_outputs"].call_args
        assert static_call.kwargs["compare_truth_climatology"] is False
        assert video_call.kwargs["compare_truth_climatology"] is False

    def test_filters_loggers_by_image_and_video_support(
        self,
        make_plots_args: tuple[MagicMock, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only pass loggers with log_image/log_video to the respective plot calls."""
        callback = MediaLoggingCallback()
        stubs = _stub_media_publisher(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = make_plots_args

        class ImageLogger:
            def log_image(self, *args: object, **kwargs: object) -> None: ...

        class VideoLogger:
            def log_video(self, *args: object, **kwargs: object) -> None: ...

        class PlainLogger: ...

        image_logger = ImageLogger()
        video_logger = VideoLogger()
        trainer.loggers = [image_logger, video_logger, PlainLogger()]

        callback.make_plots(trainer, pl_module, dataset, 1)

        assert stubs["log_static_outputs"].call_args[0][1] == [image_logger]
        assert stubs["log_video_outputs"].call_args[0][1] == [video_logger]


class TestOnTestBatchEnd:
    """Tests for on_test_batch_end."""

    def test_caches_and_plots_when_per_batch_frequency_matches(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache and plot when the batch index matches the configured frequency."""
        callback = MediaLoggingCallback(frequency={"batch": 2})
        cache_batch = MagicMock()
        dataset = MagicMock(spec=CombinedDataset)
        load_dataset = MagicMock(return_value=(dataset, 2))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        monkeypatch.setattr(callback, "make_plots", make_plots)
        mock_trainer.is_last_batch = False
        mock_trainer.num_test_batches = [100]
        outputs = MagicMock()

        callback.on_test_batch_end(mock_trainer, mock_module, outputs, {}, 4, 0)

        cache_batch.assert_called_once_with(4, 0, outputs)
        load_dataset.assert_called_once_with(mock_trainer.test_dataloaders)
        make_plots.assert_called_once_with(mock_trainer, mock_module, dataset, 2)

    def test_caches_without_plotting_on_last_batch_only(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache the final batch of the epoch but defer plotting to epoch end."""
        callback = MediaLoggingCallback()
        cache_batch = MagicMock()
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        monkeypatch.setattr(callback, "make_plots", make_plots)
        mock_trainer.is_last_batch = True
        mock_trainer.num_test_batches = [100]
        outputs = MagicMock()

        callback.on_test_batch_end(mock_trainer, mock_module, outputs, {}, 4, 0)

        cache_batch.assert_called_once_with(4, 0, outputs)
        make_plots.assert_not_called()

    def test_does_nothing_when_batch_not_targeted(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip caching entirely when the batch matches no trigger."""
        callback = MediaLoggingCallback()
        cache_batch = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        mock_trainer.is_last_batch = False
        mock_trainer.num_test_batches = [100]

        callback.on_test_batch_end(mock_trainer, mock_module, MagicMock(), {}, 4, 0)

        cache_batch.assert_not_called()

    def test_warns_when_dataset_cannot_be_loaded(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn and skip plotting when the dataset cannot be loaded."""
        callback = MediaLoggingCallback(frequency={"batch": 1})
        monkeypatch.setattr(callback, "cache_batch", MagicMock())
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        mock_trainer.is_last_batch = False
        mock_trainer.num_test_batches = [100]

        with caplog.at_level(logging.WARNING):
            callback.on_test_batch_end(mock_trainer, mock_module, MagicMock(), {}, 0, 0)

        assert "Could not load dataset" in caplog.text
        make_plots.assert_not_called()


class TestOnTestEpochEnd:
    """Tests for on_test_epoch_end."""

    def test_skips_when_frequency_epoch_negative(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Never load the dataset when epoch-based plotting is disabled."""
        callback = MediaLoggingCallback()
        load_dataset = MagicMock()
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        mock_trainer.current_epoch = 5

        callback.on_test_epoch_end(mock_trainer, mock_module)

        load_dataset.assert_not_called()

    def test_skips_when_epoch_not_at_frequency(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip plotting on epochs that do not match the configured frequency."""
        callback = MediaLoggingCallback(frequency={"epoch": 2})
        load_dataset = MagicMock()
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        mock_trainer.current_epoch = 3

        callback.on_test_epoch_end(mock_trainer, mock_module)

        load_dataset.assert_not_called()

    def test_plots_when_epoch_at_frequency(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plot on epochs that match the configured frequency."""
        callback = MediaLoggingCallback(frequency={"epoch": 2})
        dataset = MagicMock(spec=CombinedDataset)
        monkeypatch.setattr(
            callback, "load_dataset", MagicMock(return_value=(dataset, 2))
        )
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        mock_trainer.current_epoch = 4

        callback.on_test_epoch_end(mock_trainer, mock_module)

        make_plots.assert_called_once_with(mock_trainer, mock_module, dataset, 2)

    def test_warns_when_dataset_cannot_be_loaded(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when the dataset cannot be loaded at epoch end."""
        callback = MediaLoggingCallback(frequency={"epoch": 1})
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        mock_trainer.current_epoch = 0

        with caplog.at_level(logging.WARNING):
            callback.on_test_epoch_end(mock_trainer, mock_module)

        assert "Could not load dataset" in caplog.text


class TestOnValidationBatchEnd:
    """Tests for on_validation_batch_end."""

    def test_skips_during_sanity_checking(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ignore the initial sanity-checking run entirely."""
        callback = MediaLoggingCallback(frequency={"batch": 1})
        cache_batch = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        mock_trainer.sanity_checking = True

        callback.on_validation_batch_end(
            mock_trainer, mock_module, MagicMock(), {}, 0, 0
        )

        cache_batch.assert_not_called()

    def test_caches_and_plots_when_per_batch_frequency_matches(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache and plot when the batch index matches the configured frequency."""
        callback = MediaLoggingCallback(frequency={"batch": 2})
        cache_batch = MagicMock()
        dataset = MagicMock(spec=CombinedDataset)
        load_dataset = MagicMock(return_value=(dataset, 3))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        monkeypatch.setattr(callback, "make_plots", make_plots)
        mock_trainer.sanity_checking = False
        mock_trainer.fit_loop = MagicMock()
        mock_trainer.fit_loop.epoch_loop.val_loop.batch_progress.is_last_batch = False
        mock_trainer.num_val_batches = [50]
        outputs = MagicMock()

        callback.on_validation_batch_end(mock_trainer, mock_module, outputs, {}, 4, 0)

        cache_batch.assert_called_once_with(4, 0, outputs)
        load_dataset.assert_called_once_with(mock_trainer.val_dataloaders)
        make_plots.assert_called_once_with(mock_trainer, mock_module, dataset, 3)

    def test_warns_when_dataset_cannot_be_loaded(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn and skip plotting when the dataset cannot be loaded."""
        callback = MediaLoggingCallback(frequency={"batch": 1})
        monkeypatch.setattr(callback, "cache_batch", MagicMock())
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        mock_trainer.sanity_checking = False
        mock_trainer.fit_loop = MagicMock()
        mock_trainer.fit_loop.epoch_loop.val_loop.batch_progress.is_last_batch = False
        mock_trainer.num_val_batches = [50]

        with caplog.at_level(logging.WARNING):
            callback.on_validation_batch_end(
                mock_trainer, mock_module, MagicMock(), {}, 0, 0
            )

        assert "Could not load dataset" in caplog.text
        make_plots.assert_not_called()


class TestOnValidationEpochEnd:
    """Tests for on_validation_epoch_end."""

    def test_skips_when_epoch_not_at_frequency(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip plotting on epochs that do not match the configured frequency."""
        callback = MediaLoggingCallback(frequency={"epoch": 2})
        load_dataset = MagicMock()
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        mock_trainer.current_epoch = 3

        callback.on_validation_epoch_end(mock_trainer, mock_module)

        load_dataset.assert_not_called()

    def test_plots_when_epoch_at_frequency(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plot on epochs that match the configured frequency."""
        callback = MediaLoggingCallback(frequency={"epoch": 2})
        dataset = MagicMock(spec=CombinedDataset)
        monkeypatch.setattr(
            callback, "load_dataset", MagicMock(return_value=(dataset, 2))
        )
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        mock_trainer.current_epoch = 4

        callback.on_validation_epoch_end(mock_trainer, mock_module)

        make_plots.assert_called_once_with(mock_trainer, mock_module, dataset, 2)

    def test_warns_when_dataset_cannot_be_loaded(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when the dataset cannot be loaded at epoch end."""
        callback = MediaLoggingCallback(frequency={"epoch": 1})
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        mock_trainer.current_epoch = 0

        with caplog.at_level(logging.WARNING):
            callback.on_validation_epoch_end(mock_trainer, mock_module)

        assert "Could not load dataset" in caplog.text
