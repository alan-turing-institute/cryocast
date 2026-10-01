import logging
from unittest.mock import MagicMock

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from torchmetrics import MeanAbsoluteError, MetricCollection

from icenet_mp.callbacks.metric_summary_callback import MetricSummaryCallback
from icenet_mp.metrics import (
    DistanceAveragedIceEdgeErrorPerForecastDay,
    FractionalSkillScorePerForecastDay,
    IceNetAccuracyPerForecastDay,
    IntegratedIceEdgeErrorPerForecastDay,
    MAEPerForecastDay,
    RMSEPerForecastDay,
    SeaIceExtentErrorPerForecastDay,
    SpatialMeanGroundTruthPerForecastDay,
    SpatialMeanPredictionPerForecastDay,
)
from icenet_mp.types import ModelStepOutput


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Override the default mock_trainer with sanity checking off and one plain logger."""
    trainer = MagicMock(spec=Trainer)
    trainer.sanity_checking = False
    mock_logger = MagicMock()
    trainer.loggers = [mock_logger]
    return trainer


class MockWandbRun:
    """A stand-in for a live W&B run that records .log() calls."""

    def __init__(self) -> None:
        """Initialise the mock run with a MagicMock log method."""
        self.log = MagicMock()


@pytest.fixture
def wandb_run(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MockWandbRun]:
    """Mock the wandb module and get_wandb_run to return a working MockWandbRun.

    Used by the tests that need W&B to look "present" with a run whose .log() calls can
    be asserted on; both duplicated this wiring verbatim before this fixture existed.
    """
    mock_wandb = MagicMock()
    mock_wandb.Run = MockWandbRun
    mock_run = MockWandbRun()
    mock_get_wandb_run = MagicMock(return_value=mock_run)
    monkeypatch.setattr("icenet_mp.callbacks.metric_summary_callback.wandb", mock_wandb)
    monkeypatch.setattr(
        "icenet_mp.callbacks.metric_summary_callback.get_wandb_run",
        mock_get_wandb_run,
    )
    return mock_wandb, mock_run


class TestOnTestEnd:
    """Tests for on_test_end method."""

    def test_on_test_end_with_metric_collection(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end with a valid MetricCollection."""
        callback = MetricSummaryCallback()
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        mock_module.test_metrics = metric_collection

        # Create sample predictions and targets
        preds = torch.randn(10)
        targets = torch.randn(10)

        for pred, target in zip(preds, targets, strict=False):
            metric_collection.update(pred.unsqueeze(0), target.unsqueeze(0))

        callback.on_test_epoch_end(mock_trainer, mock_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called()

    def test_on_test_end_with_invalid_test_metrics(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end when test_metrics is not a MetricCollection."""
        callback = MetricSummaryCallback()
        mock_module.test_metrics = "invalid"

        callback.on_test_epoch_end(mock_trainer, mock_module)

        # Should not raise an error, just log a warning
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_not_called()

    def test_on_test_end_with_wandb_logger_vector_metric(
        self,
        mock_module: MagicMock,
        wandb_run: tuple[MagicMock, MockWandbRun],
    ) -> None:
        """Test on_test_end with WandbLogger and a metric returning a vector."""
        callback = MetricSummaryCallback()
        mock_wandb, mock_run = wandb_run

        # Create a trainer with WandbLogger
        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False
        wandb_logger = MagicMock(spec=WandbLogger)
        trainer.loggers = [wandb_logger]

        # Create a metric that returns multiple values (daily metric)
        metric_collection = MetricCollection({"mae_daily": MAEPerForecastDay()})
        mock_module.test_metrics = metric_collection

        # Create sample 5D data: (batch=1, time=3, channels=1, height=2, width=2)
        preds = torch.randn(1, 3, 1, 2, 2)
        targets = torch.randn(1, 3, 1, 2, 2)
        metric_collection.update(preds, targets)

        mock_plot = MagicMock()
        mock_wandb.plot.line_series.return_value = mock_plot

        callback.teardown(trainer, mock_module, stage="test")

        # Assert that wandb.plot.line_series was called with the daily values
        mock_wandb.plot.line_series.assert_called_once()
        line_series_kwargs = mock_wandb.plot.line_series.call_args[1]
        assert line_series_kwargs["keys"] == ["test"]
        assert line_series_kwargs["title"] == "mae_daily_per_forecast_day"
        assert line_series_kwargs["xname"] == "day"

        # Assert that wandb.log was called with the plot under the correct key
        mock_run.log.assert_called_once()
        log_call_args = mock_run.log.call_args[0][0]
        assert "mae_daily_per_forecast_day" in log_call_args

    def test_on_test_end_without_wandb_logger_vector_metric(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end with non-WandbLogger and a metric returning a vector."""
        callback = MetricSummaryCallback()

        # Create a metric that returns multiple values (daily metric)
        metric_collection = MetricCollection({"mae_daily": MAEPerForecastDay()})
        mock_module.test_metrics = metric_collection

        # Create sample 5D data: (batch=1, time=3, channels=1, height=2, width=2)
        preds = torch.randn(1, 3, 1, 2, 2)
        targets = torch.randn(1, 3, 1, 2, 2)
        metric_collection.update(preds, targets)

        callback.on_test_epoch_end(mock_trainer, mock_module)

        # Assert that the mean value was logged without wandb plotting
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()
        metrics_call_args = mock_logger.log_metrics.call_args[0][0]
        assert "test_mae_daily_mean" in metrics_call_args

    def test_on_test_end_with_wandb_logger_grouped_vector_metrics(
        self,
        mock_module: MagicMock,
        mock_trainer: MagicMock,
        wandb_run: tuple[MagicMock, MockWandbRun],
    ) -> None:
        """FSS at different neighbourhood sizes combines onto a single plot."""
        callback = MetricSummaryCallback()
        mock_wandb, _ = wandb_run

        # FSS at two neighbourhood sizes, plus an unrelated daily metric that
        # should stay on its own plot
        metric_collection = MetricCollection(
            {
                "fss_neighbourhood_size_1": FractionalSkillScorePerForecastDay(
                    neighbourhood_size=1
                ),
                "fss_neighbourhood_size_5": FractionalSkillScorePerForecastDay(
                    neighbourhood_size=5
                ),
                "mae_daily": MAEPerForecastDay(),
            }
        )
        mock_module.test_metrics = metric_collection

        # Create sample 5D data: (batch=1, time=3, channels=1, height=6, width=6)
        preds = torch.rand(1, 3, 1, 6, 6)
        targets = torch.rand(1, 3, 1, 6, 6)
        metric_collection.update(preds, targets)

        mock_plot = MagicMock()
        mock_wandb.plot.line_series.return_value = mock_plot

        callback.teardown(mock_trainer, mock_module, stage="test")

        # One combined per-forecast-day plot for the fss_* group, one for the
        # unrelated metric, and one FSS-vs-neighbourhood-size plot
        assert mock_wandb.plot.line_series.call_count == 3
        titles = {
            call.kwargs["title"] for call in mock_wandb.plot.line_series.call_args_list
        }
        assert titles == {
            "fss_per_forecast_day",
            "mae_daily_per_forecast_day",
            "fss_vs_neighbourhood_size",
        }

        fss_call = next(
            call
            for call in mock_wandb.plot.line_series.call_args_list
            if call.kwargs["title"] == "fss_per_forecast_day"
        )
        assert fss_call.kwargs["keys"] == [
            "fss_neighbourhood_size_1",
            "fss_neighbourhood_size_5",
        ]

        fss_vs_size_call = next(
            call
            for call in mock_wandb.plot.line_series.call_args_list
            if call.kwargs["title"] == "fss_vs_neighbourhood_size"
        )
        assert fss_vs_size_call.kwargs["xs"] == [1, 5]
        assert fss_vs_size_call.kwargs["keys"] == ["test"]
        assert len(fss_vs_size_call.kwargs["ys"]) == 1
        assert len(fss_vs_size_call.kwargs["ys"][0]) == 2

    def test_on_test_end_with_wandb_logger_groups_spatial_mean_trace(
        self,
        mock_module: MagicMock,
        mock_trainer: MagicMock,
        wandb_run: tuple[MagicMock, MockWandbRun],
    ) -> None:
        """Ground-truth and prediction spatial-mean traces combine onto one plot."""
        callback = MetricSummaryCallback()
        mock_wandb, _ = wandb_run

        metric_collection = MetricCollection(
            {
                "spatial_mean_ground_truth": SpatialMeanGroundTruthPerForecastDay(),
                "spatial_mean_prediction": SpatialMeanPredictionPerForecastDay(),
            }
        )
        mock_module.test_metrics = metric_collection

        # Sample 5D data: (batch=1, time=3, channels=1, height=2, width=2)
        preds = torch.rand(1, 3, 1, 2, 2)
        targets = torch.rand(1, 3, 1, 2, 2)
        metric_collection.update(preds, targets)

        mock_plot = MagicMock()
        mock_wandb.plot.line_series.return_value = mock_plot

        callback.teardown(mock_trainer, mock_module, stage="test")

        mock_wandb.plot.line_series.assert_called_once()
        line_series_kwargs = mock_wandb.plot.line_series.call_args[1]
        assert line_series_kwargs["title"] == "spatial_mean_per_forecast_day"
        assert set(line_series_kwargs["keys"]) == {
            "spatial_mean_ground_truth",
            "spatial_mean_prediction",
        }


class TestLogPerEpochMetrics:
    """Tests for log_per_epoch_metrics."""

    def test_skips_during_sanity_checking(self, mock_trainer: MagicMock) -> None:
        """Do not log anything while Lightning's sanity check is running."""
        callback = MetricSummaryCallback()
        mock_trainer.sanity_checking = True
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))

        callback.log_per_epoch_metrics(mock_trainer, metric_collection, stage="test")

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_not_called()

    def test_skips_metrics_that_were_never_updated(
        self, mock_trainer: MagicMock
    ) -> None:
        """Skip metrics in the collection whose update() was never called."""
        callback = MetricSummaryCallback()
        metric_collection = MetricCollection(
            {"mae": MeanAbsoluteError(), "unused": MeanAbsoluteError()}
        )
        metric_collection["mae"].update(torch.zeros(1), torch.ones(1))

        callback.log_per_epoch_metrics(mock_trainer, metric_collection, stage="test")

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()
        logged_metrics = mock_logger.log_metrics.call_args[0][0]
        assert "test_mae_mean" in logged_metrics
        assert "test_unused_mean" not in logged_metrics


class TestLogPerRunMetrics:
    """Tests for log_per_run_metrics."""

    def test_skips_during_sanity_checking(
        self,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Return before evaluating anything while Lightning's sanity check is running."""
        callback = MetricSummaryCallback()
        mock_trainer.sanity_checking = True

        with caplog.at_level(logging.WARNING):
            callback.log_per_run_metrics(mock_trainer, {})

        assert caplog.text == ""

    def test_warns_and_skips_without_wandb_logger(
        self,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn and skip logging when no WandbLogger/run is available."""
        callback = MetricSummaryCallback()
        with caplog.at_level(logging.WARNING):
            callback.log_per_run_metrics(mock_trainer, {})

        assert "W&B is not being used as a logger" in caplog.text

    def test_skips_metrics_that_were_never_updated(
        self,
        wandb_run: tuple[MagicMock, MockWandbRun],
    ) -> None:
        """Skip metrics whose update() was never called when building the per-day plot."""
        callback = MetricSummaryCallback()
        mock_wandb, _ = wandb_run

        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False

        metric_collection = MetricCollection(
            {"mae_daily": MAEPerForecastDay(), "unused_daily": MAEPerForecastDay()}
        )
        preds = torch.randn(1, 3, 1, 2, 2)
        targets = torch.randn(1, 3, 1, 2, 2)
        metric_collection["mae_daily"].update(preds, targets)

        callback.log_per_run_metrics(trainer, {"test": metric_collection})

        mock_wandb.plot.line_series.assert_called_once()
        line_series_kwargs = mock_wandb.plot.line_series.call_args[1]
        assert line_series_kwargs["title"] == "mae_daily_per_forecast_day"

    def test_strips_group_prefix_from_spatial_mean_keys_across_stages(
        self,
        wandb_run: tuple[MagicMock, MockWandbRun],
    ) -> None:
        """Multi-stage spatial-mean keys drop the redundant group prefix.

        e.g. "train_spatial_mean_ground_truth" becomes "train_ground_truth", and
        "validation_spatial_mean_prediction" becomes "validation_prediction".
        """
        callback = MetricSummaryCallback()
        mock_wandb, _ = wandb_run

        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False

        preds = torch.rand(1, 3, 1, 2, 2)
        targets = torch.rand(1, 3, 1, 2, 2)
        metrics_by_stage = {}
        for stage in ("train", "validation"):
            metric_collection = MetricCollection(
                {
                    "spatial_mean_ground_truth": SpatialMeanGroundTruthPerForecastDay(),
                    "spatial_mean_prediction": SpatialMeanPredictionPerForecastDay(),
                }
            )
            metric_collection.update(preds, targets)
            metrics_by_stage[stage] = metric_collection

        callback.log_per_run_metrics(trainer, metrics_by_stage)

        mock_wandb.plot.line_series.assert_called_once()
        line_series_kwargs = mock_wandb.plot.line_series.call_args[1]
        assert line_series_kwargs["title"] == "spatial_mean_per_forecast_day"
        assert set(line_series_kwargs["keys"]) == {
            "train_ground_truth",
            "train_prediction",
            "validation_ground_truth",
            "validation_prediction",
        }

    def test_removes_fss_prefix_across_stages(
        self,
        wandb_run: tuple[MagicMock, MockWandbRun],
    ) -> None:
        """Multi-stage FSS keys remove the "fss" prefix."""
        callback = MetricSummaryCallback()
        mock_wandb, _ = wandb_run

        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False

        preds = torch.rand(1, 3, 1, 6, 6)
        targets = torch.rand(1, 3, 1, 6, 6)
        metrics_by_stage = {}
        for stage in ("train", "validation"):
            metric_collection = MetricCollection(
                {
                    "fss_neighbourhood_size_1": FractionalSkillScorePerForecastDay(
                        neighbourhood_size=1
                    ),
                    "fss_neighbourhood_size_5": FractionalSkillScorePerForecastDay(
                        neighbourhood_size=5
                    ),
                }
            )
            metric_collection.update(preds, targets)
            metrics_by_stage[stage] = metric_collection

        callback.log_per_run_metrics(trainer, metrics_by_stage)

        fss_call = next(
            call
            for call in mock_wandb.plot.line_series.call_args_list
            if call.kwargs["title"] == "fss_per_forecast_day"
        )
        assert set(fss_call.kwargs["keys"]) == {
            "train_neighbourhood_size_1",
            "train_neighbourhood_size_5",
            "validation_neighbourhood_size_1",
            "validation_neighbourhood_size_5",
        }


class TestEpochStartResets:
    """Tests for the on_*_epoch_start metric-reset hooks."""

    def test_on_test_epoch_start_resets_test_metrics(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Reset test_metrics at the start of a test epoch."""
        callback = MetricSummaryCallback()
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        mock_module.test_metrics = metric_collection

        callback.on_test_epoch_start(mock_trainer, mock_module)

        assert metric_collection["mae"].update_called is False

    def test_on_train_epoch_start_resets_train_metrics(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Reset train_metrics at the start of a training epoch."""
        callback = MetricSummaryCallback()
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        mock_module.train_metrics = metric_collection

        callback.on_train_epoch_start(mock_trainer, mock_module)

        assert metric_collection["mae"].update_called is False

    def test_on_validation_epoch_start_resets_validation_metrics(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Reset validation_metrics at the start of a validation epoch."""
        callback = MetricSummaryCallback()
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        mock_module.validation_metrics = metric_collection

        callback.on_validation_epoch_start(mock_trainer, mock_module)

        assert metric_collection["mae"].update_called is False


class TestOnTrainEpochEnd:
    """Tests for on_train_epoch_end."""

    def test_logs_when_train_metrics_present(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Log per-epoch metrics when train_metrics is a MetricCollection."""
        callback = MetricSummaryCallback()
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        mock_module.train_metrics = metric_collection

        callback.on_train_epoch_end(mock_trainer, mock_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()

    def test_warns_when_train_metrics_missing(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when train_metrics is not a MetricCollection."""
        callback = MetricSummaryCallback()
        mock_module.train_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.on_train_epoch_end(mock_trainer, mock_module)

        assert "Could not load train metrics!" in caplog.text


class TestOnValidationEpochEnd:
    """Tests for on_validation_epoch_end."""

    def test_logs_when_validation_metrics_present(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Log per-epoch metrics when validation_metrics is a MetricCollection."""
        callback = MetricSummaryCallback()
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        mock_module.validation_metrics = metric_collection

        callback.on_validation_epoch_end(mock_trainer, mock_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()

    def test_warns_when_validation_metrics_missing(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when validation_metrics is not a MetricCollection."""
        callback = MetricSummaryCallback()
        mock_module.validation_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.on_validation_epoch_end(mock_trainer, mock_module)

        assert "Could not load validation metrics!" in caplog.text


class TestTeardown:
    """Tests for teardown's per-stage metric collection."""

    def test_fitting_stage_collects_train_and_validation_metrics(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gather both train and validation metrics for a fit run."""
        callback = MetricSummaryCallback()
        mock_module.train_metrics = MetricCollection({"mae": MeanAbsoluteError()})
        mock_module.validation_metrics = MetricCollection({"mae": MeanAbsoluteError()})
        mock_log_per_run_metrics = MagicMock()
        monkeypatch.setattr(callback, "log_per_run_metrics", mock_log_per_run_metrics)

        callback.teardown(mock_trainer, mock_module, stage=TrainerFn.FITTING.value)

        mock_log_per_run_metrics.assert_called_once()
        metrics = mock_log_per_run_metrics.call_args[0][1]
        assert set(metrics) == {"train", "validation"}

    def test_fitting_stage_warns_when_metrics_missing(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn for each stage whose metrics collection is missing during a fit run."""
        callback = MetricSummaryCallback()
        mock_module.train_metrics = "invalid"
        mock_module.validation_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.teardown(mock_trainer, mock_module, stage=TrainerFn.FITTING.value)

        assert "Could not load train metrics!" in caplog.text
        assert "Could not load validation metrics!" in caplog.text

    def test_testing_stage_warns_when_test_metrics_missing(
        self,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when test_metrics is missing during a test run."""
        callback = MetricSummaryCallback()
        mock_module.test_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.teardown(mock_trainer, mock_module, stage=TrainerFn.TESTING.value)

        assert "Could not load test metrics!" in caplog.text


class TestMetricCalculations:
    """Tests for per-forecast-day metric calculation correctness."""

    @staticmethod
    def to_5d(values_2d: list[list[float]]) -> torch.Tensor:
        """Reshape a 4-cell x 3-day grid into (batch=1, time=3, channels=1, height=2, width=2)."""
        return (
            torch.tensor(values_2d)
            .view(2, 2, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .unsqueeze(2)
        )

    def test_accumulates_multiple_batches(self) -> None:
        """Accumulate daily errors across batches with matching lead times."""
        metric = MAEPerForecastDay()
        metric.update(torch.zeros(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1))
        metric.update(torch.zeros(1, 1, 1, 1, 1), torch.full((1, 1, 1, 1, 1), 3.0))

        assert torch.allclose(metric.compute(), torch.tensor([2.0]))

    def test_calculates_mean_mae_daily_correctly(self) -> None:
        """Test that MAE daily is calculated correctly."""
        preds = self.to_5d(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets = self.to_5d(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        computed_mae = MAEPerForecastDay()
        computed_mae.update(preds, targets)
        daily_result = computed_mae.compute()
        # Expected MAE per day:
        # Day 1: (|1.0-1.5| + |1.0-0.5| + |2.0-2.0| + |2.0-2.5|) / 4 = 0.375
        # Day 2: (|2.0-2.5| + |3.0-3.5| + |3.0-4.0| + |4.0-3.0|) / 4 = 0.75
        # Day 3: (|4.0-4.0| + |4.0-4.0| + |5.0-5.0| + |6.0-6.0|) / 4 = 0.0
        expected_mae = torch.tensor([0.375, 0.75, 0.0])

        assert torch.allclose(daily_result, expected_mae, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(0.375, abs=1e-5)

    def test_calculates_mean_rmse_daily_correctly(self) -> None:
        """Test that RMSE daily is calculated correctly."""
        preds = self.to_5d(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets = self.to_5d(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        computed_rmse = RMSEPerForecastDay()
        computed_rmse.update(preds, targets)
        daily_result = computed_rmse.compute()

        # Expected RMSE per day:
        # Day 1: sqrt(mean([0.25, 0.25, 0.0, 0.25])) = sqrt(0.1875) = 0.4330127
        # Day 2: sqrt(mean([0.25, 0.25, 1.0, 1.0])) = sqrt(0.625) = 0.7905694
        # Day 3: sqrt(mean([0.0, 0.0, 0.0, 0.0])) = 0.0
        expected_rmse = torch.tensor([0.4330127, 0.7905694, 0.0])

        assert torch.allclose(daily_result, expected_rmse, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(0.40786, abs=1e-5)

    def test_calculates_mean_sieerror_daily_correctly(self) -> None:
        """Test that SIEError daily is calculated correctly."""
        preds = self.to_5d(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets = self.to_5d(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        computed_sie = SeaIceExtentErrorPerForecastDay(pixel_size=1)
        computed_sie.update(preds, targets)
        daily_result = computed_sie.compute()

        # Expected SIEError per day:
        # Day 1: sie error = (0-1 + 0-1 + 1-1 + 0-0) * 1^2 = -2.0
        # Day 2: sie error = (0-1 + 1-0 + 1-1 + 0-0) * 1^2 = 0.0
        # Day 3: sie error = (1-0 + 1-0 + 1-1 + 0-1) * 1^2 = 1.0
        expected_sie = torch.tensor([-2.0, 0.0, 1.0])  # pixel_size=1 -> no scaling

        assert torch.allclose(daily_result, expected_sie, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(-0.33333, abs=1e-5)

    def test_calculates_mean_sieerror_daily_pixel_size(self) -> None:
        """Test that SIEError daily is calculated correctly."""
        preds = self.to_5d(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets = self.to_5d(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        computed_sie = SeaIceExtentErrorPerForecastDay()
        computed_sie.update(preds, targets)
        daily_result = computed_sie.compute()

        # Day 1: sie error = (0-1 + 0-1 + 1-1 + 0-0) * 1^2 = -2.0
        # Day 2: sie error = (0-1 + 1-0 + 1-1 + 0-0) * 1^2 = 0.0
        # Day 3: sie error = (1-0 + 1-0 + 1-1 + 0-1) * 1^2 = 1.0
        # Scale factor is default pixel_size^2 = 625
        expected_sie = torch.tensor([-1250.0, 0.0, 625.0])
        assert torch.allclose(daily_result, expected_sie, atol=1e-5)
        assert daily_result.mean().item() == pytest.approx(-208.33333, abs=1e-5)

    def test_calculates_mean_iiee_daily_correctly(self) -> None:
        """Test that IIEE daily is calculated correctly."""
        preds_2d = torch.tensor(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets_2d = torch.tensor(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_iiee = IntegratedIceEdgeErrorPerForecastDay(pixel_size=1)
        computed_iiee.update(preds, targets)
        daily_result = computed_iiee.compute()

        # Expected IIEE per day (unsigned disagreement, unlike SIEError which cancels):
        # Day 1: |0-1| + |0-1| + |1-1| + |0-0| = 2.0
        # Day 2: |0-1| + |1-0| + |1-1| + |0-0| = 2.0
        # Day 3: |1-0| + |1-0| + |1-1| + |0-1| = 3.0
        expected_iiee = torch.tensor([2.0, 2.0, 3.0])  # pixel_size=1 -> no scaling

        assert torch.allclose(daily_result, expected_iiee, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(2.33333, abs=1e-5)

    def test_calculates_mean_iiee_daily_pixel_size(self) -> None:
        """Test that IIEE daily is calculated correctly with pixel-size scaling."""
        preds_2d = torch.tensor(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets_2d = torch.tensor(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_iiee = IntegratedIceEdgeErrorPerForecastDay()
        computed_iiee.update(preds, targets)
        daily_result = computed_iiee.compute()

        # Expected IIEE per day (before pixel-size scaling): [2.0, 2.0, 3.0]
        expected_iiee = torch.tensor(
            [1250.0, 1250.0, 1875.0]
        )  # default pixel_size=25 -> scaled by 25^2

        assert torch.allclose(daily_result, expected_iiee, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(1458.33333, abs=1e-3)

    def test_calculates_mean_accuracy_daily_correctly(self) -> None:
        """Test that IceNetAccuracy daily is calculated correctly."""
        preds_2d = torch.tensor(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets_2d = torch.tensor(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_sie = IceNetAccuracyPerForecastDay()
        computed_sie.update(preds, targets)
        daily_result = computed_sie.compute()

        # Expected IceNetAccuracy per day:
        # Day 1: accuracy = (0 + 0 + 1 + 1)  = 2.0
        # Day 2: accuracy = (0 + 0 + 1 + 1) = 2.0
        # Day 3: accuracy = (0 + 0 + 1 + 0) = 1.0
        expected_sie = torch.tensor([50.0, 50.0, 25.0])  # pixel_size=1 -> no scaling

        assert torch.allclose(daily_result, expected_sie, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(41.66667, abs=1e-5)

    def test_calculates_mean_weighted_accuracy_daily_correctly(self) -> None:
        """Test that IceNetAccuracy daily is calculated correctly."""
        preds_2d = torch.tensor(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets_2d = torch.tensor(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        sample_weight_2d = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(1).repeat(1, 3)
        sample_weight = (
            sample_weight_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        )

        computed_sie = IceNetAccuracyPerForecastDay()
        computed_sie.update(preds, targets, sample_weight=sample_weight)
        daily_result = computed_sie.compute()

        # Expected IceNetAccuracy per day:
        # Day 1: accuracy = (0 + 0 + 1 + 1) * weights = 0 * 0 + 0 * 1 + 0 * 0 + 1 * 1 = 1.0
        # Day 2: accuracy = (0 + 0 + 1 + 1) * weights = 0 * 0 + 0 * 1 + 0 * 0 + 1 * 1 = 1.0
        # Day 3: accuracy = (0 + 0 + 1 + 0) * weights = 0 * 0 + 0 * 1 + 0 * 0 + 0 * 1 = 0.0
        expected_sie = torch.tensor([50.0, 50.0, 0.0])  # pixel_size=1 -> no scaling

        assert torch.allclose(daily_result, expected_sie, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(33.33333, abs=1e-5)

    def test_calculates_diiee_daily_correctly(self) -> None:
        """Test that DIIEE daily is calculated correctly."""
        # Day 1: predicted 2x2 ice block vs a 2x3 target block (extra column of
        # underestimation). Day 2: predicted and target blocks match exactly.
        # Day 3: neither field has any ice (edge length undefined -> NaN).
        day1_pred = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.3, 0.0],
                [0.0, 0.3, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        day1_target = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.3, 0.3],
                [0.0, 0.3, 0.3, 0.3],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        day2 = day1_pred
        day3 = torch.zeros(4, 4)

        preds = torch.stack([day1_pred, day2, day3]).unsqueeze(0).unsqueeze(2)
        targets = torch.stack([day1_target, day2, day3]).unsqueeze(0).unsqueeze(2)

        computed_diiee = DistanceAveragedIceEdgeErrorPerForecastDay(pixel_size=1)
        computed_diiee.update(preds, targets)
        daily_result = computed_diiee.compute()

        # Day 1: mismatch area = 2 (the extra column), edge length = 4 (pred) + 6
        #   (target) = 10 -> diiee = 2 * 2 / 10 = 0.4
        # Day 2: perfect match -> mismatch area = 0 -> diiee = 0.0
        # Day 3: no ice edge in either field -> undefined -> NaN
        assert torch.allclose(daily_result[:2], torch.tensor([0.4, 0.0]), atol=1e-5)
        assert torch.isnan(daily_result[2])

    def test_calculates_diiee_daily_pixel_size(self) -> None:
        """Test that DIIEE daily is calculated correctly with pixel-size scaling."""
        day1_pred = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.3, 0.0],
                [0.0, 0.3, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        day1_target = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.3, 0.3],
                [0.0, 0.3, 0.3, 0.3],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        day2 = day1_pred

        preds = torch.stack([day1_pred, day2]).unsqueeze(0).unsqueeze(2)
        targets = torch.stack([day1_target, day2]).unsqueeze(0).unsqueeze(2)

        computed_diiee = DistanceAveragedIceEdgeErrorPerForecastDay()
        computed_diiee.update(preds, targets)
        daily_result = computed_diiee.compute()

        # DIIEE is linear in pixel_size (area scales as pixel_size^2, edge length as
        # pixel_size), so the pixel_size=1 result of [0.4, 0.0] scales by 25.
        expected_diiee = torch.tensor([10.0, 0.0])  # default pixel_size=25

        assert torch.allclose(daily_result, expected_diiee, atol=1e-5)

    def test_calculates_ground_truth_mean_daily_correctly(self) -> None:
        """Test that the ground-truth spatial-mean trace is calculated correctly."""
        preds = self.to_5d(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets = self.to_5d(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        metric = SpatialMeanGroundTruthPerForecastDay()
        metric.update(preds, targets)
        daily_result = metric.compute()

        # Expected spatial-mean ground truth per day (preds are ignored):
        # Day 1: (1.5+0.5+2.0+2.5)/4 = 1.625
        # Day 2: (2.5+3.5+4.0+3.0)/4 = 3.25
        # Day 3: (4.0+4.0+5.0+6.0)/4 = 4.75
        expected = torch.tensor([1.625, 3.25, 4.75])

        assert torch.allclose(daily_result, expected, atol=1e-5)

    def test_calculates_prediction_mean_daily_correctly(self) -> None:
        """Test that the prediction spatial-mean trace is calculated correctly."""
        preds = self.to_5d(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets = self.to_5d(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        metric = SpatialMeanPredictionPerForecastDay()
        metric.update(preds, targets)
        daily_result = metric.compute()

        # Expected spatial-mean prediction per day (targets are ignored):
        # Day 1: (1.0+1.0+2.0+2.0)/4 = 1.5
        # Day 2: (2.0+3.0+3.0+4.0)/4 = 3.0
        # Day 3: (4.0+4.0+5.0+6.0)/4 = 4.75
        expected = torch.tensor([1.5, 3.0, 4.75])

        assert torch.allclose(daily_result, expected, atol=1e-5)

    def test_ground_truth_mean_daily_excludes_land_cells(self) -> None:
        """Land cells are excluded from the spatial-mean ground-truth trace."""
        preds = self.to_5d(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets = self.to_5d(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )
        # Mask out the second grid cell (the [0.5, 3.5, 4.0] row above).
        land_mask = torch.tensor([[True, False], [True, True]])

        metric = SpatialMeanGroundTruthPerForecastDay(land_mask=land_mask)
        metric.update(preds, targets)
        daily_result = metric.compute()

        # Expected spatial-mean ground truth per day over the 3 unmasked cells only:
        # Day 1: (1.5+2.0+2.5)/3 = 2.0
        # Day 2: (2.5+4.0+3.0)/3 = 3.16667
        # Day 3: (4.0+5.0+6.0)/3 = 5.0
        expected = torch.tensor([2.0, 3.16667, 5.0])

        assert torch.allclose(daily_result, expected, atol=1e-4)

    def test_prediction_mean_daily_excludes_land_cells(self) -> None:
        """Land cells are excluded from the spatial-mean prediction trace."""
        preds = self.to_5d(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets = self.to_5d(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )
        # Mask out the second grid cell (the [1.0, 3.0, 4.0] row above).
        land_mask = torch.tensor([[True, False], [True, True]])

        metric = SpatialMeanPredictionPerForecastDay(land_mask=land_mask)
        metric.update(preds, targets)
        daily_result = metric.compute()

        # Expected spatial-mean prediction per day over the 3 unmasked cells only:
        # Day 1: (1.0+2.0+2.0)/3 = 1.66667
        # Day 2: (2.0+3.0+4.0)/3 = 3.0
        # Day 3: (4.0+5.0+6.0)/3 = 5.0
        expected = torch.tensor([1.66667, 3.0, 5.0])

        assert torch.allclose(daily_result, expected, atol=1e-4)


class TestClimatologyMetrics:
    """Tests for climatology baseline metrics in on_test_batch_end."""

    @staticmethod
    def _batch_and_outputs() -> tuple[dict, ModelStepOutput]:
        """Build a test batch with a climatology key and matching ModelStepOutput."""
        batch = {
            "input": torch.rand(1, 1, 1, 2, 2),
            "climatology": torch.rand(1, 3, 1, 2, 2),
        }
        outputs = ModelStepOutput(
            prediction=torch.rand(1, 3, 1, 2, 2),
            target=torch.rand(1, 3, 1, 2, 2),
            loss=torch.tensor(0.0),
        )
        return batch, outputs

    def test_on_test_batch_end_builds_climatology_metrics(
        self,
        mock_module: MagicMock,
    ) -> None:
        """The first batch containing a climatology entry builds the collection."""
        callback = MetricSummaryCallback()
        mock_module.test_metrics = MetricCollection(
            {"accuracy": IceNetAccuracyPerForecastDay()}
        )
        batch, outputs = self._batch_and_outputs()

        callback.on_test_batch_end(
            MagicMock(spec=Trainer), mock_module, outputs, batch, 0
        )

        assert callback.climatology_metrics is not None
        assert set(callback.climatology_metrics) == {"accuracy"}
        assert callback.climatology_metrics["accuracy"].update_called is True

    def test_on_test_batch_end_noop_without_climatology_key(
        self,
        mock_module: MagicMock,
    ) -> None:
        """Batches without a climatology key leave the callback state untouched."""
        callback = MetricSummaryCallback()
        mock_module.test_metrics = MetricCollection(
            {"accuracy": IceNetAccuracyPerForecastDay()}
        )
        _, outputs = self._batch_and_outputs()
        batch = {
            "input": torch.rand(1, 1, 1, 2, 2),
            "target": torch.rand(1, 3, 1, 2, 2),
        }

        callback.on_test_batch_end(
            MagicMock(spec=Trainer), mock_module, outputs, batch, 0
        )

        assert callback.climatology_metrics is None

    def test_on_test_batch_end_noop_when_outputs_not_mapping(
        self,
        mock_module: MagicMock,
    ) -> None:
        """A non-Mapping outputs value (e.g. a bare Tensor) is ignored safely."""
        callback = MetricSummaryCallback()
        mock_module.test_metrics = MetricCollection(
            {"accuracy": IceNetAccuracyPerForecastDay()}
        )
        batch, _ = self._batch_and_outputs()

        callback.on_test_batch_end(
            MagicMock(spec=Trainer), mock_module, torch.rand(1), batch, 0
        )

        assert callback.climatology_metrics is None

    def test_on_test_epoch_start_resets_climatology_metrics(
        self,
        mock_module: MagicMock,
    ) -> None:
        """The climatology collection is reset at the start of each test epoch."""
        callback = MetricSummaryCallback()
        mock_module.test_metrics = MetricCollection(
            {"accuracy": IceNetAccuracyPerForecastDay()}
        )
        batch, outputs = self._batch_and_outputs()
        callback.on_test_batch_end(
            MagicMock(spec=Trainer), mock_module, outputs, batch, 0
        )
        assert callback.climatology_metrics is not None
        assert callback.climatology_metrics["accuracy"].update_called is True

        callback.on_test_epoch_start(MagicMock(spec=Trainer), mock_module)

        assert callback.climatology_metrics["accuracy"].update_called is False

    def test_teardown_includes_climatology_baseline(
        self,
        mock_module: MagicMock,
        wandb_run: tuple[MagicMock, MockWandbRun],
    ) -> None:
        """The per-run plot carries both the test and climatology series."""
        callback = MetricSummaryCallback()
        mock_wandb, mock_run = wandb_run
        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False

        mock_module.test_metrics = MetricCollection(
            {"accuracy": IceNetAccuracyPerForecastDay()}
        )
        batch, outputs = self._batch_and_outputs()
        # Mirror BaseModel.test_step, which updates the model's test metrics.
        mock_module.test_metrics.update(outputs["prediction"], outputs["target"])
        callback.on_test_batch_end(trainer, mock_module, outputs, batch, 0)

        callback.teardown(trainer, mock_module, stage=TrainerFn.TESTING.value)

        mock_wandb.plot.line_series.assert_called_once()
        line_series_kwargs = mock_wandb.plot.line_series.call_args[1]
        assert line_series_kwargs["keys"] == ["test", "climatology"]
        assert line_series_kwargs["title"] == "accuracy_per_forecast_day"
        assert line_series_kwargs["xs"] == [1, 2, 3]
        ys = line_series_kwargs["ys"]
        assert len(ys) == 2
        assert all(len(series) == 3 for series in ys)
        mock_run.log.assert_called_once()

    def test_teardown_without_climatology_omits_baseline(
        self,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A test run whose batches never carried climatology has no baseline stage."""
        callback = MetricSummaryCallback()
        mock_module.test_metrics = MetricCollection({"mae": MAEPerForecastDay()})
        mock_module.test_metrics.update(
            torch.rand(1, 3, 1, 2, 2), torch.rand(1, 3, 1, 2, 2)
        )
        mock_log_per_run_metrics = MagicMock()
        monkeypatch.setattr(callback, "log_per_run_metrics", mock_log_per_run_metrics)

        callback.teardown(
            MagicMock(spec=Trainer), mock_module, stage=TrainerFn.TESTING.value
        )

        mock_log_per_run_metrics.assert_called_once()
        metrics = mock_log_per_run_metrics.call_args[0][1]
        assert set(metrics) == {"test"}
