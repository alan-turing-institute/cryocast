import logging
from pathlib import Path

import pytest

from icenet_mp.data import SingleDataset
from icenet_mp.visualisations import DatasetMediaWriter

from .conftest import CustomCliRunner


class TestDatasetsCLI:
    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["datasets", "--help"],
            expected_patterns=[
                r"Usage: imp datasets \[OPTIONS\] COMMAND \[ARGS\]...",
                r"Manage datasets",
                r"--help\s+-h\s+Show this message and exit.",
                r"create\s+Create all datasets.",
                r"inspect\s+Inspect all datasets.",
                r"plot\s+Plot one timestep of configured datasets.",
                r"masks\s+Create land / active grid cell masks.",
            ],
        )


class TestDatasetsCreateCLI:
    class FakeDownloader:
        """A minimal downloader stub exposing only what `create` needs."""

        def __init__(self, name: str, *, error: Exception | None = None) -> None:
            """Store the downloader's name and an optional error to raise."""
            self.name = name
            self.error = error
            self.create_calls: list[bool] = []

        def create(self, *, overwrite: bool) -> None:
            """Record the overwrite flag and raise the configured error, if any."""
            self.create_calls.append(overwrite)
            if self.error is not None:
                raise self.error

    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["datasets", "create", "--help"],
            expected_patterns=[
                r"Usage: imp datasets create \[OPTIONS\] \[overrides\]...",
                r"Create all datasets.",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--config-name\s+<str>\s+Name of a file to load from the",
                r"--overwrite\s+--no-overwrite\s+Specify whether to overwrite",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    def test_calls_create_on_each_downloader_with_overwrite_flag(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Call create(overwrite=True) on every downloader when --overwrite is set."""
        first = self.FakeDownloader("first")
        second = self.FakeDownloader("second")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [first, second]
        )

        result = runner.call(["datasets", "create", "--overwrite"])

        assert result.exit_code == 0, result.output
        assert first.create_calls == [True]
        assert second.create_calls == [True]

    def test_defaults_overwrite_to_false(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default --overwrite to False when not passed."""
        downloader = self.FakeDownloader("example")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = runner.call(["datasets", "create"])

        assert result.exit_code == 0, result.output
        assert downloader.create_calls == [False]

    def test_exits_with_error_and_stops_on_runtime_error(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Exit 1 and skip remaining downloaders when one fails to create."""
        failing = self.FakeDownloader("failing", error=RuntimeError("boom"))
        never_reached = self.FakeDownloader("never-reached")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [failing, never_reached],
        )

        with caplog.at_level(logging.ERROR):
            result = runner.call(["datasets", "create"])

        assert result.exit_code == 1
        assert "Failed to create failing: boom" in caplog.text
        assert never_reached.create_calls == []


class TestDatasetsInspectCLI:
    class FakeDownloader:
        """A minimal downloader stub exposing only what `inspect` needs."""

        def __init__(self, name: str, *, error: Exception | None = None) -> None:
            """Store the downloader's name and an optional error to raise."""
            self.name = name
            self.error = error
            self.inspect_calls: list[bool] = []

        def inspect(self, *, verbose: bool) -> None:
            """Record the verbose flag and raise the configured error, if any."""
            self.inspect_calls.append(verbose)
            if self.error is not None:
                raise self.error

    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["datasets", "inspect", "--help"],
            expected_patterns=[
                r"Usage: imp datasets inspect \[OPTIONS\] \[overrides\]...",
                r"Inspect all datasets.",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--config-name\s+<str>\s+Name of a file to load from the",
                r"--verbose\s+--no-verbose\s+Show detailed dataset information",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    def test_calls_inspect_on_each_downloader_with_verbose_flag(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Call inspect(verbose=True) on every downloader when --verbose is set."""
        first = self.FakeDownloader("first")
        second = self.FakeDownloader("second")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [first, second]
        )

        result = runner.call(["datasets", "inspect", "--verbose"])

        assert result.exit_code == 0, result.output
        assert first.inspect_calls == [True]
        assert second.inspect_calls == [True]

    def test_defaults_verbose_to_false(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default --verbose to False when not passed."""
        downloader = self.FakeDownloader("example")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = runner.call(["datasets", "inspect"])

        assert result.exit_code == 0, result.output
        assert downloader.inspect_calls == [False]

    def test_logs_and_continues_past_runtime_error(
        self,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Log and continue with remaining downloaders when one fails to inspect."""
        failing = self.FakeDownloader("failing", error=RuntimeError("boom"))
        still_runs = self.FakeDownloader("still-runs")
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [failing, still_runs],
        )

        with caplog.at_level(logging.ERROR):
            result = runner.call(["datasets", "inspect"])

        assert result.exit_code == 0, result.output
        assert "Inspecting dataset failing failed, skipping." in caplog.text
        assert still_runs.inspect_calls == [False]


class TestDatasetsPlotCLI:
    class FakeDownloader:
        """A minimal downloader stub exposing only what `plot` needs."""

        def __init__(
            self, name: str, path_dataset: Path, group: str | None = None
        ) -> None:
            """Store the downloader's name, group, and dataset path."""
            self.name = name
            self.group = name if group is None else group
            self.path_dataset = path_dataset

    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["datasets", "plot", "--help"],
            expected_patterns=[
                r"Usage: imp datasets plot \[OPTIONS\] \[overrides\]...",
                r"Plot one timestep of configured datasets.",
                r"--dataset\s+<str>\s+Only plot the named configured",
                r"--timestep\s+<int>\s+Dataset timestep index to plot",
                r"--video\s+--no-video\s+Animate --n-steps consecutive",
                r"--n-steps\s+<int>\s+Number of consecutive timesteps to",
            ],
        )

    def test_plot_calls_plot_dataset_for_each_matched_existing_dataset(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Call plot_dataset once per configured downloader whose data exists."""
        existing_path = tmp_path / "example.zarr"
        existing_path.mkdir()
        downloaders = [
            self.FakeDownloader("example", existing_path),
            self.FakeDownloader("missing", tmp_path / "missing.zarr"),
        ]
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: downloaders
        )

        calls = []

        def fake_plot_dataset(
            self: DatasetMediaWriter,
            *,
            dataset: SingleDataset,
            timestep: int,
        ) -> int:
            calls.append((self.base_path, dataset.name, dataset._input_files, timestep))
            return 2

        monkeypatch.setattr(DatasetMediaWriter, "static", fake_plot_dataset)

        result = runner.call(
            ["datasets", "plot", f"base_path={tmp_path}", "--timestep", "1"]
        )

        assert result.exit_code == 0, result.output
        assert calls == [(tmp_path.resolve(), "example", (existing_path,), 1)]

    def test_plot_video_calls_plot_dataset_video_with_n_steps(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Call plot_dataset_video instead of plot_dataset when --video is passed."""
        existing_path = tmp_path / "example.zarr"
        existing_path.mkdir()
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [self.FakeDownloader("example", existing_path)],
        )
        monkeypatch.setattr(
            DatasetMediaWriter,
            "static",
            lambda *_args, **_kwargs: pytest.fail(
                "DatasetMediaWriter.static should not be called with --video"
            ),
        )

        calls = []

        def fake_plot_dataset_video(
            self: DatasetMediaWriter,
            *,
            dataset: SingleDataset,
            timestep: int,
            n_steps: int,
        ) -> int:
            calls.append(
                (self.base_path, dataset.name, dataset._input_files, timestep, n_steps)
            )
            return 4

        monkeypatch.setattr(DatasetMediaWriter, "video", fake_plot_dataset_video)

        result = runner.call(
            [
                "datasets",
                "plot",
                f"base_path={tmp_path}",
                "--video",
                "--n-steps",
                "5",
            ]
        )

        assert result.exit_code == 0, result.output
        assert calls == [(tmp_path.resolve(), "example", (existing_path,), 0, 5)]

    def test_plot_rejects_unmatched_dataset_name(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exit non-zero when --dataset names a dataset that isn't configured."""
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [self.FakeDownloader("example", tmp_path / "example.zarr")],
        )

        result = runner.call(["datasets", "plot", "--dataset", "unknown"])

        assert result.exit_code == 1


class TestDatasetsPostProcessorCLI:
    class FakePostprocessor:
        """A minimal postprocessor stub exposing only what `masks` needs."""

        def __init__(self, *, error: Exception | None = None) -> None:
            """Store an optional error to raise from process()."""
            self.error = error
            self.process_calls: list[tuple[Path, bool]] = []

        def process(self, path_dataset: Path, *, overwrite: bool) -> None:
            """Record the call and raise the configured error, if any."""
            self.process_calls.append((path_dataset, overwrite))
            if self.error is not None:
                raise self.error

    class FakeDownloader:
        """A minimal downloader stub exposing only what `masks` needs."""

        def __init__(
            self, name: str, path_dataset: Path, postprocessor: object
        ) -> None:
            """Store the downloader's name, dataset path, and postprocessor."""
            self.name = name
            self.path_dataset = path_dataset
            self.postprocessor = postprocessor

    def test_calls_postprocessor_process_with_overwrite_flag(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Call postprocessor.process(path_dataset, overwrite=True) per downloader."""
        postprocessor = self.FakePostprocessor()
        downloader = self.FakeDownloader(
            "example", tmp_path / "example.zarr", postprocessor
        )
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = runner.call(["datasets", "masks", "--overwrite"])

        assert result.exit_code == 0, result.output
        assert postprocessor.process_calls == [(downloader.path_dataset, True)]

    def test_defaults_overwrite_to_false(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default --overwrite to False when not passed."""
        postprocessor = self.FakePostprocessor()
        downloader = self.FakeDownloader(
            "example", tmp_path / "example.zarr", postprocessor
        )
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = runner.call(["datasets", "masks"])

        assert result.exit_code == 0, result.output
        assert postprocessor.process_calls == [(downloader.path_dataset, False)]

    def test_propagates_errors_without_catching_them(
        self,
        tmp_path: Path,
        runner: CustomCliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Let a postprocessor failure propagate uncaught, unlike create/inspect."""
        postprocessor = self.FakePostprocessor(error=RuntimeError("boom"))
        downloader = self.FakeDownloader(
            "example", tmp_path / "example.zarr", postprocessor
        )
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: [downloader]
        )

        result = runner.call(["datasets", "masks"])

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
