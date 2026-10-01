import io
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pytest
from PIL import Image

from icenet_mp.data import SingleDataset
from icenet_mp.visualisations import DatasetMediaWriter
from icenet_mp.visualisations.panel_renderer import PanelRenderer


def fake_dataset() -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in, cast to satisfy static's typing."""

    class FakeDataset:
        """Minimal dataset stub for plotting tests."""

        name = "example"
        group = "example"
        hemisphere = "north"
        dates: ClassVar[list[np.datetime64]] = [np.datetime64("2020-01-01")]
        variable_names: ClassVar[list[str]] = ["ice_conc", "temperature"]

        def __len__(self) -> int:
            """Return the single available timestep."""
            return 1

        def __getitem__(self, _idx: int) -> np.ndarray:
            """Return two deterministic variables for the requested timestep."""
            return np.ones((2, 4, 4), dtype=np.float32)

    return cast("SingleDataset", FakeDataset())


def fake_video_dataset() -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in, cast to satisfy video's typing."""

    class FakeVideoDataset:
        """Minimal dataset stub for video plotting tests."""

        name = "example"
        group = "example"
        hemisphere = "north"
        dates: ClassVar[list[np.datetime64]] = [
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.datetime64("2020-01-03"),
        ]
        variable_names: ClassVar[list[str]] = ["ice_conc", "temperature"]

        def __len__(self) -> int:
            """Return the three available timesteps."""
            return 3

        def get_tchw_slice(
            self, _start_date: np.datetime64, n_steps: int
        ) -> np.ndarray:
            """Return deterministic values for the requested number of timesteps."""
            return np.ones((n_steps, 2, 4, 4), dtype=np.float32)

    return cast("SingleDataset", FakeVideoDataset())


class TestPrepare:
    def test_plot_spec_disables_default_zero_one_range(self, tmp_path: Path) -> None:
        """Raw (unnormalised) dataset previews auto-infer their colour range.

        DatasetMediaWriter plots datasets loaded with normalise=False, so unlike
        MediaPublisher's already-[0, 1]-normalised training/evaluation inputs, the
        PlotSpec default vmin/vmax=[0, 1] would otherwise clip real physical values
        (e.g. Kelvin temperatures) to a flat colour.
        """
        renderer, _ = DatasetMediaWriter(tmp_path)._prepare(fake_dataset())

        assert renderer.plot_spec.vmin is None
        assert renderer.plot_spec.vmax is None


class TestPlotDataset:
    def test_plot_dataset_saves_each_variable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save one PNG for each variable returned by the plotting helper."""

        def fake_static_singlet(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return Image.new("RGB", (4, 4))

        monkeypatch.setattr(PanelRenderer, "static_singlet", fake_static_singlet)

        saved = DatasetMediaWriter(tmp_path).static(
            dataset=fake_dataset(),
            timestep=0,
        )

        output_dir = tmp_path / "data" / "input_plots"
        assert saved == 2
        assert (output_dir / "example" / "2020-01-01-example_ice_conc.png").is_file()
        assert (output_dir / "example" / "2020-01-01-example_temperature.png").is_file()

    def test_plot_dataset_rejects_out_of_range_timestep(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject a timestep index outside the available dataset range."""
        with pytest.raises(IndexError, match="Timestep 2 is out of range"):
            DatasetMediaWriter(tmp_path).static(
                dataset=fake_dataset(),
                timestep=2,
            )


class TestPlotDatasetVideo:
    def test_plot_dataset_video_saves_each_variable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save one video for each variable returned by the plotting helper."""

        def fake_video_singlet(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return io.BytesIO(b"video data")

        monkeypatch.setattr(PanelRenderer, "video_singlet", fake_video_singlet)

        saved = DatasetMediaWriter(tmp_path).video(
            dataset=fake_video_dataset(),
            n_steps=3,
            timestep=0,
        )

        output_dir = tmp_path / "data" / "input_plots"
        assert saved == 2
        assert (output_dir / "example" / "2020-01-01-example_ice_conc.mp4").is_file()
        assert (output_dir / "example" / "2020-01-01-example_temperature.mp4").is_file()

    def test_plot_dataset_video_rejects_out_of_range_timestep(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject a timestep/n_steps combination outside the dataset range."""
        with pytest.raises(IndexError, match="Timesteps 2:5 are out of range"):
            DatasetMediaWriter(tmp_path).video(
                dataset=fake_video_dataset(),
                n_steps=3,
                timestep=2,
            )
