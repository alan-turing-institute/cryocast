from pathlib import Path

from icenet_mp.data import SingleDataset
from icenet_mp.types import Metadata, PlotSpec, Timespan
from icenet_mp.utils import (
    datetime_from_npdatetime,
    iso_from_date,
    mask_dir,
    sanitise_filename,
)

from .land_mask import LandMask
from .panel_renderer import PanelRenderer


class DatasetMediaWriter:
    """Write static and video media for the variables in downloaded datasets."""

    def __init__(self, base_path: Path) -> None:
        """Initialise the media writer with the base path for downloaded datasets."""
        self.base_path = base_path

    def _prepare(self, dataset: SingleDataset) -> tuple[PanelRenderer, Path]:
        """Build the renderer and output directory shared by `static` and `video`."""
        # Unset the default vmin/vmax since this dataset is unnormalised.
        plot_spec = PlotSpec(hemisphere=dataset.hemisphere, vmin=None, vmax=None)
        land_mask = LandMask(mask_dir(self.base_path, dataset.name) / "land_mask.npy")
        renderer = PanelRenderer(land_mask, Metadata(), plot_spec)
        output_dir = self.base_path / "data" / "input_plots" / dataset.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return renderer, output_dir

    def static(
        self,
        *,
        dataset: SingleDataset,
        timestep: int,
    ) -> int:
        """Save static plots for one timestep of a downloaded dataset."""
        if timestep < 0 or timestep >= len(dataset):
            msg = (
                f"Timestep {timestep} is out of range for dataset {dataset.name} "
                f"with {len(dataset)} timesteps"
            )
            raise IndexError(msg)

        when = datetime_from_npdatetime(dataset.dates[timestep])
        frame = dataset[timestep]
        variables = {
            f"{dataset.group}:{variable_name}": frame[channel]
            for channel, variable_name in enumerate(dataset.variable_names)
        }
        renderer, output_dir = self._prepare(dataset)

        saved = 0
        for variable_name, variable_values in variables.items():
            image = renderer.static_singlet(
                variable_values,
                when=when,
                variable_name=variable_name,
            )
            image.save(
                output_dir
                / sanitise_filename(f"{iso_from_date(when)}-{variable_name}.png")
            )
            saved += 1
        return saved

    def video(
        self,
        *,
        dataset: SingleDataset,
        n_steps: int,
        timestep: int,
    ) -> int:
        """Save one animation per variable for a run of consecutive timesteps."""
        if timestep < 0 or n_steps < 1 or timestep + n_steps > len(dataset):
            msg = (
                f"Timesteps {timestep}:{timestep + n_steps} are out of range for dataset "
                f"{dataset.name} with {len(dataset)} timesteps"
            )
            raise IndexError(msg)

        dates = Timespan(
            datetime_from_npdatetime(date)
            for date in dataset.dates[timestep : timestep + n_steps]
        )
        tchw = dataset.get_tchw_slice(dataset.dates[timestep], n_steps)
        variables = {
            f"{dataset.group}:{variable_name}": tchw[:, channel]
            for channel, variable_name in enumerate(dataset.variable_names)
        }
        renderer, output_dir = self._prepare(dataset)

        saved = 0
        for variable_name, variable_values in variables.items():
            video_buffer = renderer.video_singlet(
                variable_values,
                dates=dates,
                variable_name=variable_name,
            )
            video_buffer.seek(0)
            video_path = output_dir / sanitise_filename(
                f"{iso_from_date(dates[0])}-{variable_name}.{renderer.video_format}"
            )
            video_path.write_bytes(video_buffer.read())
            saved += 1
        return saved
