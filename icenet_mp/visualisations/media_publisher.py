import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from io import BytesIO
from typing import TypeVar

import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.data import CombinedDataset, SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import (
    ArrayHW,
    ArrayTCHW,
    ArrayTHW,
    Metadata,
    ModelStepOutput,
    PlotSpec,
    SupportsImageLogging,
    SupportsVideoLogging,
    Timespan,
)
from icenet_mp.utils import iso_from_date, npdatetime_from_datetime

from .land_mask import LandMask
from .panel_renderer import PanelRenderer

log = logging.getLogger(__name__)

RenderedMedia = TypeVar("RenderedMedia", ImageFile, BytesIO)


class MediaPublisher:
    """Publish static and video plots for a dataset/plot_spec/land_mask context."""

    def __init__(
        self,
        *,
        dataset: CombinedDataset,
        plot_spec: PlotSpec,
        land_mask: LandMask,
        model_name: str | None = None,
        trained_epochs: int | None = None,
    ) -> None:
        """Build a publisher bound to one dataset/plot_spec/land_mask context.

        Args:
            dataset: The training dataset; may be different from the one plotted.
            plot_spec: Plotting specification (difference settings, timestep, etc.).
            land_mask: Land mask to apply when rendering panels.
            model_name: Model name, shown in the footer if given.
            trained_epochs: The number of epochs trained for.

        """
        self.idx_date = plot_spec.selected_timestep
        self.panel_renderer = PanelRenderer(
            land_mask,
            Metadata.from_dataset(
                dataset, model_name=model_name, trained_epochs=trained_epochs
            ),
            plot_spec,
        )

    @staticmethod
    def _channel_name(channel_names: list[str], idx_channel: int) -> str:
        """Return the configured channel name or a stable fallback."""
        if idx_channel < len(channel_names):
            return channel_names[idx_channel]
        return f"channel_{idx_channel}"

    @staticmethod
    def _log_images(
        images: dict[str, list[ImageFile]],
        image_loggers: list[SupportsImageLogging],
        log_path: str,
    ) -> None:
        """Send rendered image groups to every configured image logger."""
        for image_name, image_list in images.items():
            for image_logger in image_loggers:
                image_logger.log_image(
                    key=f"{log_path}/{image_name}", images=image_list
                )

    @staticmethod
    def _log_path(prefix: str | None, name: str) -> str:
        """Build a consistent logger namespace."""
        return f"{prefix}/{name}" if prefix else name

    def _log_videos(
        self,
        videos: dict[str, BytesIO],
        video_loggers: list[SupportsVideoLogging],
        log_path: str,
    ) -> None:
        """Rewind and send rendered videos to every configured video logger."""
        for video_logger in video_loggers:
            for video_name, video_buffer in videos.items():
                video_buffer.seek(0)
                video_logger.log_video(
                    key=f"{log_path}/{video_name}",
                    videos=[video_buffer],
                    format=[self.panel_renderer.video_format],
                )

    def _render_multipanel_media(  # noqa: PLR0913
        self,
        *,
        climatology: np.ndarray | None,
        compare_truth_climatology: bool = False,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        render: Callable[..., RenderedMedia],
        uncertainty: np.ndarray | None,
        variable_name: str,
        **render_kwargs: object,
    ) -> dict[str, RenderedMedia]:
        """Render the configured multipanel media.

        These may include:
        - truth/prediction/difference
        - climatology/prediction/difference
        - truth/climatology/difference
        - truth/prediction/z-score

        Shared by the static and video loggers, which differ only in the `render`
        callable used and the extra history/forecast context keywords each one needs.
        """
        media: dict[str, RenderedMedia] = {
            "truth-vs-prediction": render(
                ground_truth, prediction, variable_name=variable_name, **render_kwargs
            )
        }
        if climatology is not None:
            with suppress(IndexError, InvalidArrayError, TypeError):
                media["climatology-vs-prediction"] = render(
                    climatology,
                    prediction,
                    panel_titles={"ground_truth": "Climatology"},
                    variable_name=variable_name,
                    **render_kwargs,
                )
            # Only render a truth/climatology/difference triplet once
            if compare_truth_climatology:
                with suppress(IndexError, InvalidArrayError, TypeError):
                    # Drop history context so the header will be set correctly
                    filtered_render_kwargs = {
                        key: None if key == "history_ctx" else value
                        for key, value in render_kwargs.items()
                    }
                    media["truth-vs-climatology"] = render(
                        ground_truth,
                        climatology,
                        panel_titles={"prediction": "Climatology"},
                        variable_name=variable_name,
                        **filtered_render_kwargs,
                    )
        # Only render a z-score output if the plot spec is configured to include a
        # difference panel and an uncertainty array is provided.
        if uncertainty is not None and self.panel_renderer.plot_spec.include_difference:
            media["truth-vs-prediction-z-score"] = render(
                ground_truth,
                prediction,
                panel_titles={"difference": "Standardised Difference (z)"},
                uncertainty=uncertainty,
                variable_name=variable_name,
                **render_kwargs,
            )
        return media

    @staticmethod
    def _select_climatology(
        climatology: ArrayTCHW | None,
        idx_channel: int,
        idx_date: int | None,
    ) -> np.ndarray | None:
        """Return the climatology array for a given channel and (optionally) date."""
        if climatology is None:
            return None
        with suppress(IndexError, TypeError):
            return climatology[
                slice(None) if idx_date is None else idx_date, idx_channel
            ]
        return None

    @staticmethod
    def _select_uncertainty(
        uncertainties: dict[int, ArrayTHW] | None,
        idx_channel: int,
        idx_date: int | None,
    ) -> np.ndarray | None:
        """Return the uncertainty array for a given channel and (optionally) date."""
        if (
            uncertainties is None
            or (uncertainty := uncertainties.get(idx_channel)) is None
        ):
            return None
        return uncertainty[idx_date] if idx_date is not None else uncertainty

    def log_static_inputs(
        self,
        inputs: list[SingleDataset],
        dates: list[datetime],
        image_loggers: list[SupportsImageLogging],
        *,
        prefix: str | None = None,
    ) -> None:
        """Extract and log static raw input plots.

        Args:
            inputs: List of SingleDataset instances containing the input data.
            dates: List of datetime objects corresponding to the timesteps in the datasets.
            image_loggers: List of image loggers to send the rendered images to.
            prefix: Optional prefix for the log path to namespace the logged images.

        """
        try:
            when = dates[self.idx_date]
            log_path = self._log_path(prefix, "input_static")
            date_key = iso_from_date(when)
            images: dict[str, list[ImageFile]] = {}
            for input_ds in inputs:
                # Get data for all variables at the selected timestep
                for idx_channel, v_name in enumerate(input_ds.variable_names):
                    variable_name = f"{input_ds.name}:{v_name}"
                    # Render a single-panel static image for this variable
                    images[f"{date_key}-{variable_name}"] = [
                        self.panel_renderer.static_singlet(
                            input_ds[self.idx_date][idx_channel, :],
                            when=when,
                            variable_name=variable_name,
                        )
                    ]
            # Log input images
            self._log_images(images, image_loggers, log_path)
        except (InvalidArrayError, IndexError, ValueError, MemoryError, OSError) as exc:
            log.warning("Image logging failed: %s", exc)

    def log_static_outputs(  # noqa: PLR0913
        self,
        outputs: ModelStepOutput,
        image_loggers: list[SupportsImageLogging],
        *,
        channel_names: list[str] | None = None,
        climatology: ArrayTCHW | None = None,
        compare_truth_climatology: bool = False,
        forecast_dates: list[datetime],
        history_dates: list[datetime],
        prefix: str | None = None,
        uncertainties: dict[int, ArrayTHW] | None = None,
    ) -> None:
        """Create and log static output plots, including climatology when available.

        When a matching entry is present, also logs a standardised uncertainty
        (z-score) plot and a calendar-day-mean (climatology) map for the plotted
        date and channel.

        Args:
            outputs: ModelStepOutput containing the ground truth and prediction arrays.
            image_loggers: List of image loggers to send the rendered images to.
            channel_names: Optional list of channel names corresponding to the output channels.
            climatology: Optional climatology array for the output channels.
            compare_truth_climatology: Whether to also log a ground-truth/climatology/
                difference triplet, with a plain "on <date>" title and no footer since
                it doesn't depend on the model. Callers should only set this once per
                date rather than every epoch.
            forecast_dates: List of forecast dates corresponding to the output timesteps.
            history_dates: List of history dates corresponding to the output timesteps.
            prefix: Optional prefix for the log path to namespace the logged images.
            uncertainties: Optional dictionary mapping channel indices to uncertainty arrays for the output channels.

        """
        try:
            log_path = self._log_path(prefix, "output_static")
            date_key = iso_from_date(forecast_dates[self.idx_date])
            images: dict[str, list[ImageFile]] = {}
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayHW = (
                    outputs.target[0, self.idx_date, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayHW = (
                    outputs.prediction[0, self.idx_date, idx_channel]
                    .detach()
                    .cpu()
                    .numpy()
                )
                variable_name = self._channel_name(channel_names or [], idx_channel)
                images.update(
                    {
                        f"{date_key}-{variable_name}-{suffix}": [image]
                        for suffix, image in self._render_multipanel_media(
                            climatology=self._select_climatology(
                                climatology, idx_channel, self.idx_date
                            ),
                            forecast_date=forecast_dates[self.idx_date],
                            ground_truth=ground_truth,
                            history_ctx=Timespan(history_dates),
                            compare_truth_climatology=compare_truth_climatology,
                            prediction=prediction,
                            render=self.panel_renderer.static_triplet,
                            uncertainty=self._select_uncertainty(
                                uncertainties, idx_channel, self.idx_date
                            ),
                            variable_name=variable_name,
                        ).items()
                    }
                )
            # Log output images
            self._log_images(images, image_loggers, log_path)
        except (InvalidArrayError, IndexError, ValueError, MemoryError, OSError) as exc:
            log.warning("Image logging failed: %s", exc)

    def log_video_inputs(
        self,
        inputs: list[SingleDataset],
        dates: list[datetime],
        video_loggers: list[SupportsVideoLogging],
        *,
        prefix: str | None = None,
    ) -> None:
        """Extract and log raw input videos.

        Args:
            inputs: List of SingleDataset instances containing the input data.
            dates: List of datetime objects corresponding to the timesteps in the datasets.
            video_loggers: List of video loggers to send the rendered videos to.
            prefix: Optional prefix for the log path to namespace the logged videos.

        """
        try:
            log_path = self._log_path(prefix, "input_video")
            np_dates = [npdatetime_from_datetime(date) for date in dates]
            date_key = iso_from_date(dates[0])
            videos: dict[str, BytesIO] = {}
            for input_ds in inputs:
                # Get data for all variables over the full date range
                for channel, unqualified_name in enumerate(input_ds.variable_names):
                    variable_name = f"{input_ds.name}:{unqualified_name}"
                    videos[f"{date_key}-{variable_name}"] = (
                        self.panel_renderer.video_singlet(
                            input_ds.get_tchw(np_dates)[:, channel, :],
                            dates=Timespan(dates),
                            variable_name=variable_name,
                        )
                    )
            # Log input videos
            self._log_videos(videos, video_loggers, log_path)
        except (
            IndexError,
            InvalidArrayError,
            MemoryError,
            OSError,
            ValueError,
            VideoRenderError,
        ) as exc:
            log.warning("Video logging failed: %s", exc)

    def log_video_outputs(  # noqa: PLR0913
        self,
        outputs: ModelStepOutput,
        video_loggers: list[SupportsVideoLogging],
        *,
        channel_names: list[str] | None = None,
        climatology: ArrayTCHW | None = None,
        forecast_dates: list[datetime],
        history_dates: list[datetime],
        compare_truth_climatology: bool = False,
        prefix: str | None = None,
        uncertainties: dict[int, ArrayTHW] | None = None,
    ) -> None:
        """Create and log output videos.

        Args:
            outputs: ModelStepOutput containing the ground truth and prediction arrays.
            video_loggers: List of video loggers to send the rendered videos to.
            channel_names: Optional list of channel names corresponding to the output channels.
            climatology: Optional climatology array for the output channels.
            forecast_dates: List of forecast dates corresponding to the output timesteps.
            history_dates: List of history dates corresponding to the output timesteps.
            compare_truth_climatology: Whether to also log a ground-truth/climatology/
                difference triplet, with a plain "on <date>" title and no footer since
                it doesn't depend on the model. Callers should only set this once per
                date rather than every epoch.
            prefix: Optional prefix for the log path to namespace the logged videos.
            uncertainties: Optional dictionary mapping channel indices to uncertainty arrays for the output channels.

        """
        try:
            log_path = self._log_path(prefix, "output_video")
            date_key = iso_from_date(forecast_dates[0])
            videos: dict[str, BytesIO] = {}
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayTHW = (
                    outputs.target[0, :, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayTHW = (
                    outputs.prediction[0, :, idx_channel].detach().cpu().numpy()
                )
                variable_name = self._channel_name(channel_names or [], idx_channel)
                videos.update(
                    {
                        f"{date_key}-{variable_name}-{suffix}": video
                        for suffix, video in self._render_multipanel_media(
                            climatology=self._select_climatology(
                                climatology, idx_channel, None
                            ),
                            forecast_ctx=Timespan(forecast_dates),
                            ground_truth=ground_truth,
                            history_ctx=Timespan(history_dates),
                            compare_truth_climatology=compare_truth_climatology,
                            prediction=prediction,
                            render=self.panel_renderer.video_triplet,
                            uncertainty=self._select_uncertainty(
                                uncertainties, idx_channel, None
                            ),
                            variable_name=variable_name,
                        ).items()
                    }
                )
            # Log output videos
            self._log_videos(videos, video_loggers, log_path)
        except (
            IndexError,
            InvalidArrayError,
            MemoryError,
            OSError,
            ValueError,
            VideoRenderError,
        ) as exc:
            log.warning("Video logging failed: %s", exc)
