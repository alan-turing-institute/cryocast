import logging
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, ClassVar, cast
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch

from icenet_mp.data import CombinedDataset, SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import Hemisphere, ModelStepOutput, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.matplotlib_renderer import MatplotlibRenderer
from icenet_mp.visualisations.media_publisher import MediaPublisher

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile

N_TIMESTEPS = 2
N_CHANNELS = 2
HEIGHT = 4
WIDTH = 4

FORECAST_DATES = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
HISTORY_DATES = [datetime(2019, 12, 30), datetime(2019, 12, 31)]


def fake_single_dataset() -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in, cast to satisfy MediaPublisher's typing."""

    class FakeSingleDataset:
        """Minimal SingleDataset stand-in exposing the attributes MediaPublisher reads."""

        name = "example"
        variable_names: ClassVar[list[str]] = ["ice_conc", "temperature"]

        def __getitem__(self, _idx: int) -> np.ndarray:
            """Return a deterministic [C, H, W] array for the requested timestep."""
            return np.ones((N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

        def get_tchw(self, dates: list) -> np.ndarray:
            """Return a deterministic [T, C, H, W] array for the requested dates."""
            return np.ones((len(dates), N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

    return cast("SingleDataset", FakeSingleDataset())


def fake_combined_dataset(
    *,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-10",
    frequency: np.timedelta64 | None = None,
    length: int = 10,
    n_history_steps: int = 0,
    inputs: list[SingleDataset] | None = None,
) -> CombinedDataset:
    """Return a duck-typed CombinedDataset stand-in for MediaPublisher's dataset arg."""

    class FakeCombinedDataset:
        """Minimal CombinedDataset stand-in for MediaPublisher's metadata construction."""

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

    return cast("CombinedDataset", FakeCombinedDataset())


def make_model_step_output(channels: int = N_CHANNELS) -> ModelStepOutput:
    """Build a ModelStepOutput with real tensors shaped [N, T, C, H, W]."""
    shape = (1, N_TIMESTEPS, channels, HEIGHT, WIDTH)
    return ModelStepOutput(
        prediction=torch.zeros(shape),
        target=torch.ones(shape),
        loss=torch.tensor(0.0),
    )


class TestLoggingHelpers:
    def test_log_path_handles_optional_prefix(self) -> None:
        """Build the same namespaces for prefixed and unprefixed logging."""
        assert MediaPublisher._log_path(None, "output_static") == "output_static"
        assert MediaPublisher._log_path("test", "output_static") == "test/output_static"

    def test_channel_name_uses_stable_fallback(self) -> None:
        """Use configured names when available and indexed fallbacks otherwise."""
        assert MediaPublisher._channel_name(["sic"], 0) == "sic"
        assert MediaPublisher._channel_name(["sic"], 2) == "channel_2"

    def test_log_images_fans_out_to_all_loggers(self) -> None:
        """Send every image group to each configured logger."""
        first = MagicMock()
        second = MagicMock()
        images = cast(
            "dict[str, list[ImageFile]]",
            {"comparison": [object()], "error": [object()]},
        )

        MediaPublisher._log_images(images, [first, second], "validation/output_static")

        expected = [
            {
                "key": "validation/output_static/comparison",
                "images": images["comparison"],
            },
            {"key": "validation/output_static/error", "images": images["error"]},
        ]
        assert [c.kwargs for c in first.log_image.call_args_list] == expected
        assert [c.kwargs for c in second.log_image.call_args_list] == expected

    def test_log_videos_rewinds_for_each_logger_and_preserves_format(self) -> None:
        """Rewind shared buffers before every logger handoff."""
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(video_format="mp4"),
        )
        first = MagicMock()
        second = MagicMock()
        buffer = BytesIO(b"video")
        buffer.seek(3)

        media_publisher._log_videos(
            {"forecast": buffer},
            [first, second],
            "test/output_video",
        )

        first.log_video.assert_called_once_with(
            key="test/output_video/forecast", videos=[buffer], format=["mp4"]
        )
        second.log_video.assert_called_once_with(
            key="test/output_video/forecast", videos=[buffer], format=["mp4"]
        )
        assert buffer.tell() == 0


class TestMetadataAndHemisphere:
    def test_metadata_subtitle_reflects_constructor_dataset(self) -> None:
        """Metadata built from the constructor's dataset/epoch/model appears in the footer."""
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
            model_name="unet",
            trained_epochs=50,
        )

        footer = media_publisher.panel_renderer.annotator.footer()

        assert "Model: unet" in footer
        assert "50 epoch(s), 10 samples/epoch)" in footer

    def test_plot_spec_hemisphere_is_used_as_given(self) -> None:
        """Hemisphere is read straight from the given plot_spec, not set separately."""
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(hemisphere=Hemisphere.SOUTH),
        )

        assert media_publisher.panel_renderer.plot_spec.hemisphere == "south"

    def test_land_mask_kwarg_is_used_by_the_renderer(self) -> None:
        """The given land_mask is passed straight through to the panel renderer."""
        new_land_mask = LandMask(None)

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            plot_spec=PlotSpec(),
            land_mask=new_land_mask,
        )

        assert media_publisher.panel_renderer.land_mask is new_land_mask

    def test_trained_epochs_and_model_name_are_optional(self) -> None:
        """Omitting trained_epochs/model_name leaves the footer without those lines."""
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )

        footer = media_publisher.panel_renderer.annotator.footer()

        assert "Model:" not in footer
        assert "epoch(s)" not in footer
        # The dataset still yields a sample count with no epoch count alongside it.
        assert "(10 samples/epoch)" in footer


class TestLogStaticInputs:
    def test_logs_images_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Render and log one image per variable, under the input_static prefix."""
        image = object()
        fake_render = MagicMock(return_value=image)
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_static_inputs(
            [fake_single_dataset()], FORECAST_DATES, [image_logger], prefix="validation"
        )

        assert fake_render.call_count == N_CHANNELS
        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "validation/input_static/2020-01-01-example:ice_conc",
            "validation/input_static/2020-01-01-example:temperature",
        ]
        assert all(
            c.kwargs["images"] == [image] for c in image_logger.log_image.call_args_list
        )

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An InvalidArrayError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_inputs(
                [fake_single_dataset()], FORECAST_DATES, [MagicMock()]
            )

        assert "Image logging failed: bad array" in caplog.text

    def test_skips_on_generic_plotting_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ValueError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_inputs(
                [fake_single_dataset()], FORECAST_DATES, [MagicMock()]
            )

        assert "Image logging failed: bad shape" in caplog.text


class TestLogStaticOutputs:
    def test_logs_images_per_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Render and log one image per output channel, keyed by date and variable name."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_static_outputs(
            make_model_step_output(),
            [image_logger],
            channel_names=["sic"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
        )

        assert fake_render.call_count == N_CHANNELS
        assert image_logger.log_image.call_count == N_CHANNELS
        # Second channel has no configured name, so it falls back to channel_1.
        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "output_static/2020-01-01-sic-truth-vs-prediction",
            "output_static/2020-01-01-channel_1-truth-vs-prediction",
        ]

    def test_climatology_panel_is_labelled_climatology(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The climatology/prediction/difference image titles its first panel Climatology."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        climatology = np.zeros(
            (N_TIMESTEPS, N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_static_outputs(
            make_model_step_output(),
            [MagicMock()],
            channel_names=["sic"],
            climatology=climatology,
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
        )

        # Second render call for channel 0 is the climatology/prediction/difference image.
        climatology_call = fake_render.call_args_list[1]
        panel_titles = climatology_call.kwargs["panel_titles"]
        assert panel_titles[0] == "Climatology"
        assert panel_titles[1] == "Prediction"

    def test_climatology_triplet_omitted_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without compare_truth_climatology, don't log a truth/climatology image."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()
        climatology = np.zeros((N_TIMESTEPS, 1, HEIGHT, WIDTH), dtype=np.float32)

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_static_outputs(
            make_model_step_output(channels=1),
            [image_logger],
            channel_names=["sic"],
            climatology=climatology,
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
        )

        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "output_static/2020-01-01-sic-truth-vs-prediction",
            "output_static/2020-01-01-sic-climatology-vs-prediction",
        ]

    def test_compare_truth_climatology_adds_truth_vs_climatology_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting compare_truth_climatology also logs a ground-truth/climatology image."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()
        climatology = np.zeros((N_TIMESTEPS, 1, HEIGHT, WIDTH), dtype=np.float32)

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_static_outputs(
            make_model_step_output(channels=1),
            [image_logger],
            channel_names=["sic"],
            climatology=climatology,
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            compare_truth_climatology=True,
        )

        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "output_static/2020-01-01-sic-truth-vs-prediction",
            "output_static/2020-01-01-sic-climatology-vs-prediction",
            "output_static/2020-01-01-sic-truth-vs-climatology",
        ]
        # The truth-vs-climatology render omits history_ctx, so it falls back to a
        # plain date title with no footer, rather than the leadtime/history line.
        triplet_call = fake_render.call_args_list[-1]
        assert triplet_call.kwargs["footer_text"] is None
        assert "Leadtime" not in triplet_call.kwargs["figure_title"]

    def test_includes_uncertainty_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Log an extra standardised-difference image for channels with uncertainty data."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()
        uncertainties = {0: torch.zeros((N_TIMESTEPS, HEIGHT, WIDTH)).numpy()}

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_static_outputs(
            make_model_step_output(),
            [image_logger],
            channel_names=["sic", "temperature"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            uncertainties=uncertainties,
        )

        # Channel 0 (has uncertainty) renders twice: difference and z-score.
        # Channel 1 (no uncertainty) renders once: difference only.
        assert fake_render.call_count == N_CHANNELS + 1
        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "output_static/2020-01-01-sic-truth-vs-prediction",
            "output_static/2020-01-01-sic-truth-vs-prediction-z-score",
            "output_static/2020-01-01-temperature-truth-vs-prediction",
        ]

        # The z-score render's extra panel is titled distinctly from a plain
        # difference panel's.
        z_score_call = fake_render.call_args_list[1]
        assert z_score_call.kwargs["panel_titles"][-1] == "Standardised Difference (z)"
        for difference_call in (
            fake_render.call_args_list[0],
            fake_render.call_args_list[2],
        ):
            assert difference_call.kwargs["panel_titles"][-1] == "Difference (signed)"

    def test_skips_z_score_when_difference_panel_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Don't log a z-score image when include_difference is off.

        PanelRenderer only draws a third (z-score) panel when include_difference
        is set, so logging a "z-score" image while it's off would just be a
        mislabeled duplicate of the plain truth/prediction render.
        """
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()
        uncertainties = {0: torch.zeros((N_TIMESTEPS, HEIGHT, WIDTH)).numpy()}

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(include_difference=False),
        )
        media_publisher.log_static_outputs(
            make_model_step_output(),
            [image_logger],
            channel_names=["sic", "temperature"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            uncertainties=uncertainties,
        )

        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "output_static/2020-01-01-sic-truth-vs-prediction",
            "output_static/2020-01-01-temperature-truth-vs-prediction",
        ]

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An InvalidArrayError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_outputs(
                make_model_step_output(),
                [MagicMock()],
                channel_names=["sic"],
                forecast_dates=FORECAST_DATES,
                history_dates=HISTORY_DATES,
            )

        assert "Image logging failed: bad array" in caplog.text

    def test_skips_on_generic_plotting_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A MemoryError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=MemoryError),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_outputs(
                make_model_step_output(),
                [MagicMock()],
                channel_names=["sic"],
                forecast_dates=FORECAST_DATES,
                history_dates=HISTORY_DATES,
            )

        assert "Image logging failed:" in caplog.text

    def test_preserves_prefix_and_channel_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static routing keeps prefixes and fallback channel names stable."""
        image = object()
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            MagicMock(return_value=image),
        )
        image_logger = MagicMock()
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(selected_timestep=1),
        )

        media_publisher.log_static_outputs(
            make_model_step_output(),
            [image_logger],
            channel_names=["ice_conc"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            prefix="evaluate",
        )

        assert [c.kwargs for c in image_logger.log_image.call_args_list] == [
            {
                "key": "evaluate/output_static/2020-01-02-ice_conc-truth-vs-prediction",
                "images": [image],
            },
            {
                "key": "evaluate/output_static/2020-01-02-channel_1-truth-vs-prediction",
                "images": [image],
            },
        ]

    def test_without_prefix_uses_default_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static routing keeps the established default logging namespace."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            MagicMock(return_value=object()),
        )
        image_logger = MagicMock()

        MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        ).log_static_outputs(
            make_model_step_output(channels=1),
            [image_logger],
            channel_names=["ice_conc"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
        )

        assert (
            image_logger.log_image.call_args.kwargs["key"]
            == "output_static/2020-01-01-ice_conc-truth-vs-prediction"
        )


class TestLogVideoInputs:
    def test_logs_videos_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one video per variable, under the input_video prefix."""
        buffer = MagicMock()
        fake_render = MagicMock(return_value=buffer)
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_video_inputs(
            [fake_single_dataset()], FORECAST_DATES, [video_logger], prefix="validation"
        )

        assert fake_render.call_count == N_CHANNELS
        assert video_logger.log_video.call_args_list == [
            call(
                key="validation/input_video/2020-01-01-example:ice_conc",
                videos=[buffer],
                format=[media_publisher.panel_renderer.video_format],
            ),
            call(
                key="validation/input_video/2020-01-01-example:temperature",
                videos=[buffer],
                format=[media_publisher.panel_renderer.video_format],
            ),
        ]

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An InvalidArrayError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_inputs(
                [fake_single_dataset()], FORECAST_DATES, [MagicMock()]
            )

        assert "Video logging failed: bad array" in caplog.text

    def test_skips_on_video_render_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A VideoRenderError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_inputs(
                [fake_single_dataset()], FORECAST_DATES, [MagicMock()]
            )

        assert "Video logging failed: encoding failed" in caplog.text

    def test_skips_on_generic_plotting_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ValueError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_inputs(
                [fake_single_dataset()], FORECAST_DATES, [MagicMock()]
            )

        assert "Video logging failed: bad shape" in caplog.text


class TestLogVideoOutputs:
    def test_logs_videos_per_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Render and log one video per output channel, keyed by date and variable name."""
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_video_outputs(
            make_model_step_output(),
            [video_logger],
            channel_names=["sic"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
        )

        assert fake_render.call_count == N_CHANNELS
        assert video_logger.log_video.call_count == N_CHANNELS
        # Second channel has no configured name, so it falls back to channel_1.
        logged_keys = [c.kwargs["key"] for c in video_logger.log_video.call_args_list]
        assert logged_keys == [
            "output_video/2020-01-01-sic-truth-vs-prediction",
            "output_video/2020-01-01-channel_1-truth-vs-prediction",
        ]

    def test_climatology_panel_is_labelled_climatology(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The climatology/prediction/difference video titles its first panel Climatology."""
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        climatology = np.zeros(
            (N_TIMESTEPS, N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_video_outputs(
            make_model_step_output(),
            [MagicMock()],
            channel_names=["sic"],
            climatology=climatology,
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
        )

        # Second render call for channel 0 is the climatology/prediction/difference video.
        climatology_call = fake_render.call_args_list[1]
        panel_titles = climatology_call.kwargs["panel_titles"]
        assert panel_titles[0] == "Climatology"
        assert panel_titles[1] == "Prediction"

    def test_climatology_triplet_omitted_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without compare_truth_climatology, don't log a truth/climatology video."""
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()
        climatology = np.zeros((N_TIMESTEPS, 1, HEIGHT, WIDTH), dtype=np.float32)

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_video_outputs(
            make_model_step_output(channels=1),
            [video_logger],
            channel_names=["sic"],
            climatology=climatology,
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
        )

        logged_keys = [c.kwargs["key"] for c in video_logger.log_video.call_args_list]
        assert logged_keys == [
            "output_video/2020-01-01-sic-truth-vs-prediction",
            "output_video/2020-01-01-sic-climatology-vs-prediction",
        ]

    def test_compare_truth_climatology_adds_truth_vs_climatology_video(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting compare_truth_climatology also logs a ground-truth/climatology video."""
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()
        climatology = np.zeros((N_TIMESTEPS, 1, HEIGHT, WIDTH), dtype=np.float32)

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_video_outputs(
            make_model_step_output(channels=1),
            [video_logger],
            channel_names=["sic"],
            climatology=climatology,
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            compare_truth_climatology=True,
        )

        logged_keys = [c.kwargs["key"] for c in video_logger.log_video.call_args_list]
        assert logged_keys == [
            "output_video/2020-01-01-sic-truth-vs-prediction",
            "output_video/2020-01-01-sic-climatology-vs-prediction",
            "output_video/2020-01-01-sic-truth-vs-climatology",
        ]
        # The truth-vs-climatology render omits history_ctx, so it falls back to a
        # plain date title with no footer, rather than the leadtime/history line.
        triplet_call = fake_render.call_args_list[-1]
        assert triplet_call.kwargs["footer_text"] is None
        title_for_frame = triplet_call.kwargs["figure_title"]
        assert "Leadtime" not in title_for_frame(0)

    def test_includes_uncertainty_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Log an extra standardised-difference video for channels with uncertainty data."""
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()
        uncertainties = {0: torch.zeros((N_TIMESTEPS, HEIGHT, WIDTH)).numpy()}

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_video_outputs(
            make_model_step_output(),
            [video_logger],
            channel_names=["sic", "temperature"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            uncertainties=uncertainties,
        )

        # Channel 0 (has uncertainty) renders twice: difference and z-score.
        # Channel 1 (no uncertainty) renders once: difference only.
        assert fake_render.call_count == N_CHANNELS + 1
        logged_keys = [c.kwargs["key"] for c in video_logger.log_video.call_args_list]
        assert logged_keys == [
            "output_video/2020-01-01-sic-truth-vs-prediction",
            "output_video/2020-01-01-sic-truth-vs-prediction-z-score",
            "output_video/2020-01-01-temperature-truth-vs-prediction",
        ]

        # The z-score render's extra panel is titled distinctly from a plain
        # difference panel's.
        z_score_call = fake_render.call_args_list[1]
        assert z_score_call.kwargs["panel_titles"][-1] == "Standardised Difference (z)"
        for difference_call in (
            fake_render.call_args_list[0],
            fake_render.call_args_list[2],
        ):
            assert difference_call.kwargs["panel_titles"][-1] == "Difference (signed)"

    def test_skips_z_score_when_difference_panel_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Don't log a z-score video when include_difference is off.

        PanelRenderer only draws a third (z-score) panel when include_difference
        is set, so logging a "z-score" video while it's off would just be a
        mislabeled duplicate of the plain truth/prediction render.
        """
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()
        uncertainties = {0: torch.zeros((N_TIMESTEPS, HEIGHT, WIDTH)).numpy()}

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(include_difference=False),
        )
        media_publisher.log_video_outputs(
            make_model_step_output(),
            [video_logger],
            channel_names=["sic", "temperature"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            uncertainties=uncertainties,
        )

        logged_keys = [c.kwargs["key"] for c in video_logger.log_video.call_args_list]
        assert logged_keys == [
            "output_video/2020-01-01-sic-truth-vs-prediction",
            "output_video/2020-01-01-temperature-truth-vs-prediction",
        ]

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An InvalidArrayError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_outputs(
                make_model_step_output(),
                [MagicMock()],
                channel_names=["sic"],
                forecast_dates=FORECAST_DATES,
                history_dates=HISTORY_DATES,
            )

        assert "Video logging failed: bad array" in caplog.text

    def test_skips_on_video_render_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A VideoRenderError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_outputs(
                make_model_step_output(),
                [MagicMock()],
                channel_names=["sic"],
                forecast_dates=FORECAST_DATES,
                history_dates=HISTORY_DATES,
            )

        assert "Video logging failed: encoding failed" in caplog.text

    def test_skips_on_generic_plotting_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ValueError is consumed and logged at WARNING level."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_outputs(
                make_model_step_output(),
                [MagicMock()],
                channel_names=["sic"],
                forecast_dates=FORECAST_DATES,
                history_dates=HISTORY_DATES,
            )

        assert "Video logging failed" in caplog.text

    def test_rewinds_buffers_before_logging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Video routing rewinds rendered buffers and forwards the format."""
        buffer = BytesIO(b"video-bytes")
        buffer.seek(5)

        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            lambda *args, **kwargs: buffer,  # noqa: ARG005
        )
        video_logger = MagicMock()
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(video_format="gif"),
        )

        media_publisher.log_video_outputs(
            make_model_step_output(channels=1),
            [video_logger],
            channel_names=["ice_conc"],
            forecast_dates=FORECAST_DATES,
            history_dates=HISTORY_DATES,
            prefix="test",
        )

        video_logger.log_video.assert_called_once_with(
            key="test/output_video/2020-01-01-ice_conc-truth-vs-prediction",
            videos=[buffer],
            format=["gif"],
        )
        assert buffer.tell() == 0
