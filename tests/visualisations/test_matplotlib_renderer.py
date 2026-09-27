"""Tests for the minimal panel-plot/panel-video rendering core."""

import logging
from io import BytesIO
from unittest.mock import MagicMock

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import VideoRenderError
from icenet_mp.types import ArrayHW, ArrayTHW
from icenet_mp.visualisations.matplotlib_renderer import MatplotlibRenderer

renderer = MatplotlibRenderer()


def make_figure() -> Figure:
    """Build a small real matplotlib figure for conversion tests."""
    fig, ax = plt.subplots()
    ax.imshow(np.random.default_rng(0).random((8, 8)))
    return fig


def make_animation(fig: Figure) -> animation.FuncAnimation:
    """Build a minimal real FuncAnimation with a trivial per-frame update."""
    ax = fig.axes[0]
    image = ax.imshow(np.zeros((8, 8)))

    def animate(frame: int) -> tuple[()]:
        image.set_data(np.full((8, 8), frame))
        return ()

    return animation.FuncAnimation(fig, animate, frames=2, interval=200, blit=False)


class TestRenderPanels:
    def test_single_panel(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static([era5_temperature_2d])

        assert isinstance(result, ImageFile)
        assert result.width > 0
        assert result.height > 0

    def test_three_panels_wider_than_one(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        single = renderer.panels_static([era5_temperature_2d])
        triple = renderer.panels_static(
            [era5_temperature_2d, osisaf_ice_conc_2d, osisaf_ice_conc_2d]
        )

        assert triple.width > single.width

    def test_titles_applied(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static(
            [era5_temperature_2d], panel_titles=["Ground Truth"]
        )

        assert isinstance(result, ImageFile)

    def test_shared_vmin_vmax(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static([era5_temperature_2d], vmin=260.0, vmax=290.0)

        assert isinstance(result, ImageFile)

    def test_mismatched_titles_length_raises(
        self, era5_temperature_2d: ArrayHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            renderer.panels_static([era5_temperature_2d], panel_titles=["a", "b"])

    def test_mismatched_cmap_length_raises(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        """A cmap list not matching the panel count raises, not silently mis-pairs."""
        with pytest.raises(ValueError, match=r"zip\(\)"):
            renderer.panels_static(
                [era5_temperature_2d, osisaf_ice_conc_2d, osisaf_ice_conc_2d],
                cmap=["viridis", "magma"],
            )

    def test_per_panel_cmap_and_scale(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        result = renderer.panels_static(
            [era5_temperature_2d, osisaf_ice_conc_2d],
            cmap=["RdBu_r", "Blues_r"],
            vmin=[260.0, 0.0],
            vmax=[290.0, 1.0],
        )

        assert isinstance(result, ImageFile)

    def test_suptitle(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static(
            [era5_temperature_2d], figure_title="Shown: 2020-01-15"
        )

        assert isinstance(result, ImageFile)

    def test_contour_drawn_on_selected_panel(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        """A contour_arrays entry draws a contour; a None entry draws none."""
        result = renderer.panels_static(
            [era5_temperature_2d, osisaf_ice_conc_2d],
            contour_arrays=[None, osisaf_ice_conc_2d],
            contour_level=0.15,
        )

        assert isinstance(result, ImageFile)

    def test_no_contour_without_level(self, osisaf_ice_conc_2d: ArrayHW) -> None:
        """contour_arrays without a contour_level draws nothing, not an error."""
        result = renderer.panels_static(
            [osisaf_ice_conc_2d], contour_arrays=[osisaf_ice_conc_2d]
        )

        assert isinstance(result, ImageFile)

    def test_contour_mismatched_length_raises(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            renderer.panels_static(
                [era5_temperature_2d, osisaf_ice_conc_2d],
                contour_arrays=[osisaf_ice_conc_2d],
                contour_level=0.15,
            )


class TestCmapWithBad:
    def test_named_cmap_is_used(self) -> None:
        """A named colourmap is looked up and returned by that name."""
        cmap = MatplotlibRenderer()._cmap_with_bad("magma")

        assert cmap.name == "magma"

    def test_bad_colour_is_configured(self) -> None:
        """The bad (NaN) colour is set to the renderer's configured colour."""
        cmap = MatplotlibRenderer(bad_colour="#ff00ff")._cmap_with_bad("viridis")

        np.testing.assert_allclose(cmap.get_bad(), to_rgba("#ff00ff"))

    def test_default_bad_colour(self) -> None:
        """With no bad_colour override, the light grey default is used."""
        cmap = MatplotlibRenderer()._cmap_with_bad("viridis")

        np.testing.assert_allclose(cmap.get_bad(), to_rgba("#dcdcdc"))

    def test_named_cmap_is_cached(self) -> None:
        """A second lookup by the same name returns the identical cached object."""
        local_renderer = MatplotlibRenderer()

        first = local_renderer._cmap_with_bad("viridis")
        second = local_renderer._cmap_with_bad("viridis")

        assert first is second


class TestCmapBadColourConfigured:
    def test_string_cmap_gets_opaque_bad_colour(
        self, era5_temperature_2d: ArrayHW
    ) -> None:
        """A string cmap name is copied and given a non-transparent bad colour.

        LandMask.apply_to represents masked land as NaN; imshow's default bad
        colour is transparent, which would let the axes background show
        through instead of a consistent land colour.
        """
        _, axes = renderer.panels([era5_temperature_2d], cmap="viridis")

        bad_colour = axes[0].images[0].get_cmap().get_bad()
        assert bad_colour[3] == pytest.approx(1.0)

    def test_named_cmap_unaffected_by_shared_default_instance(
        self, era5_temperature_2d: ArrayHW
    ) -> None:
        """Configuring the bad colour copies the colourmap, not the registered one."""
        renderer.panels([era5_temperature_2d], cmap="viridis")

        assert mpl.colormaps.get_cmap("viridis").get_bad()[3] == pytest.approx(0.0)


class TestRenderPanelsVideo:
    def test_single_panel(self, era5_temperature_thw: ArrayTHW) -> None:
        result = renderer.panels_video([era5_temperature_thw])

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_three_panels(self, era5_temperature_thw: ArrayTHW) -> None:
        result = renderer.panels_video(
            [era5_temperature_thw, era5_temperature_thw, era5_temperature_thw]
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_mismatched_titles_length_raises(
        self, era5_temperature_thw: ArrayTHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            renderer.panels_video([era5_temperature_thw], panel_titles=["a", "b"])

    def test_fps_is_configurable(self, era5_temperature_thw: ArrayTHW) -> None:
        result = renderer.panels_video([era5_temperature_thw], fps=4)

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_contour_redrawn_across_frames(
        self, era5_temperature_thw: ArrayTHW
    ) -> None:
        """A per-frame contour array renders without error across all frames."""
        result = renderer.panels_video(
            [era5_temperature_thw],
            contour_arrays=[era5_temperature_thw],
            contour_level=273.15,
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_contour_none_entry_skipped(self, era5_temperature_thw: ArrayTHW) -> None:
        """A None contour_arrays entry draws no contour on that panel."""
        result = renderer.panels_video(
            [era5_temperature_thw, era5_temperature_thw],
            contour_arrays=[None, era5_temperature_thw],
            contour_level=273.15,
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_callable_figure_title_updates_suptitle_per_frame(
        self, era5_temperature_thw: ArrayTHW, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A callable figure_title is re-invoked per frame and redraws the suptitle.

        Regression test: the animation callback used to update only the image
        and contour artists, while the figure title stayed a plain string
        computed once from frame 0 -- so a video's on-screen title never
        changed even as the panels animated over later dates.
        """
        captured: dict[str, object] = {}

        def fake_video_from_animation(
            _self: MatplotlibRenderer,
            anim: animation.FuncAnimation,
            **_kwargs: object,
        ) -> BytesIO:
            # FuncAnimation only exposes the figure and per-frame callback it was
            # built with via its private attributes.
            captured["figure"] = getattr(anim, "_fig")  # noqa: B009
            captured["func"] = getattr(anim, "_func")  # noqa: B009
            return BytesIO(b"fake")

        monkeypatch.setattr(
            MatplotlibRenderer, "_video_from_animation", fake_video_from_animation
        )

        renderer.panels_video(
            [era5_temperature_thw], figure_title=lambda tt: f"Frame: {tt}"
        )

        figure = captured["figure"]
        func = captured["func"]
        assert isinstance(figure, Figure)
        assert callable(func)
        assert figure.get_suptitle() == "Frame: 0"

        func(1)

        assert figure.get_suptitle() == "Frame: 1"


class TestImageFromFigure:
    def test_returns_image_with_positive_dimensions(self) -> None:
        """A real figure converts to a PIL image with a non-trivial size."""
        fig = make_figure()

        image = renderer._image_from_figure(fig, dpi=100)

        assert isinstance(image, ImageFile)
        assert image.width > 0
        assert image.height > 0


class TestVideoFromAnimation:
    def test_gif_output_has_gif_signature(self) -> None:
        """A GIF-format render returns a non-empty BytesIO starting with the GIF header."""
        fig = make_figure()
        anim = make_animation(fig)

        buffer = renderer._video_from_animation(
            anim, dpi=100, fps=2, video_format="gif"
        )

        assert isinstance(buffer, BytesIO)
        content = buffer.read()
        assert len(content) > 0
        assert content[:6] == b"GIF89a"

    def test_mp4_output_has_mp4_signature(self) -> None:
        """An MP4-format render returns a non-empty BytesIO containing an ftyp box."""
        fig = make_figure()
        anim = make_animation(fig)

        buffer = renderer._video_from_animation(
            anim, dpi=100, fps=2, video_format="mp4"
        )

        assert isinstance(buffer, BytesIO)
        content = buffer.read()
        assert len(content) > 0
        assert b"ftyp" in content[:100]

    def test_save_failure_raises_video_render_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An OSError from anim.save is wrapped and re-raised as VideoRenderError."""
        fig = make_figure()
        anim = make_animation(fig)
        monkeypatch.setattr(anim, "save", MagicMock(side_effect=OSError("disk full")))

        with pytest.raises(VideoRenderError, match="Video encoding failed"):
            renderer._video_from_animation(anim, dpi=100, fps=2, video_format="gif")

    def test_save_memory_error_raises_video_render_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A MemoryError from anim.save is wrapped and re-raised as VideoRenderError."""
        fig = make_figure()
        anim = make_animation(fig)
        monkeypatch.setattr(
            anim, "save", MagicMock(side_effect=MemoryError("out of memory"))
        )

        with pytest.raises(VideoRenderError, match="Video encoding failed"):
            renderer._video_from_animation(anim, dpi=100, fps=2, video_format="mp4")


class TestSuppressMplAnimationLogs:
    def test_sets_warning_level_and_restores_original(self) -> None:
        """The matplotlib.animation logger is raised to WARNING and restored after."""
        mpl_logger = logging.getLogger("matplotlib.animation")
        mpl_logger.setLevel(logging.INFO)

        with renderer._suppress_mpl_animation_logs():
            assert mpl_logger.level == logging.WARNING

        assert mpl_logger.level == logging.INFO
