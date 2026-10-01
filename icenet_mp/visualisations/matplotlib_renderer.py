import contextlib
import gc
import logging
import tempfile
from collections.abc import Callable, Generator, Sequence
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from PIL import Image
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import VideoRenderError
from icenet_mp.types import ArrayHW, ArrayTHW


class MatplotlibRenderer:
    """Minimal matplotlib rendering of figures and videos from raw arrays."""

    def __init__(
        self,
        *,
        bad_colour: str = "#dcdcdc",
        colourbar_aspect: float = 25,
        colourbar_label_size: float = 9,
        contour_linewidth: float = 1.2,
        panel_height_in: float = 6,
    ) -> None:
        """Initialise a MatplotlibRenderer with an empty named-colourmap cache.

        Args:
            bad_colour: Colour used for NaN/bad values in a colourmap that doesn't
                already have one configured, preventing transparent/white artifacts.
            colourbar_aspect: Long:short-axis ratio of a single-panel colourbar.
            colourbar_label_size: Font size of colourbar tick labels.
            contour_linewidth: Line width of drawn contours (e.g. the sea ice edge).
            panel_height_in: Height in inches of each panel (width scales with it).

        """
        self._bad_colour = bad_colour
        self._cmap_cache: dict[str, Colormap] = {}
        self._colourbar_aspect = colourbar_aspect
        self._colourbar_label_size = colourbar_label_size
        self._contour_linewidth = contour_linewidth
        self._panel_height_in = panel_height_in

    def _cmap_with_bad(self, name: str) -> Colormap:
        """Get a Colormap for NaN values, defaulting to `self._bad_colour`.

        `self._bad_colour` is only applied if the colormap's own bad colour is
        fully transparent; a colormap that already ships a non-transparent bad
        colour keeps it. A named colour map is cached after its first build for
        re-use.
        """
        # If the requested name is not cached then build a new Colormap and cache it.
        if (cmap := self._cmap_cache.get(name)) is None:
            # Load the corresponding matplotlib colormap
            cmap = colormaps.get_cmap(name)

            # Apply the default bad colour if the current one is transparent (alpha=0)
            if cmap.get_bad()[-1] == 0:
                cmap.set_bad(self._bad_colour)

            # Cache the modified colormap for future use
            self._cmap_cache[name] = cmap

        return cmap

    @contextlib.contextmanager
    def _suppress_mpl_animation_logs(self) -> Generator[None]:
        """Temporarily suppress matplotlib animation INFO log messages."""
        mpl_logger = logging.getLogger("matplotlib.animation")
        original_level = mpl_logger.level
        try:
            mpl_logger.setLevel(logging.WARNING)
            yield
        finally:
            mpl_logger.setLevel(original_level)

    def _image_from_figure(self, figure: Figure, *, dpi: int) -> ImageFile:
        """Convert a matplotlib figure to a PIL image file."""
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        return Image.open(buffer)

    def _video_from_animation(
        self,
        anim: animation.FuncAnimation,
        *,
        dpi: int = 200,
        fps: int = 2,
        video_format: Literal["mp4", "gif"] = "gif",
    ) -> BytesIO:
        """Save an animation to a temporary file and return BytesIO (with cleanup)."""
        suffix = ".gif" if video_format.lower() == "gif" else ".mp4"

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                try:
                    writer = (
                        animation.PillowWriter(fps=fps)
                        if suffix == ".gif"
                        else animation.FFMpegWriter(
                            fps=fps,
                            codec="libx264",
                            bitrate=1800,
                            # Ensure dimensions are compatible with yuv420p (even width/height)
                            # by applying a scale filter that truncates to the nearest even integers.
                            extra_args=[
                                "-pix_fmt",
                                "yuv420p",
                                "-vf",
                                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            ],
                        )
                    )
                    # anim.save's own writer.saving() context manager calls
                    # writer.finish() for us on both success and failure, so no
                    # separate writer cleanup is needed here.
                    # Suppress matplotlib's INFO log message about writer selection
                    with self._suppress_mpl_animation_logs():
                        anim.save(tmp.name, writer=writer, dpi=dpi)
                    # Load tempfile into a BytesIO buffer
                    with Path(tmp.name).open("rb") as fh:
                        buffer = BytesIO(fh.read())
                except (OSError, MemoryError) as err:
                    msg = f"Video encoding failed: {err!s}"
                    raise VideoRenderError(msg) from err
        finally:
            # Force garbage collection to clean up any remaining resources
            gc.collect()

        buffer.seek(0)
        return buffer

    def _add_colourbar(
        self, fig: Figure, image: AxesImage, axes: Axes | list[Axes]
    ) -> None:
        """Add a horizontal colourbar sized/labelled consistently regardless of span.

        `aspect` (long:short-axis ratio) is scaled by how many panels the colourbar
        spans, so a colourbar shared across N panels comes out the same thickness
        as one attached to a single panel, not N times thicker.
        """
        span = len(axes) if isinstance(axes, list) else 1
        cbar = fig.colorbar(
            image,
            ax=axes,
            orientation="horizontal",
            aspect=self._colourbar_aspect * span,
            pad=0.04,
        )
        cbar.ax.tick_params(labelsize=self._colourbar_label_size)

    def _draw_contours(
        self,
        axes: Sequence[Axes],
        contour_arrays: Sequence[np.ndarray | None] | None,
        *,
        color: str,
        level: float | None,
    ) -> list[QuadContourSet | None]:
        """Draw a single-level contour on each panel that has a contour array.

        Returns one `QuadContourSet` (or `None`) per axes, in panel order, so a
        caller can `.remove()` it before redrawing (e.g. per animation frame).
        """
        if contour_arrays is None or level is None:
            return [None] * len(axes)
        contour_sets: list[QuadContourSet | None] = []
        for ax, contour_arr in zip(axes, contour_arrays, strict=True):
            if contour_arr is None:
                contour_sets.append(None)
                continue
            contour_sets.append(
                ax.contour(
                    contour_arr,
                    colors=color,
                    levels=[level],
                    linewidths=self._contour_linewidth,
                )
            )
        return contour_sets

    def panels(  # noqa: PLR0913
        self,
        arrays: Sequence[ArrayHW],
        *,
        cmap: str | Sequence[str] = "viridis",
        contour_arrays: Sequence[ArrayHW | None] | None = None,
        contour_color: str = "red",
        contour_level: float | None = None,
        figure_title: str | None = None,
        footer_text: str | None = None,
        group_axes: tuple[int, int] | None = None,
        panel_titles: Sequence[str] | None = None,
        vmax: float | Sequence[float | None] | None = None,
        vmin: float | Sequence[float | None] | None = None,
    ) -> tuple[Figure, list[Axes]]:
        """Render multiple panels side-by-side.

        Args:
            arrays: 1 to 3 2D `[H, W]` arrays, one per panel.
            cmap: Optional colourmap(s), either shared by all panels or one per panel.
            contour_arrays: Optional per-panel array to draw a single-level contour
                over (e.g. the sea ice edge); `None` entries draw no contour.
            contour_color: Colour of the contour line(s).
            contour_level: Value at which to draw the contour; no contours are
                drawn if `None`.
            figure_title: Optional figure-level title drawn above all panels.
            footer_text: Optional footer text drawn below the colourbars.
            group_axes: Optional inclusive `(start, end)` panel index range to share a colourbar.
            panel_titles: Optional per-panel title, one per panel.
            vmax: Optional upper colour-scale bound(s), either shared or one per panel.
            vmin: Optional lower colour-scale bound(s), either shared or one per panel.

        Returns:
            The matplotlib Figure, and the list of panel Axes in panel order (each
            Axes' drawn image is available as `ax.images[0]`, e.g. for animation).

        """
        # Create a figure with `n` subplots, each with the same height and an aspect ratio
        n = len(arrays)
        fig, axes_ = plt.subplots(
            1,
            n,
            figsize=(self._panel_height_in * n, self._panel_height_in),
            layout="compressed",
        )
        axes: list[Axes] = np.atleast_1d(axes_).tolist()
        for ax, array in zip(axes, arrays, strict=True):
            ax.set_box_aspect(array.shape[0] / array.shape[1])

        _cmaps = (
            self._cmap_with_bad(name)
            for name in ([cmap] * n if isinstance(cmap, str) else list(cmap))
        )
        _ranges = zip(
            list(vmin) if isinstance(vmin, Sequence) else [vmin] * n,
            list(vmax) if isinstance(vmax, Sequence) else [vmax] * n,
            strict=True,
        )

        # Render each panel's array with its corresponding colourmap and normalisation
        images = [
            ax.imshow(array, cmap=_cmap, vmin=_vmin, vmax=_vmax, origin="upper")
            for ax, array, _cmap, (_vmin, _vmax) in zip(
                axes, arrays, _cmaps, _ranges, strict=True
            )
        ]

        # Set panel titles and turn off axes ticks/labels
        for ax, title in zip(axes, panel_titles or [""] * n, strict=True):
            ax.set_title(title)
            ax.axis("off")

        # Draw contours on each panel that has a contour array, if requested
        self._draw_contours(
            axes, contour_arrays, color=contour_color, level=contour_level
        )

        # Add colourbars, either one per panel or a single shared one for a range of panels
        if group_axes is not None:
            start, end = group_axes
            self._add_colourbar(fig, images[start], axes[start : end + 1])
            for i, (ax, image) in enumerate(zip(axes, images, strict=True)):
                if i < start or i > end:
                    self._add_colourbar(fig, image, ax)
        else:
            for ax, image in zip(axes, images, strict=True):
                self._add_colourbar(fig, image, ax)

        # Add figure-level title and footer text if provided
        if figure_title:
            fig.suptitle(figure_title)
        if footer_text:
            fig.supxlabel(footer_text)

        return fig, axes

    def panels_static(  # noqa: PLR0913
        self,
        arrays: Sequence[ArrayHW],
        *,
        cmap: str | Sequence[str] = "viridis",
        contour_arrays: Sequence[ArrayHW | None] | None = None,
        contour_color: str = "red",
        contour_level: float | None = None,
        dpi: int = 150,
        figure_title: str | None = None,
        footer_text: str | None = None,
        group_axes: tuple[int, int] | None = None,
        panel_titles: Sequence[str] | None = None,
        vmax: float | Sequence[float | None] | None = None,
        vmin: float | Sequence[float | None] | None = None,
    ) -> ImageFile:
        """Render multiple panels side-by-side.

        Args:
            arrays: 1 to 3 2D `[H, W]` arrays, one per panel.
            cmap: Optional colourmap(s), either shared by all panels or one per panel.
            contour_arrays: Optional per-panel array to draw a single-level contour
                over (e.g. the sea ice edge); `None` entries draw no contour.
            contour_color: Colour of the contour line(s).
            contour_level: Value at which to draw the contour; no contours are
                drawn if `None`.
            dpi: Dots per inch for the rendered image.
            figure_title: Optional figure-level title drawn above all panels.
            footer_text: Optional footer text drawn below the colourbars.
            group_axes: Optional inclusive `(start, end)` panel index range to share a colourbar.
            panel_titles: Optional per-panel title, one per panel.
            vmax: Optional upper colour-scale bound(s), either shared or one per panel.
            vmin: Optional lower colour-scale bound(s), either shared or one per panel.

        Returns:
            A PIL image of the rendered panels.

        """
        figure, _ = self.panels(
            arrays,
            cmap=cmap,
            contour_arrays=contour_arrays,
            contour_color=contour_color,
            contour_level=contour_level,
            figure_title=figure_title,
            footer_text=footer_text,
            group_axes=group_axes,
            panel_titles=panel_titles,
            vmax=vmax,
            vmin=vmin,
        )
        try:
            return self._image_from_figure(figure, dpi=dpi)
        finally:
            plt.close(figure)

    def panels_video(  # noqa: PLR0913
        self,
        arrays: Sequence[ArrayTHW],
        *,
        cmap: str | Sequence[str] = "viridis",
        contour_arrays: Sequence[ArrayTHW | None] | None = None,
        contour_color: str = "red",
        contour_level: float | None = None,
        dpi: int = 150,
        figure_title: str | Callable[[int], str] | None = None,
        footer_text: str | None = None,
        fps: int = 2,
        group_axes: tuple[int, int] | None = None,
        panel_titles: Sequence[str] | None = None,
        vmax: float | Sequence[float | None] | None = None,
        vmin: float | Sequence[float | None] | None = None,
        video_format: Literal["mp4", "gif"] = "mp4",
    ) -> BytesIO:
        """Render multiple panels side-by-side, animated over time.

        Builds the first frame with `panels`, then updates each panel's
        image data (and any contour) per frame.

        Args:
            arrays: 1 to 3 3D `[T, H, W]` arrays, one per panel, animated in lockstep.
            cmap: Optional colourmap(s), either shared by all panels or one per panel.
            contour_arrays: Optional per-panel `[T, H, W]` array to draw a
                single-level contour over per frame (e.g. the sea ice edge);
                `None` entries draw no contour.
            contour_color: Colour of the contour line(s).
            contour_level: Value at which to draw the contour; no contours are
                drawn if `None`.
            dpi: Dots per inch for the rendered video frames.
            figure_title: Optional figure-level title drawn above all panels. Either a
                fixed string, or a callable that maps the frame index to the title of
                that frame (e.g. to show a per-frame date).
            footer_text: Optional footer text drawn below the colourbars.
            fps: Frames per second for the rendered video.
            group_axes: Optional inclusive `(start, end)` panel index range to share a colourbar.
            panel_titles: Optional per-panel titles, one per panel.
            vmax: Optional upper colour-scale bound(s), either shared or one per panel.
            vmin: Optional lower colour-scale bound(s), either shared or one per panel.
            video_format: Encode as "mp4" or "gif".

        Returns:
            A BytesIO buffer containing the encoded video.

        """
        if callable(figure_title):
            title_for_frame: Callable[[int], str] | None = figure_title
            first_title: str | None = figure_title(0)
        else:
            title_for_frame = None
            first_title = figure_title

        figure, axes = self.panels(
            [array[0] for array in arrays],
            cmap=cmap,
            figure_title=first_title,
            footer_text=footer_text,
            group_axes=group_axes,
            panel_titles=panel_titles,
            vmax=vmax,
            vmin=vmin,
        )
        images = [ax.images[0] for ax in axes]

        def _frame_contours(tt: int) -> list[ArrayHW | None]:
            if contour_arrays is None:
                return [None] * len(axes)
            return [None if arr is None else arr[tt] for arr in contour_arrays]

        contour_sets = self._draw_contours(
            axes, _frame_contours(0), color=contour_color, level=contour_level
        )

        def animate(tt: int) -> tuple[()]:
            for image, array in zip(images, arrays, strict=True):
                image.set_data(array[tt])
            if title_for_frame:
                figure.suptitle(title_for_frame(tt))
            frame_contours = _frame_contours(tt)
            for i, contour_set in enumerate(contour_sets):
                if contour_set is not None:
                    contour_set.remove()
                contour_sets[i] = self._draw_contours(
                    [axes[i]],
                    [frame_contours[i]],
                    color=contour_color,
                    level=contour_level,
                )[0]
            return ()

        try:
            anim = animation.FuncAnimation(
                figure, animate, frames=arrays[0].shape[0], interval=1000 // fps
            )
            return self._video_from_animation(
                anim, dpi=dpi, fps=fps, video_format=video_format
            )
        finally:
            plt.close(figure)
