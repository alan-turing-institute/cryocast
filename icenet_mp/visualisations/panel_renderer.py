from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Literal

from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayHW, ArrayTHW, Metadata, PlotSpec, Timespan

from .difference_panel import DifferencePanel
from .land_mask import LandMask
from .matplotlib_renderer import MatplotlibRenderer
from .media_annotator import MediaAnnotator
from .style_resolver import StyleResolver

if TYPE_CHECKING:
    import numpy as np


class PanelRenderer:
    """Renders styled, land-masked panels for one land_mask/plot_spec pairing."""

    VIDEO_NDIM = 3

    def __init__(
        self, land_mask: LandMask, metadata: Metadata, plot_spec: PlotSpec
    ) -> None:
        """Build a PanelRenderer for a given land mask, metadata, and plot spec."""
        self.land_mask = land_mask
        self.plot_spec = plot_spec
        self.annotator = MediaAnnotator(metadata, plot_spec)
        self.renderer = MatplotlibRenderer()
        self.resolver = StyleResolver(plot_spec)

    @property
    def video_format(self) -> Literal["mp4", "gif"]:
        return self.plot_spec.video_format

    def _validate_arrays(
        self, *arrays: ArrayHW | ArrayTHW, dates: Timespan | None = None
    ) -> list[ArrayHW] | list[ArrayTHW]:
        """Validate that arrays share a common shape.

        When `dates` is given (the video case), the first array must additionally
        be 3D [T, H, W] with one frame per date.

        Raises:
            InvalidArrayError: If the arrays' shapes don't all match, or (when
                `dates` is given) the first array isn't 3D with a matching frame
                count.

        """
        shape = arrays[0].shape
        if dates is not None and (
            len(shape) != self.VIDEO_NDIM or shape[0] != dates.steps
        ):
            msg = (
                f"Expected a 3D [T, H, W] array with {dates.steps} frames, got {shape}."
            )
            raise InvalidArrayError(msg)
        # The remaining arrays only need to match the shape of the first array
        for array in arrays[1:]:
            if array.shape != shape:
                msg = f"Array shapes must match; expected {shape}, got {array.shape}."
                raise InvalidArrayError(msg)
        return list(arrays)

    def static_singlet(
        self,
        values: ArrayHW,
        *,
        when: datetime,
        variable_name: str,
    ) -> ImageFile:
        """Render a single panel ImageFile via MatplotlibRenderer.panels_static().

        Args:
            values: 2D array of the variable field to render.
            when: Datetime of the plotted timestep.
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            An ImageFile containing the rendered panel.

        """
        arrays = [self.land_mask.apply_to(values)]
        scale = self.resolver.colour_scale(variable_name)
        title = self.annotator.header_for_variable(
            units=scale.units, when=when, variable_name=variable_name
        )
        return self.renderer.panels_static(
            arrays,
            cmap=scale.cmap,
            dpi=self.plot_spec.dpi,
            figure_title=title,
            vmax=scale.vmax,
            vmin=scale.vmin,
        )

    def static_triplet(  # noqa: PLR0913
        self,
        ground_truth: ArrayHW,
        prediction: ArrayHW,
        *,
        forecast_date: datetime,
        history_ctx: Timespan | None = None,
        panel_titles: dict[str, str] | None = None,
        uncertainty: ArrayHW | None = None,
        variable_name: str,
    ) -> ImageFile:
        """Render a three panel ImageFile via MatplotlibRenderer.panels().

        Args:
            ground_truth: 2D array of the ground truth field.
            prediction: 2D array of the predicted field.
            history_ctx: Timespan covering the history/input period. When omitted,
                the title falls back to a plain "<variable> on <date>" line and the
                model/training/input-dataset footer is skipped; use this for plots
                that don't depend on the model, such as a ground-truth/climatology
                comparison.
            forecast_date: Datetime of the forecast being shown.
            uncertainty: Optional 2D array of the reported standard uncertainty of the
                prediction field. When given, the third panel divides the configured
                difference (see `plot_spec.diff_mode`) by `uncertainty`; this is a
                signed z-score only when `diff_mode` is `DiffMode.SIGNED`.
            panel_titles: Optional overrides for the panel titles, keyed by
                "ground_truth", "prediction" and/or "difference".
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            An ImageFile containing the rendered panels.

        Raises:
            InvalidArrayError: If `ground_truth` and `prediction` don't have
                matching shapes.

        """
        arrays = self._validate_arrays(
            self.land_mask.apply_to(ground_truth),
            self.land_mask.apply_to(prediction),
        )
        masked_ground_truth, masked_prediction = arrays

        panel_titles = panel_titles or {}
        titles = [
            panel_titles.get("ground_truth", self.plot_spec.title_groundtruth),
            panel_titles.get("prediction", self.plot_spec.title_prediction),
        ]

        scale = self.resolver.colour_scale(variable_name)
        cmaps: list[str] = [scale.cmap, scale.cmap]
        vmins: list[float | None] = [scale.vmin, scale.vmin]
        vmaxs: list[float | None] = [scale.vmax, scale.vmax]

        # Optionally add a difference panel
        if self.plot_spec.include_difference:
            diff_panel = DifferencePanel(
                self.plot_spec.diff_mode,
                masked_ground_truth,
                masked_prediction,
                uncertainty,
            )
            arrays.append(diff_panel.difference)
            titles.append(
                panel_titles.get(
                    "difference",
                    f"{self.plot_spec.title_difference} ({self.plot_spec.diff_mode})",
                )
            )
            cmaps.append(diff_panel.colour_scale.cmap)
            vmins.append(diff_panel.colour_scale.vmin)
            vmaxs.append(diff_panel.colour_scale.vmax)

        contour_arrays: list[np.ndarray | None] | None = None
        if self.plot_spec.include_ice_edge:
            contour_arrays = [masked_ground_truth, masked_prediction]
            contour_arrays += [None] * (len(arrays) - len(contour_arrays))

        title = self.annotator.header(
            forecast_date=forecast_date,
            history_ctx=history_ctx,
            variable_name=variable_name,
        )

        return self.renderer.panels_static(
            arrays,
            cmap=cmaps,
            contour_arrays=contour_arrays,
            contour_level=self.plot_spec.ice_edge_threshold,
            dpi=self.plot_spec.dpi,
            figure_title=title,
            footer_text=None if history_ctx is None else self.annotator.footer(),
            group_axes=(0, 1) if self.plot_spec.include_difference else None,
            panel_titles=titles,
            vmax=vmaxs,
            vmin=vmins,
        )

    def video_singlet(
        self,
        values: ArrayTHW,
        *,
        dates: Timespan,
        variable_name: str,
    ) -> BytesIO:
        """Render a single panel video BytesIO via MatplotlibRenderer.panels_video().\

        Args:
            values: 3D array of the variable field to render, with shape (time, height, width).
            dates: Timespan corresponding to each timestep in `values`.
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            A BytesIO object containing the rendered video.

        Raises:
            InvalidArrayError: If `values` isn't 3D, or `dates` doesn't have
                one entry per frame.

        """
        arrays = self._validate_arrays(self.land_mask.apply_to(values), dates=dates)
        scale = self.resolver.colour_scale(variable_name)

        def title_for_frame(frame: int) -> str:
            return self.annotator.header_for_variable(
                units=scale.units, when=dates[frame], variable_name=variable_name
            )

        return self.renderer.panels_video(
            arrays,
            cmap=scale.cmap,
            dpi=self.plot_spec.dpi,
            figure_title=title_for_frame,
            fps=self.plot_spec.video_fps,
            vmax=scale.vmax,
            vmin=scale.vmin,
            video_format=self.video_format,
        )

    def video_triplet(  # noqa: PLR0913
        self,
        ground_truth: ArrayTHW,
        prediction: ArrayTHW,
        *,
        forecast_ctx: Timespan,
        history_ctx: Timespan | None = None,
        panel_titles: dict[str, str] | None = None,
        uncertainty: ArrayTHW | None = None,
        variable_name: str,
    ) -> BytesIO:
        """Render a three-panel video BytesIO via MatplotlibRenderer.panels_video().

        Args:
            ground_truth: 3D array of the ground truth field.
            prediction: 3D array of the predicted field.
            forecast_ctx: Timespan for the forecast period.
            history_ctx: Timespan for the history period. When omitted, each frame's
                title falls back to a plain "<variable> on <date>" line and the
                model/training/input-dataset footer is skipped; use this for plots
                that don't depend on the model, such as a ground-truth/climatology
                comparison.
            uncertainty: Optional 3D array of the reported standard uncertainty of the
                prediction field. When given, the third panel divides the configured
                difference (see `plot_spec.diff_mode`) by `uncertainty`; this is a
                signed z-score only when `diff_mode` is `DiffMode.SIGNED`.
            panel_titles: Optional overrides for the panel titles, keyed by
                "ground_truth", "prediction" and/or "difference".
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            A BytesIO object containing the rendered video.

        Raises:
            InvalidArrayError: If `ground_truth` and `prediction` don't have
                matching shapes, if either isn't 3D, or if `forecast_ctx` doesn't
                have one entry per frame.

        """
        arrays = self._validate_arrays(
            self.land_mask.apply_to(ground_truth),
            self.land_mask.apply_to(prediction),
            dates=forecast_ctx,
        )
        masked_ground_truth, masked_prediction = arrays

        panel_titles = panel_titles or {}
        titles = [
            panel_titles.get("ground_truth", self.plot_spec.title_groundtruth),
            panel_titles.get("prediction", self.plot_spec.title_prediction),
        ]

        scale = self.resolver.colour_scale(variable_name)
        cmaps: list[str] = [scale.cmap, scale.cmap]
        vmins: list[float | None] = [scale.vmin, scale.vmin]
        vmaxs: list[float | None] = [scale.vmax, scale.vmax]

        # Optionally add a difference panel
        if self.plot_spec.include_difference:
            diff_panel = DifferencePanel(
                self.plot_spec.diff_mode,
                masked_ground_truth,
                masked_prediction,
                uncertainty,
            )
            arrays.append(diff_panel.difference)
            titles.append(
                panel_titles.get(
                    "difference",
                    f"{self.plot_spec.title_difference} ({self.plot_spec.diff_mode})",
                )
            )
            cmaps.append(diff_panel.colour_scale.cmap)
            vmins.append(diff_panel.colour_scale.vmin)
            vmaxs.append(diff_panel.colour_scale.vmax)

        contour_arrays: list[np.ndarray | None] | None = None
        if self.plot_spec.include_ice_edge:
            contour_arrays = [masked_ground_truth, masked_prediction]
            contour_arrays += [None] * (len(arrays) - len(contour_arrays))

        def title_for_frame(frame: int) -> str:
            return self.annotator.header(
                forecast_date=forecast_ctx[frame],
                history_ctx=history_ctx,
                variable_name=variable_name,
            )

        return self.renderer.panels_video(
            arrays,
            cmap=cmaps,
            contour_arrays=contour_arrays,
            contour_level=self.plot_spec.ice_edge_threshold,
            dpi=self.plot_spec.dpi,
            figure_title=title_for_frame,
            footer_text=None if history_ctx is None else self.annotator.footer(),
            fps=self.plot_spec.video_fps,
            group_axes=(0, 1) if self.plot_spec.include_difference else None,
            panel_titles=titles,
            vmax=vmaxs,
            vmin=vmins,
            video_format=self.video_format,
        )
