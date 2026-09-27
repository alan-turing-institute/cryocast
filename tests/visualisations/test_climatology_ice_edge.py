from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import Metadata, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.matplotlib_renderer import MatplotlibRenderer
from icenet_mp.visualisations.panel_renderer import (
    CLIMATOLOGY_ICE_EDGE_COLOUR,
    PanelRenderer,
)


def test_climatology_reference_contour_targets_prediction_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass climatology as a magenta reference contour on prediction only."""
    fake_render = MagicMock(return_value=MagicMock(spec=ImageFile))
    monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
    renderer = PanelRenderer(
        LandMask(None), Metadata(), PlotSpec(include_difference=True)
    )
    prediction = np.zeros((8, 8), dtype=np.float32)
    climatology = np.ones((8, 8), dtype=np.float32)

    renderer.static_triplet(
        prediction,
        prediction,
        forecast_date=datetime(2026, 1, 1),
        reference_contour=climatology,
        variable_name="sic",
    )

    call = fake_render.call_args
    reference_arrays = call.kwargs["reference_contour_arrays"]
    assert reference_arrays[0] is None
    np.testing.assert_array_equal(reference_arrays[1], climatology)
    assert reference_arrays[2] is None
    assert call.kwargs["reference_contour_color"] == CLIMATOLOGY_ICE_EDGE_COLOUR
    assert call.kwargs["reference_contour_level"] == 0.15


def test_climatology_reference_contour_rejects_shape_mismatch() -> None:
    """Reject a climatology reference field that does not match prediction shape."""
    renderer = PanelRenderer(LandMask(None), Metadata(), PlotSpec())
    with pytest.raises(InvalidArrayError, match="Array shapes must match"):
        renderer.static_triplet(
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            forecast_date=datetime(2026, 1, 1),
            reference_contour=np.zeros((7, 8), dtype=np.float32),
            variable_name="sic",
        )


def test_static_triplet_renders_with_climatology_reference_contour() -> None:
    """Render a static prediction triplet with an additional climatology ice edge."""
    x = np.linspace(0.0, 1.0, 48, dtype=np.float32)
    prediction = np.tile(x, (48, 1))
    ground_truth = np.roll(prediction, 1, axis=1)
    climatology = np.roll(prediction, 4, axis=1)
    renderer = PanelRenderer(
        LandMask(None), Metadata(), PlotSpec(include_difference=True, dpi=72)
    )

    image = renderer.static_triplet(
        ground_truth,
        prediction,
        forecast_date=datetime(2026, 1, 1),
        reference_contour=climatology,
        variable_name="sic",
    )

    assert image.width > 0
    assert image.height > 0
