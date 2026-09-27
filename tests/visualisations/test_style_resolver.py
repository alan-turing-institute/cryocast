"""Tests for icenet_mp/visualisations/variable_style_resolver.py."""

from typing import Any

from icenet_mp.types import PlotSpec
from icenet_mp.visualisations.style_resolver import StyleResolver

DEFAULT_CMAP = "viridis"
DEFAULT_VMIN = 3
DEFAULT_VMAX = 5


def _resolver(
    styles: Any,  # noqa: ANN401
    *,
    cmap: str = DEFAULT_CMAP,
    **plot_spec_kwargs: Any,
) -> StyleResolver:
    plot_spec_kwargs.setdefault("vmin", DEFAULT_VMIN)
    plot_spec_kwargs.setdefault("vmax", DEFAULT_VMAX)
    return StyleResolver(
        PlotSpec(per_variable_styles=styles, colourmap=cmap, **plot_spec_kwargs)
    )


class TestStyleForVariable:
    def test_empty_styles_falls_back_to_default_cmap(self) -> None:
        """An empty styles mapping resolves cmap to the bound default."""
        scale = _resolver({}).colour_scale("era5:2t")

        assert scale.cmap == DEFAULT_CMAP
        assert scale.vmin == DEFAULT_VMIN
        assert scale.vmax == DEFAULT_VMAX

    def test_none_styles_falls_back_to_default_cmap(self) -> None:
        """A None styles mapping resolves cmap to the bound default."""
        scale = _resolver(None).colour_scale("era5:2t")

        assert scale.cmap == DEFAULT_CMAP
        assert scale.units is None

    def test_non_mapping_styles_falls_back_to_default_cmap(self) -> None:
        """A styles value that is not a Mapping (e.g. a list) is ignored."""
        scale = _resolver(["not", "a", "mapping"]).colour_scale("era5:2t")

        assert scale.cmap == DEFAULT_CMAP

    def test_unnormalised_name_does_not_match(self) -> None:
        """A variable name spelled differently from the style key does not match."""
        styles = {"era5:2t": {"cmap": "RdBu_r", "units": "K"}}

        scale = _resolver(styles).colour_scale("era5-2t")

        assert scale.cmap == DEFAULT_CMAP

    def test_default_fallback(self) -> None:
        """An unmatched variable name falls back to the '_default' style."""
        styles = {"_default": {"cmap": "grey"}}

        scale = _resolver(styles).colour_scale("totally:unmatched")

        assert scale.cmap == "grey"

    def test_no_match_no_default_falls_back_to_default_cmap(self) -> None:
        """No exact/wildcard/_default match resolves cmap to the bound default."""
        styles = {"era5:2t": {"cmap": "RdBu_r"}}

        scale = _resolver(styles).colour_scale("osisaf:ice_conc")

        assert scale.cmap == DEFAULT_CMAP

    def test_bare_wildcard_key_is_skipped(self) -> None:
        """A wildcard key of just '*' (empty prefix) is skipped, not treated as catch-all."""
        styles = {"*": {"cmap": "ignored"}, "_default": {"cmap": "fallback"}}

        scale = _resolver(styles).colour_scale("anything:at_all")

        assert scale.cmap == "fallback"

    def test_wildcard_candidate_not_a_dict_falls_back_to_default_cmap(self) -> None:
        """A matching wildcard key whose value isn't a Mapping is logged and skipped."""
        styles: dict[str, Any] = {"era5:*": "not-a-mapping"}

        scale = _resolver(styles).colour_scale("era5:2t")

        assert scale.cmap == DEFAULT_CMAP

    def test_exact_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test exact variable name matching in styling."""
        scale = _resolver(variable_styles).colour_scale("era5:2t")

        assert scale.cmap == "RdBu_r"
        assert scale.units == "K"

    def test_wildcard_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test wildcard pattern matching in styling."""
        # Add wildcard pattern
        styles_with_wildcard = {
            **variable_styles,
            "era5:q_*": {"cmap": "viridis", "units": "kg/kg"},
        }

        scale = _resolver(styles_with_wildcard).colour_scale("era5:q_500")

        assert scale.cmap == "viridis"
        assert scale.units == "kg/kg"


class TestDefaultCmapFallback:
    def test_matched_style_without_cmap_falls_back_to_default(self) -> None:
        """A matched style that omits cmap still resolves to the bound default."""
        styles = {"era5:2t": {"units": "K"}}

        scale = _resolver(styles).colour_scale("era5:2t")

        assert scale.cmap == DEFAULT_CMAP
        assert scale.units == "K"
