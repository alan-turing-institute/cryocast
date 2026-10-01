import logging
from collections.abc import Mapping
from typing import Any

from icenet_mp.types import ColourScale, PlotSpec

log = logging.getLogger(__name__)


class StyleResolver:
    """Resolve variable style from per-variable style configs."""

    def __init__(self, plot_spec: PlotSpec) -> None:
        """Construct a StyleResolver from a PlotSpec."""
        self._variable_styles = plot_spec.per_variable_styles
        if not isinstance(self._variable_styles, Mapping):
            log.warning(
                "Could not interpret %s as a variable style config",
                type(self._variable_styles),
            )
            self._variable_styles = {}
        self._default_cmap = plot_spec.colourmap
        self._default_vmin = plot_spec.vmin
        self._default_vmax = plot_spec.vmax

    def _match(self, var_name: str) -> Mapping[str, Any] | None:
        """Return the best matched style config for a variable from the bound dict.

        Matching priority:
          1) exact key
          2) wildcard prefix key ending with '*'
          3) _default
          4) no match (None)
        """
        candidates = (
            self._variable_styles.get(var_name),
            self._wildcard_match(var_name),
            self._variable_styles.get("_default"),
        )
        return next((spec for spec in candidates if isinstance(spec, Mapping)), None)

    def _wildcard_match(self, var_name: str) -> Mapping[str, Any] | None:
        """Scan wildcard-suffixed keys (e.g. 'era5:*') for a prefix match."""
        for key, spec in self._variable_styles.items():
            if not (isinstance(key, str) and key.endswith("*")):
                continue
            prefix = key[:-1]
            # If prefix is empty (user wrote '*' only) skip it
            if prefix and var_name.startswith(prefix) and isinstance(spec, Mapping):
                return spec
        return None

    def colour_scale(self, variable_name: str) -> ColourScale:
        """Construct a ColourScale for this variable.

        Default to the colourmap, vmin, and vmax from the bound PlotSpec if the matched
        style does not override them.

        Args:
            variable_name: The variable name to resolve a style for.

        Returns:
            A ColourScale with the matched style's cmap, vmin, vmax, and units

        """
        spec = self._match(variable_name) or {}
        vmin = spec.get("vmin")
        vmax = spec.get("vmax")
        return ColourScale(
            cmap=spec.get("cmap") or self._default_cmap,
            vmin=vmin if vmin is not None else self._default_vmin,
            vmax=vmax if vmax is not None else self._default_vmax,
            units=spec.get("units"),
        )
