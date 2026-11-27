import logging

from anemoi.transform.filters import filter_registry

from .nan_to_num_filter import NanToNumFilter
from .reproject_filter import ReprojectFilter
from .rotate_winds_polar_filter import RotateWindsPolarFilter
from .set_geography_filter import SetGeographyFilter

logger = logging.getLogger(__name__)


def register_filters() -> None:
    """Register all filters with anemoi-transform."""
    filters = {
        "nan-to-num": NanToNumFilter,
        "reproject": ReprojectFilter,
        "rotate-winds-polar": RotateWindsPolarFilter,
        "set-geography": SetGeographyFilter,
    }
    for filter_name, filter_class in filters.items():
        if filter_name not in filter_registry.registered:
            filter_registry.register(filter_name, filter_class)
            logger.debug("Registered %s with anemoi-transform.", filter_class.__name__)


__all__ = ["register_filters"]
