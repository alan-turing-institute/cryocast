"""Regression tests for filter registration with anemoi-transform.

These tests use the filter_registry which depends on anemoi-datasets internals. If
anemoi-datasets changes those internals, these tests should fail rather than silently
breaking dataset creation at runtime.
"""

from typing import ClassVar

from anemoi.transform.filters import filter_registry

from cryocast.ingestion.filters import register_filters
from cryocast.ingestion.filters.crop_latlon_filter import CropLatLonFilter
from cryocast.ingestion.filters.nan_to_num_filter import NanToNumFilter
from cryocast.ingestion.filters.reproject_filter import ReprojectFilter
from cryocast.ingestion.filters.set_geography_filter import SetGeographyFilter


class TestFilterRegistration:
    """Test suite for custom filter registration with anemoi-transform."""

    EXPECTED_FILTERS: ClassVar[dict] = {
        "crop-latlon": CropLatLonFilter,
        "nan-to-num": NanToNumFilter,
        "reproject": ReprojectFilter,
        "set-geography": SetGeographyFilter,
    }

    def test_register_filters_adds_all_filters(self) -> None:
        """All custom filters appear in filter_registry after registration."""
        register_filters()
        for name in self.EXPECTED_FILTERS:
            assert name in filter_registry.registered, (
                f"Filter '{name}' missing from filter_registry.registered after register_filters(). "
                "Check whether anemoi-transform changed its registry API."
            )

    def test_register_filters_lookup_returns_correct_class(self) -> None:
        """filter_registry.lookup() returns our exact class, not a built-in."""
        register_filters()
        for name, cls in self.EXPECTED_FILTERS.items():
            assert filter_registry.lookup(name) is cls, (
                f"filter_registry.lookup('{name}') did not return {cls.__name__}. "
                "anemoi-transform may now ship a built-in filter under this name."
            )

    def test_register_filters_is_idempotent(self) -> None:
        """Calling register_filters() twice does not raise or change registered names."""
        register_filters()
        registered_before = set(filter_registry.registered)
        register_filters()
        registered_after = set(filter_registry.registered)
        assert registered_before == registered_after
