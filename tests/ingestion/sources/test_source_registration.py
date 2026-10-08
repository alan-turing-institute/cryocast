"""Regression tests for source registration with anemoi-datasets.

These tests use the source_registry and the Recipe.model_rebuild() path, which depend
on anemoi-datasets internals (_factories, _schemas, _action_discriminator). If
anemoi-datasets reorganises those internals, these tests should fail rather than
silently breaking dataset creation at runtime.
"""

import datetime
from typing import ClassVar

from anemoi.datasets.create.recipe import Recipe
from anemoi.datasets.create.recipe.dates import StartEndDates
from anemoi.datasets.create.sources import source_registry

from cryocast.ingestion.sources import (
    ArgoSource,
    CDSSource,
    FTPSource,
    SyntheticSource,
    register_sources,
)


class TestSourceRegistration:
    """Test suite for custom source registration with anemoi-datasets."""

    EXPECTED_SOURCES: ClassVar[dict] = {
        "ftp": FTPSource,
        "argo": ArgoSource,
        "cds": CDSSource,
        "synthetic": SyntheticSource,
    }

    def test_register_sources_with_real_registry(self) -> None:
        """Custom sources appear in the real source_registry after registration."""
        register_sources()
        for name in self.EXPECTED_SOURCES:
            assert name in source_registry.registered, (
                f"Source '{name}' missing from source_registry.registered. "
                "Check whether anemoi-datasets changed its registry API."
            )

    def test_register_sources_lookup_returns_correct_class(self) -> None:
        """source_registry.lookup() returns the exact class we registered."""
        register_sources()
        for name, cls in self.EXPECTED_SOURCES.items():
            assert source_registry.lookup(name) is cls

    def test_register_sources_rebuilds_recipe_model(self) -> None:
        """Recipe.model_rebuild() completes without error after source registration.

        This test guards against anemoi-datasets renaming or removing the internal
        _factories / _schemas / _action_discriminator symbols that register_sources()
        relies on to patch the Recipe Action union.
        """
        register_sources()
        # If model_rebuild succeeded, Recipe.model_fields should still be populated
        assert Recipe.model_fields, (
            "Recipe.model_fields is empty after register_sources() — "
            "model_rebuild() may have failed silently."
        )

    def test_recipe_accepts_registered_cds_source(self) -> None:
        """The rebuilt recipe model accepts the CDS source configuration."""
        register_sources()
        recipe = Recipe(
            dates=StartEndDates(
                start=datetime.datetime(2024, 1, 1, 12, 0, 0),
                end=datetime.datetime(2024, 1, 2, 12, 0, 0),
                frequency=datetime.timedelta(hours=24),
            ),
            input={
                "cds": {
                    "dataset": "reanalysis-pan-carra",
                    "time_from_dates": True,
                    "request": {
                        "level_type": "single_levels",
                        "variable": ["sea_ice_area_fraction"],
                        "product_type": "analysis",
                        "data_format": "grib",
                        "area": [81.0, 15.0, 76.0, 35.0],
                    },
                }
            },
        )
        assert type(recipe.input).__name__ == "cds"

    def test_recipe_accepts_registered_synthetic_source(self) -> None:
        """The rebuilt recipe model accepts the synthetic source configuration."""
        register_sources()
        recipe = Recipe(
            dates=StartEndDates(
                start=datetime.datetime(2020, 1, 1, 0, 0, 0),
                end=datetime.datetime(2020, 1, 3, 0, 0, 0),
                frequency=datetime.timedelta(hours=24),
            ),
            input={
                "synthetic": {
                    "dynamics": "moving",
                    "grid_size": 32,
                    "n_trajectories": 3,
                    "start_date": "2020-01-01T00:00:00",
                }
            },
        )
        assert type(recipe.input).__name__ == "synthetic"

    def test_register_sources_is_idempotent(self) -> None:
        """Calling register_sources() twice does not raise or corrupt the registry."""
        register_sources()
        registered_before = set(source_registry.registered)
        register_sources()
        registered_after = set(source_registry.registered)
        assert registered_before == registered_after
