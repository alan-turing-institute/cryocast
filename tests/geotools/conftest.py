import numpy as np
import pytest

from icenet_mp.geotools.geographic_grid import GeographicGrid


@pytest.fixture
def minimal_grid() -> GeographicGrid:
    """A minimal 2x2 GeographicGrid for tests that need a stand-in geography."""
    return GeographicGrid(
        "EPSG:4326", "1.0", np.array([-0.5, 0.5]), np.array([88.5, 89.5])
    )
