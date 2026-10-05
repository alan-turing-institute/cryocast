"""Shared physical/domain constants used across the package."""

# Concentration threshold defining ice/no-ice, and hence the ice edge
SEA_ICE_THRESHOLD = 0.15

# CF attributes for known prediction variables, keyed by variable name
CF_VARIABLE_ATTRIBUTES: dict[str, dict[str, str | float]] = {
    "ice_conc": {
        "standard_name": "sea_ice_area_fraction",
        "units": "1",
        "valid_min": 0.0,
        "valid_max": 1.0,
    },
}

# Array dimensions
NDIM_HW = 2  # [height, width]
NDIM_NHW = 3  # [batch, height, width]
NDIM_THW = 3  # [time, height, width]
NDIM_NTCHW = 5  # [batch, time, channels, height, width]
