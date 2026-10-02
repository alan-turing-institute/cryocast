"""Shared physical/domain constants used across the package."""

# Concentration threshold defining ice/no-ice, and hence the ice edge
SEA_ICE_THRESHOLD = 0.15

# Array dimensions
NDIM_HW = 2  # [height, width]
NDIM_CHW = 3  # [channels, height, width]
NDIM_NHW = 3  # [batch, height, width]
NDIM_THW = 3  # [time, height, width]
NDIM_NTCHW = 5  # [batch, time, channels, height, width]
