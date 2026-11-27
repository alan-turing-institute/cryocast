from .anemoi import patch_collect_analytics
from .lightning import XPUAccelerator, register_accelerators
from .torch import (
    patch_interpolate_antialias,
    patch_open_file_limit,
    patch_parameter_deepcopy,
)

__all__ = [
    "XPUAccelerator",
    "patch_collect_analytics",
    "patch_interpolate_antialias",
    "patch_open_file_limit",
    "patch_parameter_deepcopy",
    "register_accelerators",
]
