import logging
from typing import Any

import torch
from lightning.fabric.utilities.device_parser import (
    _check_data_type,
    _check_unique,
    _normalize_parse_gpu_string_input,
)
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators import Accelerator
from torch import distributed as dist
from typing_extensions import override

logger = logging.getLogger(__name__)


class XPUAccelerator(Accelerator):
    """Accelerator for Intel XPU GPU devices."""

    description = "Intel Data Center GPU Max - codename Ponte Vecchio"

    @override
    def setup_device(self, device: torch.device) -> None:
        """Configure the current process to use a specified device.

        Args:
            device: torch device to use

        Raises:
            MisconfigurationException: If the selected device is not an xpu.

        """
        if device.type != "xpu":
            msg = f"Device should be xpu, got {device} instead."
            raise MisconfigurationException(msg)
        index = getattr(device, "index", None)
        if not isinstance(index, int):
            msg = "Device index could not be determined."
            raise MisconfigurationException(msg)
        torch.xpu.set_device(index - 1)

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Get stats from torch xpu module.

        Args:
            device: torch device to use

        Returns:
            Dictionary of XPU memory allocator statistics for the device.

        """
        return torch.xpu.memory_stats(device)

    @override
    def teardown(self) -> None:
        """Ensure that distributed processes close gracefully."""
        torch.xpu.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    @override
    def parse_devices(devices: int | str | list[int] | None) -> list[int] | None:
        """Accelerator device parsing logic.

        Args:
            devices: Device(s) by number

        Returns:
            List of device numbers to use or None if no devices are requested.

        """
        # Check that devices parameter has the correct data type
        _check_data_type(devices)

        # Handle the case when no GPUs are requested
        if (
            devices is None
            or (isinstance(devices, int) and devices == 0)
            or str(devices).strip() in ("0", "[]")
        ):
            return None

        # Get all available XPUs
        available_xpus = list(range(torch.xpu.device_count()))
        if not available_xpus:
            msg = "XPUs requested but none are available."
            raise MisconfigurationException(msg)

        # Normalise the input into a list of device indices.
        xpus = _normalize_parse_gpu_string_input(devices)

        # Return all available XPUs if requested
        if xpus == -1:
            return available_xpus

        # Check that XPUs are unique.
        if isinstance(xpus, int):
            xpus = list(range(xpus))
        _check_unique(xpus)

        # Check that requested XPUs are available
        for xpu in xpus:
            if xpu not in available_xpus:
                msg = f"You requested xpu: {xpu} but only {len(available_xpus)} are available"
                raise MisconfigurationException(msg)

        return xpus

    @staticmethod
    @override
    def get_parallel_devices(devices: int | list[int]) -> list[torch.device]:
        """Get parallel devices for the Accelerator.

        Args:
            devices: List of device numbers

        Returns:
            List of devices.

        """
        if isinstance(devices, int):
            devices = list(range(devices))
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get number of available devices from torch xpu module."""
        return torch.xpu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        """Determines if an XPU is actually available.

        Returns:
            True if devices are detected, otherwise False.

        """
        try:
            return torch.xpu.device_count() > 0
        except (AttributeError, NameError):
            return False

    @staticmethod
    @override
    def name() -> str:
        return "xpu"
