import logging

from lightning.pytorch.accelerators import AcceleratorRegistry

from .xpu_accelerator import XPUAccelerator

logger = logging.getLogger(__name__)


def register_accelerators() -> None:
    """Register all accelerators with Lightning."""
    if "xpu" not in AcceleratorRegistry.available_accelerators():
        AcceleratorRegistry.register(
            "xpu",
            XPUAccelerator,
            description=XPUAccelerator.description,
        )
        logger.debug("Registered XPUAccelerator with Lightning.")


__all__ = [
    "XPUAccelerator",
    "register_accelerators",
]
