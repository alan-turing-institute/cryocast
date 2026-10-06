import torch
from torch import nn


def patch_parameter_deepcopy() -> None:
    """Patch nn.Parameter.__deepcopy__ to avoid a segfault on Apple Silicon MPS.

    ``Parameter.__deepcopy__`` calls ``self.data.clone(memory_format=preserve_format)``,
    which crashes for any tensor previously touched by the MPS runtime. The workaround
    used here is to move the data to MPS, make it contiguous, clone it, and then move it
    back to the original device.
    """
    # Only apply the patch if MPS is available and it has not already been applied
    if not torch.backends.mps.is_available():
        return
    if getattr(nn.Parameter.__deepcopy__, "is_patched", False):
        return

    def _deepcopy(self: nn.Parameter, _memo: dict) -> nn.Parameter:
        """Deepcopy a parameter, avoiding segfaults on Apple Silicon MPS."""
        device = self.data.device
        return nn.Parameter(
            self.data.detach().to("mps").contiguous().clone().to(device),
            self.requires_grad,
        )

    _deepcopy.is_patched = True  # type: ignore[attr-defined]
    nn.Parameter.__deepcopy__ = _deepcopy  # type: ignore[assignment]
