import torch
import torch.nn.functional as F


def patch_interpolate_antialias() -> None:
    """Patch F.interpolate to strip antialias=True.

    Two hardware paths require this:
    - CUDA deterministic mode: upsample_bilinear2d_aa has no deterministic backward pass.
    - MPS: upsample_bilinear2d_aa is unimplemented; the PYTORCH_ENABLE_MPS_FALLBACK
      path triggers an MPS synchronise after a UNet forward that hits a Metal driver
      bug and SIGSEGVs.
    """
    if getattr(F.interpolate, "is_patched", False):
        return

    _original = F.interpolate

    def _interpolate(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        kwargs.pop("antialias", None)
        return _original(tensor, *args, **kwargs)  # type: ignore[arg-type]

    _interpolate.is_patched = True  # type: ignore[attr-defined]
    F.interpolate = _interpolate  # type: ignore[assignment]
