from typing import TYPE_CHECKING

from torch import nn

from cryocast.types import RangeRestriction, TensorNCHW

if TYPE_CHECKING:
    from collections.abc import Callable


class RestrictRange(nn.Module):
    def __init__(
        self, method: RangeRestriction, *, min_val: float = 0.0, max_val: float = 1.0
    ) -> None:
        """Restrict the values of the input tensor to be within the given range.

        This can use torch.clamp, torch.sigmoid, or torch.tanh.
        """
        super().__init__()
        self.restrict_fn: Callable[[TensorNCHW], TensorNCHW]
        try:
            diff = max_val - min_val
            self.restrict_fn = {
                RangeRestriction.CLAMP: lambda x: x.clamp_(min=min_val, max=max_val),
                RangeRestriction.SIGMOID: lambda x: min_val + diff * x.sigmoid_(),
                RangeRestriction.TANH: lambda x: min_val + diff * (x.tanh_() + 1) / 2,
            }[method]
        except KeyError:
            self.restrict_fn = lambda x: x

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Apply the restriction function to the input tensor."""
        return self.restrict_fn(x)
