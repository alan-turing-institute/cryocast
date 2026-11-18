from typing import Self

from torch import nn
from typing_extensions import override


class Freezable(nn.Module):
    """Base class for modules that need to freeze their parameters during training.

    A frozen module stays in eval mode, also when its parent calls `train()`.
    """

    # Class-level default, so that subclasses do not need to set it in __init__
    frozen: bool = False

    def freeze(self) -> Self:
        """Freeze this module's parameters and switch it to eval mode."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.frozen = True
        return self.eval()

    @override
    def train(self, mode: bool = True) -> Self:
        """Set training mode, but keep a frozen module in eval mode."""
        return super().train(mode and not self.frozen)
