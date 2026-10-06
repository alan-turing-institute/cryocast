from typing import Self

from torch import nn


class Freezable(nn.Module):
    """Base class for modules that need to freeze their parameters during training."""

    def freeze(self) -> Self:
        """Freeze this module's parameters and switch it to eval mode."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        return self.eval()
