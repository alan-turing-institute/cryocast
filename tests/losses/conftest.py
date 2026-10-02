import torch


def make_fields(
    seed: int = 0, shape: tuple[int, ...] = (2, 2, 1, 32, 32)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded (prediction, target) pair of random NTCHW fields."""
    generator = torch.Generator().manual_seed(seed)
    prediction = torch.rand(*shape, generator=generator)
    target = torch.rand(*shape, generator=generator)
    return prediction, target
