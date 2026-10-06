from torch import nn

ACTIVATION_FROM_NAME: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}
