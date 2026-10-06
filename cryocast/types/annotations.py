from jaxtyping import Float, Int
from numpy import float32, int64
from numpy.typing import NDArray
from torch import Tensor

# Numpy arrays
ArrayHW = Float[NDArray[float32], "height width"]
ArrayHWV = Float[NDArray[float32], "height width variable"]
ArrayCHW = Float[NDArray[float32], "channels height width"]
ArrayTHW = Float[NDArray[float32], "time height width"]
ArrayTCHW = Float[NDArray[float32], "time channels height width"]
ArrayIndices2D = Int[NDArray[int64], "height width"]

# PyTorch tensors
TensorNCHW = Float[Tensor, "batch channels height width"]
TensorNTCHW = Float[Tensor, "batch time channels height width"]
