# Add a processor

A processor sits between the encoders and decoder in the encode-process-decode pipeline.
It receives the concatenated latent representations of all inputs and produces a latent forecast.

## The processor interface

All CryoCast processors extend `BaseProcessor` from `cryocast.models.processors`.
They operate on tensors in `NTCHW` format, taking in a tensor with a number of history steps and returning a tensor with a number of forecast steps.
For example, with 3 history steps, and 4 forecast steps, a processor will convert a tensor of shape `(N, 3, C, H, W)` to `(N, 4, C, H, W)`

The base class exposes two entry points, and you only need to implement one:

| Method | Signature | When to override |
|--------|-----------|-----------------|
| `forward` | `(x: TensorNCHW) -> TensorNCHW` | Stateless single-timestep transforms |
| `rollout` | `(x: TensorNTCHW, y: TensorNTCHW \| None) -> ProcessorOutput` | Any model that needs access to the full temporal history, or that behaves differently during training vs. inference |

The default `rollout` implementation calls `forward` once per forecast step, passing each prediction back as the next input.
If your architecture works on one timestep at a time and uses the same logic during training and inference, only overriding `forward` is sufficient.

## Simple processor: override `forward`

```python
from typing import Any
from cryocast.models.processors import BaseProcessor
from cryocast.types import TensorNCHW


class MyProcessor(BaseProcessor):
    def __init__(self, *, hidden_dim: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        in_channels = self.data_space.channels
        self.model = ...  # your nn.Module here

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
```

This model can be trained in either single-stage or multistage mode.


## Training vs. inference: override `rollout`

Some processors do different operations during training and during inference.
A diffusion model is an example of this type of processor.
During training, it adds noise to the ground truth and predicts that noise.
During inference, it starts from random noise and removes the noise step by step.

These processors must be trained in multiple stages.
In multistage training, CryoCast trains the encoders and the decoder before the processor.
CryoCast then trains the processor on a fixed latent space.

The `rollout` method gets two tensors:

- `x` contains the encoded inputs for the history steps. Its shape is `(N, T_history, C, H, W)`.
- `y` contains the encoded inputs for the forecast steps. Its shape is `(N, T_forecast, C, H, W)`.

`x` and `y` are in the same latent space.
They have the same channels `C`, in the same sequence.
`y` contains all the input groups, not only the target group.
Thus, the processor does not have to find the target channels in the latent space.

The value of `y` tells the processor which step is in progress:

- If `y` is a tensor, the step is **training**.
- If `y` is `None`, the step is **inference**.

```python
from cryocast.models.processors import BaseProcessor
from cryocast.types import ProcessorOutput, TensorNTCHW


class MyDiffusionProcessor(BaseProcessor):
    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        # x: (N, T_history, C, H, W) - encoded inputs for the history steps
        # y: (N, T_forecast, C, H, W) - encoded inputs for the forecast steps
        if y is not None:
            # Training: calculate a custom loss and return it with the prediction.
            prediction, loss = self._training(x, y)
            return ProcessorOutput(prediction=prediction, loss=loss)
        # Inference: no loss is necessary.
        return ProcessorOutput(prediction=self._inference(x))
```

### Custom loss

To use your own loss, set `loss` on the `ProcessorOutput` that `rollout` returns.
`ProcessorStage` then uses this loss for the backward pass.
It does not calculate its own loss.
It decodes the prediction under `torch.no_grad()`.
Thus, the decoder makes output for the metrics and the callbacks, but this output does not change the gradients.

If `loss` is `None` (the default), `ProcessorStage` decodes the prediction.
It then calculates its own loss against the target.

### Loss in latent space

If your processor calculates its loss in latent space, give `computes_loss_in_latent_space=True` to `BaseProcessor.__init__`.
CryoCast then does these steps automatically:

1. The data module loads each input group for the forecast steps.
   Each batch gets one `future_<input>` key for each input group.
2. The model encodes these inputs and gives the result to `rollout` as `y`.
   The model detaches `y` from the computation graph. Thus, the loss cannot change its own target.
3. The model freezes the decoder.

You do not set a configuration key for these steps.

!!! warning
    The `future_<input>` keys contain ground-truth data for the forecast steps.
    The model uses them only to calculate the training loss.
    The model does not give them to `forward`.
    Do not use them in a different part of the model.

!!! note
    The data module reads more data for each sample when it loads the forecast steps.
    It loads them only for the processor and finetune stages, because the encoder and decoder stages do not use them.
    A sample is valid only if all the input groups have data for all the forecast steps.
    Thus, the dataset can contain fewer samples. For example, the Argo data has some missing dates.

!!! note
    The loss is calculated on all the channels of the latent space, with the same weight for each channel.
    Most of these channels come from input groups that are not the target, for example ERA5.

## Register the processor in config

Add a model config under `cryocast/config/model/` that points `processor._target_` at your class:

```yaml
# cryocast/config/model/cnn_mydiffusion_cnn.yaml
_target_: cryocast.models.EncodeProcessDecode

name: cnn-ddpm-cnn

encoders:
  latent_space: [144, 144]
  era5:
    _target_: cryocast.models.encoders.CNNEncoder
  sic-osisaf:
    _target_: cryocast.models.encoders.CNNEncoder

processor:
  _target_: cryocast.models.processors.MyDiffusionProcessor
  timesteps: 1000

decoder:
  _target_: cryocast.models.decoders.CNNDecoder
  bounded: false
```

Then run training with:

```bash
uv run cryocast train model=cnn_mydiffusion_cnn
```
