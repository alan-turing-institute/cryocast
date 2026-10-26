# Add a model

## Tensor format

All CryoCast models operate on tensors in `NTCHW` format:

| Dimension | Meaning |
|-----------|---------|
| `N` | Batch size |
| `T` | History steps (inputs) or forecast steps (outputs) |
| `C` | Channels / variables |
| `H` | Height |
| `W` | Width |

`N` and `T` are the same across all inputs, but `C`, `H`, and `W` may differ per dataset.

For example, with 3 history steps, and 4 forecast steps, each of the `k` inputs each have shape `(N, 3, C_k, H_k, W_k)` and the output has shape `(N, 4, C_out, H_out, W_out)`.

## Standalone models

A standalone model accepts a `dict[str, TensorNTCHW]` mapping dataset names to tensors and produces output of shape `(N, T, C_out, H_out, W_out)`.
Each model instance is typically trained to predict a single output, although this is not a hard constraint.

![Standalone pipeline diagram](../assets/pipeline-standalone.png)

| | |
|---|---|
| **Pros** | All input variables available without transformation. |
| **Cons** | Combining datasets of different shapes or types must be done inside the model. |

## Processor models

A processor model sits inside an encode-process-decode pipeline.
You define a latent space `(H_latent, W_latent)` and the framework automatically creates one encoder per input and one decoder per output.

1. Each dataset-specific **encoder** maps input `(N, T_history, C_k, H_k, W_k)` to `(N, T_history, C_k_latent, H_latent, W_latent)`.
2. The `k` encoded tensors are concatenated to `(N, T_history, C_latent, H_latent, W_latent)`.
3. The **processor** maps `(N, T_history, C_latent, H_latent, W_latent)` to `(N, T_forecast, C_latent, H_latent, W_latent)`.
4. Each output-specific **decoder** maps the processor output, `(N, T_forecast, C_latent, H_latent, W_latent)`, to `(N, T_forecast, C_out, H_out, W_out)`.

![Encode-process-decode pipeline diagram](../assets/pipeline-encode-process-decode.png)

| | |
|---|---|
| **Pros** | Inputs are converted into a common latent space, freeing up the model to learn time evolution. |
| **Cons** | Latent space representation may lose some spatial correlations present in the inputs. |

### ConvLSTM processor

`ConvLSTMProcessor` keeps the time dimension explicit instead of flattening the history window into channels. It consumes encoded history frames sequentially, maintains spatial hidden and cell states, and then generates forecast frames autoregressively by feeding each prediction back into the recurrent state.

A residual forecast head is enabled by default so the processor learns a latent-space tendency around persistence. Set `residual: false` to predict the next latent frame directly.

```yaml
processor:
  _target_: cryocast.models.processors.ConvLSTMProcessor
  hidden_channels: 128
  kernel_size: 3
  n_layers: 2
  dropout: 0.1
  residual: true
```

A complete example is available as `model=cnn_convlstm_cnn`.

## Mixture-of-experts forecasting

`model=cnn_moe_cnn` selects an optional dense mixture of UNet and ViT processors.
Each expert receives the same observed latent history and generates a complete
forecast. A learned gate pools the latest history frame spatially and produces
one softmax distribution per sample. The forecast is the weighted sum of the
expert forecasts; the same weights apply to every forecast lead and channel.
Experts evolve their own forecast trajectories before blending. This differs
from blending each next step and feeding that mixture back into every expert.

All experts run on every call, including experts whose weights are near zero.
This implementation therefore provides soft model combination, with no sparse
routing or inference-speed claim. Existing model defaults are unchanged.

The wrapper accepts only prediction-only `BaseProcessor` experts. Shared latent
spaces and history/forecast lengths are supplied during construction. Runtime
checks enforce each expert's full `(batch, forecast, channels, height, width)`
output shape. Future observations are never forwarded to experts. A custom
latent-loss expert, or a mixture claiming to own a latent loss, is rejected
until an explicit objective and routing policy exists.

Routing pooling, projections and softmax use at least float32, even under
autocast. Forecast blending accumulates at that precision before returning the
experts' output dtype. Float64 inputs and parameters retain double precision.
Gate and expert parameters remain differentiable and are saved in checkpoints.

Use the standard latent rollout. A physical-space one-step mixture policy is
not implemented, so `rollout_space=physical` raises the existing unsupported
processor error. Forecasting improvements over a single expert or persistence
require matched held-out evaluation; routing and gradient tests establish
correctness only.
