# Add a model

## Tensor format

All IceNet-MP models operate on tensors in `NTCHW` format:

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
  _target_: icenet_mp.models.processors.ConvLSTMProcessor
  hidden_channels: 128
  kernel_size: 3
  n_layers: 2
  dropout: 0.1
  residual: true
```

A complete example is available as `model=cnn_convlstm_cnn`.
