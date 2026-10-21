# Run multistage training

Multistage training is an alternative to [normal training](./train.md) for `EncodeProcessDecode` models.
Instead of training the full model in one pass, each component is trained in isolation before the weights are loaded into the full model for a final finetuning step.
This can be useful when the model is large, when encoder inputs have very different characteristics, or when you want finer control over the training of individual components.

## The four stages

**Stage 1 - Train encoders**

Each encoder is trained independently as a standalone autoencoder (encoder + disposable decoder). One training run per encoder.

![Stage 1 diagram](../assets/staged-training-stage1.png)

**Stage 2 - Train decoder**

The decoder is trained on the combined frozen encoder latents from stage 1.

![Stage 2 diagram](../assets/staged-training-stage2.png)

**Stage 3 - Train processor**

The processor is trained with frozen encoders and decoder from stages 1–2.

![Stage 3 diagram](../assets/staged-training-stage3.png)

**Stage 4 - Finetune**

Pretrained weights are loaded into the full `EncodeProcessDecode` model and the entire model is trained end-to-end.

![Stage 4 diagram](../assets/staged-training-stage4.png)

## Running staged training

```bash
uv run cryocast train --multistage
```

A checkpoint is saved at the end of each stage. To resume a partially completed run, pass `--checkpoint-dir` pointing at the checkpoint directory from the original run - any stage whose checkpoint already exists there will be skipped:

```bash
uv run cryocast train --multistage --checkpoint-dir ${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints
```

## Per-stage config overrides

Each stage inherits the top-level `train` config.
To override settings for a specific stage, add a block under the `multistage` key:

```yaml
train:
  trainer:
    max_epochs: 20
  multistage:
    encoders:
      trainer:
        max_epochs: 10
    decoder:
      trainer:
        max_epochs: 10
    processor:
      trainer:
        max_epochs: 10
    finetune:
      optimizer:
        lr: 1e-4
      trainer:
        max_epochs: 5
```

Any key valid under `train` can be overridden per stage: `optimizer`, `scheduler`, `trainer`, and `callbacks`.
Keys not present in a stage block fall back to the top-level `train` values.
For example, the `lr` for the `encoders`, `decoder` and `processor` stages would be the default `5e-3` if not overridden elsewhere.

## Full combined-latent supervision

A processor that computes its loss in latent space receives the actual future
window of every input group, encoded by the same input encoders and in the same
channel order as history. Training origins must therefore have complete history
and forecast windows for every input, within the configured training periods.
Forecast inputs use reserved `__forecast_input__:<group>` batch keys; these are
added only to latent-loss training, never to validation, test or prediction.
An output group need not be an input when the decoder has no skip connection.

During processor training the encoders and decoder stay frozen. During
finetuning, gradients can reach the shared encoders through history conditioning;
the future latent is computed without an autograd graph, and the decoder remains
frozen on the latent-loss path. Encoder train/eval modes are not changed by the
future-target helper. Diffusion predicts every combined-latent channel instead of
copying auxiliary channels from the last history frame. Autoregressive training
still supervises its first forecast step; parallel training uses the full window.

## Prediction-target metadata

The full-latent training target and a prediction-only processor's output slice
are separate contracts. `data_space_target` remains the complete combined latent
for diffusion. The optional `target_latent_space` and `target_channel_offset`
identify the output dataset's existing encoder channels, in registered encoder
order. They do not create a target encoder, add weights or alter future supervision.
Physical `target_variable_indices` are not latent-channel offsets.

Prediction-only processors such as the space-time ViT can consume this metadata
to forecast only the target slice while persisting other conditioning inputs.
Legacy direct callers may still provide their target slice through
`data_space_target` and `target_channel_offset`. An output group absent from the
inputs has no corresponding slice; slice-based processors must reject that case
rather than silently predicting all conditioning channels.

## Checkpoint compatibility

New latent-loss model and processor-stage checkpoints record
`latent_target_layout: full_combined_v1`. Loading an unversioned checkpoint or a
different latent-target contract into a full-latent model is rejected, even when
parameter shapes happen to match. Older target-slice diffusion checkpoints cannot
be converted by deleting unexpected keys, changing the sampling schedule or using
`strict=False`. Evaluate those checkpoints with their original code. Existing
encoder/decoder stage checkpoints can be reused, but train a new processor and
finetuning stages for the full-latent contract. This format marker also applies
to full-latent checkpoints made before the marker was introduced; they require
explicit provenance review rather than automatic acceptance.

Ordinary non-latent-loss model checkpoints retain strict loading of their active
parameters. The obsolete, frozen `target_encoder.*` namespace is removed from the
loaded state in memory only. The source checkpoint is not rewritten, and unrelated
unexpected or missing weights still fail strict loading. Model configurations and
trained diffusion timestep counts are not changed by these compatibility checks.
