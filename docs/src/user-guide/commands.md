# Commands

!!! note
    When running locally, set `base_path` in your local config and pass `--config-name NAME_OF_LOCAL_CONFIG` to each command.

## `datasets create`

You will need a [CDS account](https://cds.climate.copernicus.eu/how-to-api) to download ERA5 data with `anemoi`.
By default, a sample dataset will be downloaded which should be small enough to fit on your personal computer. The full datasets are available on Isambard.

```bash
uv run imp datasets create
```

Anemoi tracks which date groups have been downloaded, so an interrupted download can be resumed by simply rerunning this command.

To create the synthetic dataset, use:

```bash
uv run imp datasets create --config-name synthetic
```

## `datasets inspect`

```bash
uv run imp datasets inspect
```

Prints basic properties of each dataset.
With the `--verbose` option it will also print statistical summaries of the variables.

## `datasets plot`

```bash
uv run imp datasets plot --dataset samp-sicsouth-osisaf-25p0km-2020-2024-24h-v1 --timestep 0
```

Creates one static PNG per variable for the selected timestep of a configured downloaded dataset.
Plots are written to `${base_path}/data/input_plots` in a subdirectory named after the dataset.
Running without `--dataset` will plot every configured dataset.
This is useful for inspecting raw inputs without running model training or evaluation.

Use `--timestep` to select another dataset index and the normal `--config-name` to choose the dataset configuration.

Pass `--video` to create an animation over consecutive timesteps instead of a single static plot:

```bash
uv run imp datasets plot --dataset samp-sicsouth-osisaf-25p0km-2020-2024-24h-v1 --video --timestep 0 --n-steps 30
```

This animates `--n-steps` consecutive timesteps starting at `--timestep`, writing one video file per variable to the same `input_plots` directory.

## `train`

Standard (non-synthetic) runs use [Weights & Biases](https://docs.wandb.ai/models/quickstart).
Generate an API key, then authenticate:

```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

Trains the model end-to-end:

```bash
uv run imp train
```

??? warning "macOS: MPS fallback"
    You may need to set `PYTORCH_ENABLE_MPS_FALLBACK=1`:

    ```bash
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run imp train
    ```

### Multistage training

For `EncodeProcessDecode` models, pass `--multistage` to train each component separately before finetuning.

```bash
uv run imp train --multistage
```

Checkpoints are saved to `${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints/<name>.ckpt`, where `BASE_DIR` is the base path defined in your config.

See [Train in stages](../how-to/train-multistage.md) for a full walkthrough.

### Weights & Biases logging

To disable logging to W&B, set either `reporting.loggers.wandb.offline=true` or the `WANDB_MODE=offline` environment variable

```bash
uv run imp train reporting.loggers.wandb.offline=true
WANDB_MODE=offline uv run imp train
```

Run data such as metrics and figures will still be written locally, but will not be uploaded.

Synthetic experiments do not use W&B.
Use the synthetic configuration, which saves metrics and plotting artefacts locally under `${BASE_DIR}/report`:

```bash
uv run imp train --config-name synthetic
```

## `sweep initialise`

```bash
uv run imp sweep initialise --sweep-yaml example.sweep.yaml --config-name baseline/02_cnn_unet_cnn
```

Creates a W&B sweep and initialises a local Optuna study directory; hyperparameters are sampled per trial at runtime.
See [Run a hyperparameter sweep](../how-to/sweeps.md) for the full workflow.

## `sweep trial`

```bash
uv run imp sweep trial --sweep-path <path to sweep directory created above>
```

Runs a single hyperparameter trial as part of a W&B sweep.
See [Run a hyperparameter sweep](../how-to/sweeps.md) for the full workflow.

## `sweep summarise`

```bash
uv run imp sweep summarise --sweep-path <path to sweep directory created above>
```

Reads the local Optuna study and reports the number of completed trials, plus the value and hyperparameters for the best trial.
This works without a W&B connection.
See [Run a hyperparameter sweep](../how-to/sweeps.md) for the full workflow.

## `evaluate`

```bash
uv run imp evaluate --checkpoint PATH_TO_A_CHECKPOINT
```

Pass `--save-predictions` to save denormalised model predictions for the configured test period in NetCDF format, written to `predictions.nc` in the run's directory:

```bash
uv run imp evaluate --checkpoint PATH_TO_A_CHECKPOINT --save-predictions
```

Use `data.split.test` to control which dates are exported. NetCDF export currently requires single-process evaluation.

### Visualisations

To plot static images or animations of the raw input data, add the following to your local config:

```yaml
evaluate:
  callbacks:
    plotting:
      make_input_plots: true
```

Output directories, styling, and animation parameters can be altered by changing `config.evaluate.callbacks.plotting.plot_spec`.
Any of these can be overridden at the command line.

## `feature-importance`

```bash
uv run imp feature-importance
```

Fits a single Random Forest to predict the configured training target from the spatial
and temporal mean of each input variable, then prints the variables ranked by the
forest's built-in feature importance. This is an exploratory signal for which inputs the
data supports, not a model-quality metric. The target group's own variables are included
as features and typically dominate the top of the ranking, since the target's previous
step is usually its best predictor.
