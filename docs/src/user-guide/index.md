# User Guide

This guide covers everything you need to get CryoCast running - from installing the package to training a model and visualising results.

For a first run, install the environment, choose a local data directory and configuration, prepare the required datasets, then train and evaluate a model. Run the commands below from the repository root after completing the installation steps.

## Start here

1. [Installation](installation.md) covers the Python and `uv` prerequisites, local setup, and additional requirements for supported HPC systems.
2. [Configuration](configuration.md) explains local Hydra configuration, model and dataset overrides, platform settings, and input/target variables.
3. [Data](data.md) describes the available OSI SAF, ERA5 and Argo datasets. The selected configuration determines which inputs are downloaded and used.
4. [Commands](commands.md) documents dataset preparation, inspection, training, sweeps, evaluation and feature importance, including account and logging requirements.

## First run with observational data

Create `cryocast/config/my.local.yaml`, replacing the example path with a writable local directory with enough space for the selected data and run outputs:

```yaml
defaults:
  - base
  - _self_

base_path: /path/to/cryocast-data
```

The current `base` configuration selects the southern sample datasets and the `quick_test` model. Check the [data guide](data.md) before downloading. ERA5 requires CDS access and acceptance of the relevant dataset terms; standard training uses Weights & Biases. Follow the [commands guide](commands.md) for authentication or offline logging. Offline logging does not remove the data-download requirements.

Prepare and inspect the data before training:

```bash
uv run cryocast datasets create --config-name my.local
uv run cryocast datasets inspect --config-name my.local --verbose
uv run cryocast train --config-name my.local
```

If a download is interrupted, rerun `datasets create` with the same configuration to resume it. Do not add `--overwrite` unless you intend to replace the existing dataset.

After training, evaluate a checkpoint from that run:

```bash
uv run cryocast evaluate --checkpoint /path/to/run/checkpoints/model.ckpt --config-name my.local
```

Replace the checkpoint path with an existing file. Keep its sibling `files/model_config.yaml` in the run directory: the loader uses it to restore the trained model, variables and forecast window. Continue passing the same local configuration so the data path, datasets and evaluation settings are explicit. See [evaluation](../how-to/evaluate.md) for test-period selection and prediction export.

## First run with local synthetic data

To try the pipeline without CDS downloads or a W&B account, create a separate `cryocast/config/synthetic.local.yaml`:

```yaml
defaults:
  - synthetic
  - _self_

base_path: /path/to/cryocast-synthetic
```

This selects the shipped locally generated data and local-file logger. Then run:

```bash
uv run cryocast datasets create --config-name synthetic.local
uv run cryocast datasets inspect --config-name synthetic.local --verbose
uv run cryocast train --config-name synthetic.local
```

Evaluate a checkpoint from this run with the evaluation command above, changing `--config-name` to `synthetic.local` and supplying its checkpoint path. Synthetic results check pipeline behaviour; they do not establish forecasting skill on observed sea ice. This route is separate from the external-data notebooks.

## Guide contents

| Guide | What it covers |
| --- | --- |
| [Installation](installation.md) | Python environment setup, `uv`, and HPC-specific prerequisites |
| [Configuration](configuration.md) | Hydra configs, model settings, dataset selection, variables, and platform overrides |
| [Data](data.md) | Data sources, spatial and temporal coverage, variables, and dataset structure |
| [Commands](commands.md) | Dataset, training, sweep, evaluation, and analysis commands |
| [Notebooks](notebooks.md) | Demonstrator notebooks, research examples, and notebook-specific prerequisites |
| [Loss functions](loss-functions.md) | Available training losses, configuration, and lead-time weighting |
| [Metrics](metrics.md) | Forecast metrics, interpretation, limitations, and synthetic comparison scenarios |

## Training and evaluation workflows

For task-oriented walkthroughs, use the [how-to guides](../how-to/index.md). They cover:

- [training a model](../how-to/train.md) and [multistage training](../how-to/train-multistage.md)
- [evaluating a trained model](../how-to/evaluate.md) and [running hyperparameter sweeps](../how-to/sweeps.md)
- [adding models](../how-to/add-a-model.md) and [processors](../how-to/add-a-processor.md)

The [API reference](../api/index.md) provides module, class, and function-level documentation for users extending CryoCast in Python.

## Notebooks and interpretation

The [notebooks guide](notebooks.md) describes the demonstrator and research notebooks. Check each notebook's own data, credential and checkpoint prerequisites before execution. Use the [metrics guide](metrics.md) to interpret forecasts alongside the [loss-functions guide](loss-functions.md) for training objectives. CryoCast remains a research system; consult the [model card](https://github.com/alan-turing-institute/cryocast/blob/main/docs/MODEL_CARD.md) for intended use and limitations.
