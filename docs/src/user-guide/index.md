# User Guide

This guide covers everything you need to get CryoCast running - from installing the package to training a model and visualising results.

If you are using CryoCast for the first time, follow the core pages in order: install the environment, create a local configuration, understand the available datasets, and then use the command-line interface to create data, train, and evaluate models.

## Start here

1. [Installation](installation.md) covers prerequisites, local setup, and additional requirements for supported HPC systems.
2. [Configuration](configuration.md) explains local Hydra configuration, model and dataset overrides, platform settings, and input/target variables.
3. [Data](data.md) describes the OSI SAF, ERA5, and Argo data used by CryoCast, including coverage, variables, and dataset organisation.
4. [Commands](commands.md) documents the `cryocast` CLI for dataset preparation, inspection, training, sweeps, evaluation, and feature importance.

A typical local workflow then follows this pattern:

```bash
uv run cryocast datasets create --config-name <your-config>
uv run cryocast train --config-name <your-config>
uv run cryocast evaluate --checkpoint <path-to-checkpoint> --config-name <your-config>
```

See the individual command and configuration pages for the required options and environment-specific setup.

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

- [training a model](../how-to/train.md)
- [multistage training](../how-to/train-multistage.md)
- [evaluating a trained model](../how-to/evaluate.md)
- [running hyperparameter sweeps](../how-to/sweeps.md)
- [adding models](../how-to/add-a-model.md) and [processors](../how-to/add-a-processor.md)

The [API reference](../api/index.md) provides module, class, and function-level documentation for users extending CryoCast in Python.

## Notebooks and interpretation

The [notebooks guide](notebooks.md) identifies the main demonstrator and research notebooks and explains their prerequisites. For understanding model results, use the [metrics guide](metrics.md), which documents what each forecast metric measures and where its interpretation can be misleading, together with the [loss-functions guide](loss-functions.md) for training-objective behaviour.
