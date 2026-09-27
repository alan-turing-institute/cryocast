# IceNet Multimodal Pipeline

[![Tests](https://github.com/alan-turing-institute/icenet-mp/actions/workflows/test_code.yaml/badge.svg)](https://github.com/alan-turing-institute/icenet-mp/actions/workflows/test_code.yaml)
[![Docs](https://github.com/alan-turing-institute/icenet-mp/actions/workflows/build_docs.yml/badge.svg)](https://github.com/alan-turing-institute/icenet-mp/actions/workflows/build_docs.yml)
[![Code style](https://github.com/alan-turing-institute/icenet-mp/actions/workflows/code_style.yaml/badge.svg)](https://github.com/alan-turing-institute/icenet-mp/actions/workflows/code_style.yaml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

IceNet-MP is a **multimodal machine-learning framework for sea-ice forecasting**. It combines satellite observations, Argo float sensor data, and ERA5 reanalysis fields to produce short-term Arctic and Antarctic sea-ice concentration forecasts.

Example forecasts are shown below.

**Arctic**

![Example IceNet-MP Arctic sea ice concentration forecast compared with observations](docs/src/assets/prediction-north-unet-v2026.07.png)

**Antarctic**

![Example IceNet-MP Antarctic sea ice concentration forecast compared with observations](docs/src/assets/prediction-fullsouth-ddpm-v2026.07.png)

The encode-process-decode architecture translates each input dataset into a shared latent space, allowing new data sources and model components to be added without changing the full pipeline.

## Quick start

```bash
git clone git@github.com:alan-turing-institute/icenet-mp.git
cd icenet-mp
uv sync --managed-python
```

Create a local config in `icenet_mp/config/` (see [Configuration](https://alan-turing-institute.github.io/icenet-mp/user-guide/configuration/) for details):

```yaml
# icenet_mp/config/my.local.yaml
defaults:
  - base
  - _self_

base_path: /path/to/my/data
```

Then download datasets and train:

```bash
uv run imp datasets create --config-name my.local
uv run imp train --config-name my.local
```

Evaluate a checkpoint:

```bash
uv run imp evaluate --checkpoint /path/to/checkpoint.ckpt --config-name my.local
```

## Documentation

- [Installation](https://alan-turing-institute.github.io/icenet-mp/user-guide/installation/) — prerequisites, `uv` setup, HPC-specific steps
- [Configuration](https://alan-turing-institute.github.io/icenet-mp/user-guide/configuration/) — local config files, model overrides, custom datasets
- [Commands](https://alan-turing-institute.github.io/icenet-mp/user-guide/commands/) — `datasets create`, `datasets inspect`, `train`, `evaluate`
- [Add a model](https://alan-turing-institute.github.io/icenet-mp/how-to/add-a-model/) — tensor format, standalone vs. processor model architectures

## Jupyter notebooks

The `notebooks/` folder contains demonstrator notebooks. Run them with:

```bash
uv run --group notebooks jupyter notebook
```

Start with `notebooks/demo_pipeline.ipynb` for a worked example of the full pipeline.
