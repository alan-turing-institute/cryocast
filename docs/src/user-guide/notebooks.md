# Notebooks

The [`notebooks`](https://github.com/alan-turing-institute/icenet-mp/tree/main/notebooks) directory contains supplementary examples and research artifacts for IceNet-MP. They are not required to use the package.

## Setup

Start with the IceNet-MP CLI `demo_pipeline.ipynb` notebook, using the `notebooks` environment:

```bash
uv sync --group notebooks
cd notebooks
uv run jupyter lab
```

Some of these notebooks will not run immediately as they require you to set up credentials for downloading datasets. Details can be found in the notebooks themselves.

## Main notebook set

| Notebook | Purpose | Prerequisite |
| --- | --- | --- |
| [`demo_pipeline.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/demo_pipeline.ipynb) | **CLI walk through.** IceNet-MP walkthrough covering account-free synthetic data, training/evaluation artifacts, model architecture and persistence, Hydra configuration, multimodality, and an optional real-data route. | The default route uses generated synthetic data and local-file logging; there is also an option to use CDS/W&B for real data. |
| [`layer_diagnostics.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/layer_diagnostics.ipynb) | **Model exploration.** Activation-capture investigation for the current UNet/`quick_test` model and multimodal real-data path. | Supply a compatible checkpoint and existing real datasets; the notebook uses local-file logging for evaluation. |
| [`ARGO_data.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/ARGO_data.ipynb) | **Data exploration.** Download, inspect and grid ARGO float observations for the non-gridded data path. | Independent of the demo pipeline; requires network access and its geospatial/data dependencies. |
| [`case_study_whale_corridors.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/case_study_whale_corridors.ipynb) | **Demo videos.** Produce whale-corridor and shipping visualisations. | Requires the case-study data/configuration and is not part of the default CLI walkthrough. |

## Optional synthetic non-gridded workflow

The following pair can be used to create a synthetic non-gridded dataset, using [`nongriddedenv.yaml`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/nongriddedenv.yaml):

1. [`extract_anomalies.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/extract_anomalies.ipynb) creates gridded ERA5 pressure anomalies.
2. [`degrid_and_visualise.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/degrid_and_visualise.ipynb) samples those anomalies into synthetic station and buoy observations.

## Maintenance

When adding, removing, or renaming a notebook, update this page. The notebooks are supplementary and are not part of the automated test suite.
