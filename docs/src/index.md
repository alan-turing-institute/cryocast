# CryoCast

**CryoCast** is a multimodal machine-learning framework for short-term Arctic and Antarctic sea-ice forecasting. It brings satellite observations, in-situ sensor data, and reanalysis fields into a common pipeline for sea-ice concentration forecasting.

CryoCast uses an encode-process-decode architecture in which dataset-specific encoders map inputs into a shared latent representation, a central processor combines that information, and decoders produce forecast outputs. This design supports adding new input sources, prediction targets, and model components without restructuring the full pipeline.

## Forecast examples

The examples below show Arctic and Antarctic sea-ice concentration forecasts compared with observations.

### Arctic

![Example CryoCast Arctic sea ice concentration forecast compared with observations](assets/prediction-fullnorth-ddpm-v2026.07.png)

### Antarctic

![Example CryoCast Antarctic sea ice concentration forecast compared with observations](assets/prediction-fullsouth-ddpm-v2026.07.png)

## Capabilities

- Multimodal data fusion across observational and reanalysis datasets
- Extensible inputs and prediction targets through the encode-process-decode design
- Multiple model configurations, including UNet, vision transformer, diffusion, and persistence-baseline approaches
- Training, evaluation, benchmarking, sensitivity experiments, and case-study analysis through the same pipeline

## Getting started

- [User Guide](user-guide/index.md) for installation, configuration, data preparation, commands, metrics, and notebooks
- [How-to guides](how-to/index.md) for model development, training, evaluation, sweeps, and release workflows
- [API Reference](api/index.md) for public modules, classes, and functions
- [Model card](https://github.com/alan-turing-institute/cryocast/blob/main/docs/MODEL_CARD.md) for intended use, scope, and current project maturity

## Quick install

```bash
pip install git+https://github.com/alan-turing-institute/cryocast
```

Then run:

```bash
cryocast --help
```

For a fuller setup, continue with the [installation](user-guide/installation.md), [configuration](user-guide/configuration.md), [data](user-guide/data.md), and [commands](user-guide/commands.md) guides.

## Research scope

CryoCast is developed at The Alan Turing Institute as a research system. It is intended for research and exploratory inference, including benchmarking, sensitivity experiments, and case-study analysis. The current project scope does not include operational forecasting, safety-critical decision making, or issuing public warnings. See the [model card](https://github.com/alan-turing-institute/cryocast/blob/main/docs/MODEL_CARD.md) for the detailed intended-use statement.
