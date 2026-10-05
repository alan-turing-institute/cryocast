# CryoCast - TRL2 - 2026-05-07

## Versioning and Reference Information

### Card Authors
Sophie Arana, Isabel Fenton, Maria Novitasari, Erin Quan, James Robinson, Shaerdan Shataer, Louisa van Zeeland


### Model/System Name
IceNet Multimodal Pipeline (shorthand: CryoCast)

### Version
Release 2026.07

### TRL
TRL 2

### Model/System Release Date
tbd

### TRL Date
2026-05-07 (draft)

### TRL Card Date
2026-05-07 (draft)

### Lead
Louisa van Zeeland

### Contact Information
Louisa van Zeeland (lvanzeeland@turing.ac.uk) or seaice@turing.ac.uk (this will send an email to all team members)

### Access to Products
Code for the forecasting pipeline is open source and available on [GitHub](https://github.com/alan-turing-institute/cryocast
).

There are not live forecast products being published alongside the code at this stage.

### Licence
MIT Licensed codebase

### References

### Citation
The Alan Turing Institute. (2026). IceNet Multimodal Pipeline [Source code]. Github. https://github.com/alan-turing-institute/cryocast

## Model/System Details

### Description
CryoCast is a multimodal forecasting system for Arctic and Antarctic sea ice, developed at the Alan Turing Institute. CryoCast integrates satellite imagery, reanalysis data, and point-based data (in-situ sensor data) to deliver reliable probabilistic predictions across short-term timescales. It is built around an encode-process-decode architecture, where dataset-specific encoders project each input into a shared latent space, a central processor operates on the combined representation, and output-specific decoders map back to the target resolution. This design makes it easy to add new input sources or prediction targets without restructuring the whole model. The codebase supports several model configurations, including a lightweight default setup suitable for quick tests, a CNN-based encoder-decoder variant wrapping a UNet processor, a vision transformer, a diffusion model variant and a Persistence baseline for benchmarking.

### Intended Uses
This model is intended for research and exploratory inference using historical or real-time climate and observational inputs to generate sea ice concentration forecasts. Model performance is evaluated across multiple forecast lead times, with training focused on short-range horizons. Training, fine-tuning, and evaluation are all supported through the provided pipeline. Typical direct uses include benchmarking against persistence and dynamical model baselines, sensitivity experiments such as varying input variables or data sources, and case-study analysis of notable sea ice events.

### Out-of-Scope Uses
This stage of the codebase is not intended for operational forecasting, safety-critical decision making or issuing public warnings.

### Time Lag
- Short-term forecasting (operational/near-term predictions)
- With particular focus on critical zones like the **sea ice edge** and **marginal ice zone**.

### Spatial Domain
Global (Arctic and Antarctic)

### Temporal / Seasonal Domain
Currently dependent on OSI-SAF SSMIS sensor, available 1978–2025.

###  Approaches to Uncertainty & Variability
Currently no default uncertainty quantification. The DDPM architecture supports epistemic uncertainty estimation through multiple sampling runs, producing a spread of plausible SIC forecasts rather than a single deterministic prediction.

## Training, Testing, Validation Datasets and Procedures

### Input Information
- **Data sources:** OSI-SAF L4 derived SIC product and ERA5 reanalysis data, processed and normalised using Anemoi dataset tooling
- **Data leakage:** Strict temporal separation enforced, with test years held out exclusively for benchmarking and never exposed during development
- **Input uncertainty:** Missing data identified and handled via Anemoi's built-in inspection tooling;

_tbd input data table_

### Output Sea Ice Variables
SIC in % covered by ice per grid cell, with 0% being open water and 100% being full ice cover

### Output Information
Currently numpy arrays but output is never saved (only used for evaluation).

## Evaluation Details

### Evaluation Statement
- Evaluation metrics: `sieerror (mean)`, `rmse (mean)`, `mae (mean)`, threshold of 15% sea ice for ice edge
- Model comparison: simple UNet, persistence
- key findings: tbd

### Rationale for Current TRL
The CryoCast system has made substantial progress through TRL 2, with optimised code, unit tests, an initial curated datasets for in-situ data, and a basic software architecture established. Notably, several TRL 3 coding checkpoints have already been completed, including modular/reusable code structures and integration tests for module robustness. To complete TRL 2, preliminary benchmarking and evaluation results will be written up for publication.

### On-going Progress Towards Next TRL
To complete TRL 2, there are two outstanding requirements: formal documentation of baseline model performance metrics and fully validated focused experiments confirming model behaviours and goals. For progress towards TRL 3, the sea ice team is actively looking for partners to narrow initial use cases for the CryoCast pipeline and develop tests that are grounded in real world use.

### Ethical Considerations
The IceNet Multimodal Pipeline is a research product and currently operates on publicly accessible data so there are no known ethical considerations.
