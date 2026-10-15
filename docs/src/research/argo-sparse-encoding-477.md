# Argo sparse encoding investigation for issue #477

> **Held-out Argo reconstruction is an information-preservation proxy only. It is not evidence of improved sea-ice forecasting.**

This note records the research benchmark for [CryoCast issue #477](https://github.com/alan-turing-institute/cryocast/issues/477). It does not change production Argo ingestion or forecasting behaviour.

## Current CryoCast Argo interpolation

The production `ArgoSource` fetches raw Argo observations in a four-hour window centred on each requested time and in the upper 50 m. It computes pairwise haversine distances from observations to the configured output grid and uses Gaussian weights

`exp(-0.5 * (distance_km / 2000 km)^2)`

with a minimum weight of `1e-10`, then forms weighted averages of the requested variables. The benchmark uses the same Gaussian rule as its non-learned control, but evaluates reconstruction at held-out Argo locations rather than changing the production grid path.

## Research question

Can a direct sparse Argo encoder preserve TEMP and PSAL information at least as well as the current Gaussian interpolation when both learned candidates are trained and evaluated under the same protocol?

The candidates are:

- **SetConv**: a 2000 km Gaussian aggregation over moving observation coordinates followed by a learnable pointwise projection to a 16-channel latent representation.
- **Cross-attention**: a Perceiver-style sparse cross-attention encoder with spherical Fourier position features and a 16-channel output latent representation.

Both use the same `Linear(16, 2)` reconstruction readout, initialised with the same readout seed for a given training seed.

## Real-Argo benchmark protocol

The final benchmark was run in GitHub Actions run `32969345369`, artifact `argo-sparse-benchmark`, on CPU. Two seed-specific JSON files were produced so variation across training seeds can be measured directly rather than inferred from pooled records.

Training dates:

- 2024-06-17 12:00
- 2024-06-24 12:00
- 2024-07-01 12:00
- 2024-07-08 12:00
- 2024-07-15 12:00

Evaluation dates:

- 2024-07-22 12:00
- 2024-07-29 12:00
- 2024-08-05 12:00

The training and evaluation dates are disjoint. Every date was capped deterministically at 256 usable observations. The retained benchmark group counts were 42, 52, 48, 51 and 50 for the five training dates, and 45, 45 and 46 for the three evaluation dates.

Other fixed settings:

- training seeds: 477 and 478
- evaluation seed: 477
- evaluation repeats: 2 per date
- retention: 100%, 75%, 50% and 25%
- reconstruction holdout: 20%
- epochs: 20
- optimiser: AdamW
- learning rate: 0.001
- weight decay: 0
- region: 90/-180/0/180
- depth: 0-50 m
- time window: +/-2 hours

At benchmark time, `PLATFORM_NUMBER` and `CYCLE_NUMBER` were retained as metadata and used as the grouped split key. This keeps all pressure levels from the same float cycle together and prevents those rows crossing the observed/held-out boundary. Shaerdan's later raw-data schema clarified that the exact Argo profile identity is `(PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION)`. The benchmark key is therefore conservative rather than leakage-prone: if both ascending and descending casts exist for one cycle, they are kept on the same side instead of being split, although the reported group counts can undercount true directional profiles and the holdout granularity is slightly coarser.

Evaluation cases are constructed once and reused by Gaussian interpolation, SetConv and cross-attention. Retention masks are generated once per fixed evaluation split and reused by every method. The same deterministic split schedule is used for SetConv and cross-attention training at a given seed.

TEMP and PSAL normalisation is fitted only from the five training-date observation sets. The two seed runs produced identical statistics:

| Variable | Training mean | Training scale |
|---|---:|---:|
| TEMP | 19.263551 | 8.615788 |
| PSAL | 31.506054 | 9.365001 |

Learned predictions are denormalised before MAE and RMSE are calculated, so the reported reconstruction errors are in the native physical Argo variable units. Gaussian interpolation operates directly in those units.

The live benchmark fetch exposed the following potentially relevant metadata/QC columns: `CYCLE_NUMBER`, `DATA_MODE`, `DIRECTION`, `PLATFORM_NUMBER`, `POSITION_QC`, `PRES`, `PRES_QC`, `PSAL_QC`, `TEMP_QC`, `TIME`, and `TIME_QC`. They were reported but were not model inputs.

## Benchmark results

Each learned seed-level number below is the mean over all three evaluation dates and both fixed repeats. The `+/-` term is the population standard deviation across the two training-seed means, 477 and 478. Gaussian interpolation has no training seed; it was evaluated in both seed-specific runs on the same fixed cases, producing identical physical metrics. Its small runtime variation is ordinary repeated CPU timing noise.

TEMP errors are in Argo temperature units and PSAL errors are in practical salinity units.

| Retention | Method | TEMP MAE | TEMP RMSE | PSAL MAE | PSAL RMSE | Inference ms |
|---:|---|---:|---:|---:|---:|---:|
| 100% | Gaussian interpolation | 3.580 +/- 0.000 | 4.756 +/- 0.000 | 1.584 +/- 0.000 | 2.871 +/- 0.000 | 0.477 +/- 0.004 |
| 100% | SetConv | 7.113 +/- 0.108 | 7.847 +/- 0.115 | 5.268 +/- 0.734 | 5.993 +/- 0.587 | 0.592 +/- 0.019 |
| 100% | Cross-attention | 3.791 +/- 0.437 | 4.621 +/- 0.449 | 3.608 +/- 0.799 | 4.413 +/- 0.811 | 1.618 +/- 0.025 |
| 75% | Gaussian interpolation | 3.559 +/- 0.000 | 4.700 +/- 0.000 | 1.533 +/- 0.000 | 2.781 +/- 0.000 | 0.373 +/- 0.002 |
| 75% | SetConv | 7.094 +/- 0.100 | 7.826 +/- 0.112 | 5.224 +/- 0.595 | 5.957 +/- 0.471 | 0.550 +/- 0.006 |
| 75% | Cross-attention | 3.810 +/- 0.430 | 4.639 +/- 0.450 | 3.606 +/- 0.800 | 4.411 +/- 0.836 | 1.545 +/- 0.009 |
| 50% | Gaussian interpolation | 3.625 +/- 0.000 | 4.828 +/- 0.000 | 1.669 +/- 0.000 | 2.977 +/- 0.000 | 0.278 +/- 0.004 |
| 50% | SetConv | 7.153 +/- 0.083 | 7.891 +/- 0.089 | 5.108 +/- 0.505 | 5.825 +/- 0.412 | 0.516 +/- 0.006 |
| 50% | Cross-attention | 3.772 +/- 0.429 | 4.606 +/- 0.442 | 3.606 +/- 0.777 | 4.414 +/- 0.797 | 1.498 +/- 0.023 |
| 25% | Gaussian interpolation | 3.707 +/- 0.000 | 4.961 +/- 0.000 | 1.481 +/- 0.000 | 2.732 +/- 0.000 | 0.185 +/- 0.001 |
| 25% | SetConv | 7.124 +/- 0.037 | 7.881 +/- 0.036 | 4.936 +/- 0.238 | 5.685 +/- 0.201 | 0.489 +/- 0.003 |
| 25% | Cross-attention | 3.720 +/- 0.300 | 4.527 +/- 0.305 | 3.671 +/- 0.738 | 4.471 +/- 0.777 | 1.461 +/- 0.016 |

The seed-specific JSON files also retain the within-seed standard deviation across the six fixed evaluation cases. Those case-level distributions are deliberately not collapsed into the seed-variation term above.

## Runtime, memory and parameters

| Method | Training time s | Trainable parameters | Encoder | Shared readout | Peak memory |
|---|---:|---:|---:|---:|---|
| Gaussian interpolation | n/a | 0 | 0 | 0 | n/a |
| SetConv | 0.217 +/- 0.003 | 338 | 304 | 34 | unavailable on CPU |
| Cross-attention | 0.533 +/- 0.003 | 19,202 | 19,168 | 34 | unavailable on CPU |

Training-time variation is the population standard deviation across seeds 477 and 478. These are tiny reconstruction-probe training jobs over five capped samples, not forecast-training costs. Inference times are CPU query-reconstruction timings, not end-to-end Argo ingestion or complete forecast latency. The benchmark memory hook measures CUDA peak allocation only, so CPU runs correctly report memory as unavailable rather than zero.

## Metadata and QC update

After the benchmark completed, Shaerdan posted a raw-data schema and representative sample bundle for issue #477. The important clarifications are:

- one raw row is one measurement at one pressure level; the exact profile key is `(PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION)` and position/time are constant within that profile;
- expert-mode data exposes raw, adjusted, adjusted-error and QC fields for PRES, TEMP and PSAL;
- QC flags use the standard Argo convention: 1 good, 2 probably good, 3 probably bad, 4 bad, 5 changed, 8 interpolated/estimated and 9 missing;
- `POSITION_QC = 8` is especially relevant to sea-ice work because under-ice floats can have interpolated positions between known fixes, so treating flag 8 as simply "bad" would preferentially discard precisely the under-ice observations of interest;
- in Shaerdan's 2015-2025 Southern Ocean reference download, `TEMP_QC = 1` for 99.06% of rows and `PSAL_QC = 1` for 90.34%, while 18.93% of profiles have `POSITION_QC = 8`;
- raw versus adjusted values are a non-trivial choice for salinity, and the `*_ADJUSTED_ERROR` fields provide per-observation uncertainty that could be used for filtering or weighting.

This late schema update changes the interpretation and the design of the next experiment, but it does not invalidate the completed symmetric reconstruction comparison. All three benchmark methods saw the same capped raw observations and the same held-out cases, and the benchmark's `(PLATFORM_NUMBER, CYCLE_NUMBER)` grouping is conservative with respect to the newly clarified directional profile key. The benchmark was not rerun solely to introduce a new QC/calibration policy because doing so would mix a representation comparison with a separate data-quality decision.

The benchmark results should therefore be read as **raw finite-value reconstruction results, not as QC-optimised or calibration-optimised Argo results**.

## Interpretation

The evidence does not support SetConv. It is materially worse than Gaussian interpolation on both TEMP and PSAL at every retention level.

Cross-attention is closer on TEMP and sometimes has a lower mean TEMP RMSE than Gaussian interpolation, especially at lower retention. However, it is substantially worse on PSAL at every retention level and shows large seed dependence. For example, at 100% retention cross-attention TEMP MAE is 3.354 for seed 477 but 4.228 for seed 478, while PSAL MAE is 2.809 and 4.408 respectively. A two-seed result with this instability is not a defensible architecture win.

Gaussian interpolation remains clearly stronger for PSAL and is much cheaper. The direct sparse candidates have therefore not demonstrated robust reconstruction superiority over the current control.

## Recommendation

**Decision C: neither direct sparse encoder is sufficiently supported by this reconstruction benchmark.**

Do not replace production Argo interpolation on this evidence. Cross-attention is the only candidate worth carrying into a forecast-level falsification experiment because it is much stronger than SetConv and is near the Gaussian control on TEMP, but that is an exploratory choice, not a recommendation to integrate it.

## Limitations

- Held-out reconstruction is not sea-ice forecast skill and cannot establish downstream utility.
- The benchmark uses five training dates, three evaluation dates and only two training seeds.
- Every date is capped at 256 observations, so it is not a throughput benchmark for the full raw archive.
- The date range is narrow and does not test seasonal or interannual generalisation.
- The benchmark grouped by `(PLATFORM_NUMBER, CYCLE_NUMBER)`, while the later schema clarifies that the exact profile identity also includes `DIRECTION`. This grouping is conservative for leakage but can merge ascending and descending casts and undercount true profiles.
- The benchmark predates the explicit QC/calibration schema and uses finite raw TEMP/PSAL values without a new QC or adjusted-value selection policy. This is symmetric across methods but is not an optimised Argo data-quality protocol.
- `POSITION_QC = 8` must not be discarded mechanically in future sea-ice experiments because it marks interpolated positions that are common for under-ice profiles.
- SetConv and cross-attention share the same readout but not the same encoder capacity. Cross-attention has 19,202 total trainable parameters versus 338 for SetConv.
- CPU inference timing does not represent GPU throughput or full production latency, and CPU peak memory is not measured by the CUDA hook.
- The Gaussian control contains a strong 2000 km spatial prior. The learned models have received only a small reconstruction training budget and have not been hyperparameter-tuned.

## Proposed downstream forecast experiment

No production integration should be implemented before a separate forecast experiment.

Use three arms:

1. **No Argo**: preserve the same downstream forecast architecture and latent interface, but replace the Argo contribution with a fixed missing/masked representation.
2. **Current interpolated Argo**: use the existing production Gaussian Argo path unchanged.
3. **Exploratory direct sparse Argo**: use cross-attention as the direct sparse arm for falsification only. It is chosen because SetConv is clearly weak and cross-attention is the only candidate close to the control on TEMP, not because this benchmark selected it as a winner.

Hold constant across all three arms:

- the same forecast core, processor and decoder architecture
- one pre-declared training/validation/test date split
- the same random seeds across arms, with at least 477, 478 and one additional pre-declared seed because the reconstruction benchmark exposed seed sensitivity
- identical optimiser, learning-rate schedule, loss, batch size, regularisation and training budget
- identical non-Argo inputs and preprocessing
- identical forecast leads and evaluation dates
- identical checkpoint-selection rule and stopping criterion

For the primary representation test, keep the underlying Argo value/QC policy the same between the interpolated and direct-sparse arms so that only the representation path differs. Then, if useful, run a separate pre-declared Argo data-quality ablation in which adjusted values, QC filtering, adjusted-error weighting and position-quality handling are applied symmetrically to both Argo arms. This avoids attributing a QC/calibration improvement to the encoder architecture.

Report the existing CryoCast forecast metrics per lead for every seed, together with training/inference cost. A direct sparse encoder should proceed only if its forecast improvement over current interpolated Argo is consistent across seeds and leads and is larger than run-to-run variation.
