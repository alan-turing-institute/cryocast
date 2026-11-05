# CARRA2 evaluation protocol for issue #327

This note defines the controlled comparison used to decide how CARRA2 should be used in
CryoCast. It does not treat CARRA2 as a drop-in replacement for ERA5: CARRA2 is
pan-Arctic, while the current ERA5 configurations are global and support both
hemispheres.

The first evaluation target is therefore the role already proposed in issue #327:
**CARRA2 as high-resolution Arctic supervision for downscaling**, while retaining ERA5
as the general-purpose weather input.

## Dependencies

The reconciled merge order is #457, #552, #559, then #595. This keeps both CARRA2
and generic CDS source registrations available while building the downscaling stack:

- #457 provides the direct CARRA2 source and weather sample.
- #552 defines the reproducible Svalbard CARRA2 2.5 km target region.
- #559 adds the trainable residual downscaler and its geographic interpolation
  baseline.

The evaluation branch should not be merged independently of those capabilities.

## Controlled comparison

For every held-out test sample, compare:

1. **Geographic interpolation** from the same 25 km low-resolution SIC field to the
   CARRA2 target grid.
2. **Learned residual downscaling** from exactly the same low-resolution field.
3. **CARRA2 target** on the common valid cells.

Both methods therefore use identical source dates, source values, target dates,
target grid, and validity mask. This isolates whether the learned downscaler recovers
useful high-resolution structure beyond smooth interpolation.

Run the comparison with:

~~~bash
uv run cryocast evaluate-downscaling --config-name downscaling_north \
  --checkpoint /path/to/downscaler.ckpt \
  --output carra2-comparison.json
~~~

For quick smoke checks, add `--max-batches N`. Full conclusions must use the complete
pre-declared test period.

The command reports:

- MAE and RMSE on valid CARRA2 cells;
- finite-difference gradient RMSE, to penalise incorrect local spatial structure;
- the fraction of spectral power in the high-frequency band;
- the absolute error of that high-frequency fraction relative to CARRA2;
- percentage improvement of the learned model over interpolation for each error
  quantity.

The spectral calculation de-means each frame and applies the same target validity mask
to target, interpolation, and learned fields before the FFT. Protocol
`paired-hermitian-v2` weights the real-FFT interior columns by two to account for
conjugate frequencies. It averages the absolute prediction/target fraction difference
per matched frame, preventing opposite errors on different dates from cancelling.
The radial cutoff is inclusive and evaluated as squared frequency radius
`fy**2 + fx**2 >= cutoff**2 / 2`. This avoids platform-dependent square-root
rounding at bins exactly on the cutoff; no tolerance widens the selected band.

Every frame with at least two valid target cells is included for both methods;
a spatially constant field is assigned zero high-frequency fraction by convention.
Nonfinite predictions at valid truth are rejected. Scoring uses CPU float64 even when
model inference uses an accelerator. Mask-edge spectral leakage remains shared, so
this is a comparative diagnostic, not an unmasked physical power spectrum.

## Historical real-data pipeline pilot

The numbers in this subsection are the original protocol-v1 record. Its spectral
fractions and fraction errors are superseded by the independently rescored results
below. MAE, RMSE and gradient RMSE are unaffected within numerical tolerance.

A deliberately tiny live-data pilot was run to verify the end-to-end path before
committing to the full experiment. It used actual OSI SAF SIC and CARRA2 downloaded
through the configured ingestion pipeline for 1-7 January 2024. The split was:

- training: 1-4 January;
- validation: 5 January;
- held-out test: 6-7 January;
- seeds: 327, 328, and 329;
- 40-epoch cap with early stopping, with the best validation checkpoint used for test;
- CPU execution.

This pilot is **not** decision-quality evidence: the training set contains only four
days, the validation set one day, and all dates are temporally adjacent. Its purpose is
to validate acquisition, alignment, training, checkpoint reload, and the controlled
evaluation command on real data.

The interpolation baseline on the two held-out days was identical across seeds:
MAE 0.04695, RMSE 0.15708, gradient RMSE 0.02031, and high-frequency power fraction
0.010963 versus 0.010332 in the CARRA2 target.

| Seed | Best checkpoint | Learned MAE | Learned RMSE | MAE vs interpolation | RMSE vs interpolation | Gradient RMSE vs interpolation | HF-fraction error vs interpolation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 327 | epoch 0 | 0.04695 | 0.15708 | 0.0% | 0.0% | 0.0% | 0.0% |
| 328 | epoch 0 | 0.04695 | 0.15708 | 0.0% | 0.0% | 0.0% | 0.0% |
| 329 | epoch 2 | 0.06633 | 0.15232 | -41.28% | +3.03% | -0.37% | -1.57% |

For seeds 327 and 328, validation selected the zero-initialised checkpoint, so the
learned model remained exactly the interpolation baseline. Seed 329 reduced RMSE
slightly but materially worsened MAE and also degraded both spatial-detail diagnostics.
Across the three seeds the mean change was -13.76% for MAE, +1.01% for RMSE, -0.12%
for gradient RMSE, and -0.52% for high-frequency-fraction error, where positive means
improvement over interpolation.

The correct conclusion from this pilot is therefore only that the evaluation path works
and that there is **no early evidence of a robust learned-downscaling advantage**. A
full run over the pre-declared disjoint sample periods and multiple seeds is still
required before deciding whether the CARRA2 downscaling role is useful.

## Decision rule

Do **not** replace ERA5 with CARRA2 based on this experiment.

Use CARRA2 as a high-resolution Arctic downscaling target only if the learned model,
on a disjoint test period and across multiple seeds:

- improves MAE/RMSE over geographic interpolation;
- improves or at least does not materially degrade gradient error;
- moves the high-frequency power fraction closer to the CARRA2 target rather than
  merely increasing noise;
- remains reproducible under the same timestamp, variable, grid, and mask alignment.

If those conditions are not met, keep interpolation as the downscaling baseline and
retain CARRA2 only as an evaluation/reference dataset while investigating a better
downscaler.

A later, separate experiment would be required to test CARRA2 as an additional
meteorological input against ERA5 on the common Arctic domain.


## October 21 validation and corrected scoring

Replayed the three original validation-selected checkpoints on the same two held-out
January 6-7, 2024 dates, with no new fitting or checkpoint selection. Original stored
data and checkpoint hashes remain unchanged. An independent NumPy implementation
using the full complex FFT reproduces the corrected scores, avoiding reuse of the
real-FFT helper being tested. Pointwise and gradient errors reproduce the historical
results within numerical tolerance.

The target high-frequency fraction is **0.01155104**.
Interpolation has MAE **0.04694653**, RMSE **0.15707712**, gradient RMSE
**0.02030974**, high-frequency fraction **0.01237861**,
and mean paired absolute fraction error **0.00082757**.
All methods use **67,230 values**, **131,744 adjacent
edges** and **2 matched spectral frames**.

| Seed | Learned MAE | Learned RMSE | Gradient RMSE | High-frequency fraction | Paired absolute fraction error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 327 | 0.04694653 | 0.15707712 | 0.02030974 | 0.01237861 | 0.00082757 |
| 328 | 0.04694653 | 0.15707712 | 0.02030974 | 0.01237861 | 0.00082757 |
| 329 | 0.06632687 | 0.15232136 | 0.02038497 | 0.01239212 | 0.00084107 |

The corrected evidence still does not demonstrate a consistent learned-downscaling
advantage. The four-day training set, one-day validation set and two adjacent held-out
dates remain insufficient for an adoption decision. Protocol-v1 spectral values must
not be pooled with protocol-v2 values.

A separate fresh CDS request for January 15, 2024 exercised the current source and
crop code: the returned 2880 by 2880 grid produced a 273 by 283 stored-region field,
with 33,615 finite sea-ice cells in [0, 1]. The retained pilot stores also passed current
Anemoi inspection and no-overwrite reuse. A separate one-step optimizer smoke check,
safe checkpoint roundtrip, and three-lead forecast/downscaler composition passed.
That optimizer check was not used to select or report the table above. No forecast
skill improvement, full seasonal run, or new production default is claimed.
