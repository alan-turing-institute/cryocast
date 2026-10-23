# Matched full-grid normalization continuation

This experiment continues the six preserved 96-update BatchNorm/GroupNorm models from the earlier PR #594 study. Each receives 192 additional updates on identical paired training windows, reaching 288 total. Model resolution and capacity remain unchanged. It is a continuation experiment with an explicit learning-rate restart, not full-capacity convergence or a from-scratch 288-update result.

## Protocol and source

`run.py` constructs the ordinary CryoCast CNN-UNet-CNN and uses its dataset, forward and loss paths. The original source commit and exact script hashes are recorded in `raw/protocol.json`. Run training against that frozen source revision, not an arbitrary newer main; use the final PR checkout for the audit and report scripts. The historical checkpoints are identified by their existing SHA-256 manifest; loading uses `weights_only=True`. No remote download or key access is required.

Training and evaluation use the existing southern OSI SAF and ERA5 stores. The explicit Anemoi compatibility flag repairs legacy missing-date indexing in memory only. Original data metadata, dates, masks and checkpoint checksums are checked before and after execution.

The primary comparison uses all fixed final 288-update checkpoints. The secondary comparison selects by validation MSE only, allowing the initial 96-update checkpoint to win. Every measured-run selection is frozen before its test scoring. A separate discarded two-update plumbing smoke check reused one already-explored January window beforehand; its scores are not experiment evidence. All fifteen eligible January and fifteen July 2024 test windows are evaluated over fourteen leads, with persistence on identical masks. These overlapping windows are not independent replications.

The previous scheduler had ended at 1e-6. AdamW moments are restored, but the learning rate follows the same preregistered 1e-3 to 1e-5 cosine restart for both arms. Batch size, Huber objective, weight decay, clipping and model definitions stay matched. Saved schedules and array hashes verify the pairing.

## Run the study

Use a fresh output folder on research storage and provide the existing prepared data root and prior-run folder. The latter contains `training-checkpoint-manifest.json` and its six checkpoint paths. Install the repository's normal dependencies first.

```bash
export ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH=.
python -m validation.norm_594_longer.run \
  --data-root /path/to/prepared-data-root \
  --previous /path/to/previous-96step-run \
  --out /path/to/fresh-continuation-output
```

The measured protocol uses Apple MPS, two CPU threads, full 432 x 432 grids and the 1.26-million-parameter model. MPS warns that indexed accumulation in backward is nondeterministic. Fixed seeds and schedules therefore do not promise bitwise-identical retraining on other hardware. `--smoke` runs one two-update check in a separate fresh directory and must not be used as reported experiment evidence.

## Audit raw predictions

Verify the mask snapshot against training-only status flags first. The stored masks in this experiment are exactly reproduced without using held-out years.

```bash
python -m validation.norm_594_longer.mask_provenance \
  --data-root /path/to/prepared-data-root \
  --out /path/to/continuation-output/mask-provenance-audit.json
```


```bash
python -m validation.norm_594_longer.audit \
  --run /path/to/continuation-output \
  --previous /path/to/previous-96step-run
```

The audit independently recomputes every reported per-origin/per-lead record from retained full-resolution forecasts, without importing the runner's metric function. It checks common masks, finite predictions, pixel-weighted errors, absolute per-window extent errors, calendar separation, paired training arrays, update counts and validation selection. It also compares the replayed 96-update January results with the earlier report, allowing only declared float32-versus-float64 reduction tolerance.

Weights and full-resolution forecast arrays remain on external research storage. The public `raw/` folder contains protocol, schedules, checksums, per-case sufficient statistics, aggregated tables and the completed audit receipt. It permits exact recalculation of the published tables; independent re-scoring of fields requires the retained local forecasts or a new execution.

## Rebuild tables and figures

```bash
python -m validation.norm_594_longer.report \
  --run validation/norm_594_longer/raw \
  --repository . \
  --previews /path/to/plot-previews
```

The same command accepts the original run folder. It produces the Markdown research note, per-lead SVGs, PNG previews and machine-readable tables. It refuses incomplete or unaudited experiment evidence. The reported seed standard deviations are descriptive variation across three runs, not confidence intervals or significance tests.

No production model defaults change. The original 20-update full-size and 96-update reduced-size experiments remain separate historical records. Argo, alternative encoder/decoder families, full-capacity convergence and broader multi-year generalization remain outside this experiment.
