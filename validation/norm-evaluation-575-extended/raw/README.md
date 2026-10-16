# Raw runs for CryoCast issue #575

These six small JSON files are the unmodified outputs of the 96-update,
three-seed, two-normalisation extended experiment. Each records per-lead
SIC RMSE, signed and absolute sea-ice-extent error, the persistence
baseline, runtime, model capacity, source commit and runner checksum.
Run files preserve the exact experiment-produced JSON values; a terminal LF was added to each copy for repository EOF conventions;
see `manifest.json` for SHA-256 checksums.

The comparison uses 1,256,655-parameter CNN-UNet-CNN models, **not**
the full 44,650,383-parameter production architecture, and did not
converge. The 10 holdout windows are a limited January 2024 slice.

Large checkpoint weights and execution logs are not included here.
To regenerate CSVs and plots from these frozen results, from the
repository root:

```bash
CRYOCAST_NORM_OUTPUT_DIR=validation/norm-evaluation-575-extended/raw \
    python validation/norm-evaluation-575-extended/aggregate.py
```

The `aggregate.py` script validates the six-run protocol, source
hashes, parameter counts and identical persistence baselines before
generating its figures.
