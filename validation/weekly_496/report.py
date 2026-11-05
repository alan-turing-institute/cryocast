"""Render the independently audited matched-weekly forecast evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LEADS = (7, 14)
STRIDES = (1, 7)
LABELS = {1: "Daily-trained", 7: "Weekly-trained"}


def values(audit: dict[str, Any], stride: int, lead: int, key: str) -> np.ndarray:
    """Select seed-level estimates without mixing forecast horizons."""
    return np.asarray(
        [
            row[key]
            for row in audit["summary"]
            if row["stride"] == stride and row["lead_days"] == lead
        ]
    )


def save_figure(fig: Figure, output: Path) -> None:
    """Save stable SVG plus a local PNG for visual inspection."""
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    path = output.with_suffix(".svg")
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


def render(audit: dict[str, Any], out: Path) -> None:
    """Show estimates and seed spread, with identical baseline and date scope."""
    out.mkdir(parents=True, exist_ok=True)
    for key, label in (("rmse_pp", "RMSE"), ("mae_pp", "MAE")):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for stride, marker in ((1, "o"), (7, "s")):
            means = [values(audit, stride, day, key).mean() for day in LEADS]
            deviations = [values(audit, stride, day, key).std(ddof=1) for day in LEADS]
            ax.errorbar(
                LEADS,
                means,
                yerr=deviations,
                marker=marker,
                capsize=4,
                label=LABELS[stride],
            )
        baseline = [values(audit, 1, day, "persistence_" + key)[0] for day in LEADS]
        ax.plot(
            LEADS,
            baseline,
            linestyle="--",
            marker="^",
            label="Persistence at the same origin",
        )
        ax.set_xticks(LEADS)
        ax.set_xlim(5, 16)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Forecast lead (days)")
        ax.set_ylabel(label + " (percentage points of SIC)")
        ax.set_title("Matched seven- and fourteen-day forecasts")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        fig.text(
            0.5,
            0.01,
            "15 forecast origins in February 2024; error bars show seed SD, not confidence intervals",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save_figure(fig, out / key)
    ceiling = 1.05 * max(
        p["validation_mse"] for r in audit["training"] for p in r["history"]
    )
    for stride in STRIDES:
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for record in audit["training"]:
            if record["stride"] != stride:
                continue
            ax.plot(
                [p["epoch"] for p in record["history"]],
                [p["validation_mse"] for p in record["history"]],
                marker=".",
                label=f"Seed {record['seed']}",
            )
        ax.set_xlabel("Completed training epochs")
        ax.set_ylabel("January validation MSE (fraction squared)")
        ax.set_title(LABELS[stride] + " validation history")
        ax.set_ylim(0, ceiling)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out / f"validation_stride{stride}")


def write_report(audit: dict[str, Any], path: Path) -> None:
    """Keep limitations adjacent to the measured outcomes."""
    rows = []
    for stride in STRIDES:
        for lead in LEADS:
            metric_text = []
            for key in ("mae_pp", "rmse_pp"):
                data = values(audit, stride, lead, key)
                metric_text.append(f"{data.mean():.3f} ± {data.std(ddof=1):.3f}")
            skill = 100 * values(audit, stride, lead, "mse_skill").mean()
            rows.append(
                f"| {LABELS[stride]} | {lead} | {metric_text[0]} | {metric_text[1]} | {skill:+.2f}% |"
            )
    for lead in LEADS:
        mae = values(audit, 1, lead, "persistence_mae_pp")[0]
        rmse = values(audit, 1, lead, "persistence_rmse_pp")[0]
        rows.append(f"| Persistence | {lead} | {mae:.3f} | {rmse:.3f} | 0.00% |")
    checkpoint_rows = [
        f"| {r['seed']} | {LABELS[r['stride']]} | {r['best_epoch']} | {r['updates']} | {r['train_seconds']:.1f} |"
        for r in audit["training"]
    ]
    text = """# Weekly forecasting on matched dates

This bounded, retrospective experiment exercises PR #496's real `SingleDataset`
and `CombinedDataset` paths and compares daily-trained and weekly-trained
CryoCast models with **the same persistence forecast at the same issue dates,
valid dates and ocean pixels**. It provides exploratory evidence for #243,
not a full-release benchmark or a basis for closing that issue.

## Results

Values are mean ± sample standard deviation across three paired seeds.
MAE and RMSE are percentage points of sea ice concentration (SIC).
MSE skill is `1 - model_MSE / persistence_MSE`, calculated per seed and then
averaged. RMSE means are averages of the three per-seed RMSE values.

| Method | Lead (days) | MAE (pp) | RMSE (pp) | Mean MSE skill vs persistence |
| --- | ---: | ---: | ---: | ---: |
TABLE

Weekly RMSE is modestly lower than persistence at both leads. However,
**weekly MAE is worse than persistence at both leads**, so there is no
metric-independent improvement. At fourteen days, weekly RMSE is lower
than this daily-trained control's RMSE; at seven days they are similar.
These results do not establish universal gains from changing cadence.

**Seed 497's weekly arm selected epoch 0**, the explicitly included
persistence-initialised checkpoint, because none of its trained candidates
improved validation MSE. It is retained as an exact persistence tie, not
omitted or described as learned skill. All six arms still received the
same 840-update training budget.

![Matched RMSE estimates and seed spread](figures/rmse_pp.svg)

![Matched MAE estimates and seed spread](figures/mae_pp.svg)

## Protocol fixed before fitting

- Data: the existing checksum-identified OSI SAF OSI-438 25 km northern
  EASE2 crop, 64 x 64 cells. The 3,584-cell ocean mask came from training
  data. This reuses the February block from the earlier #478 experiment;
  it is not an untouched independent confirmation dataset.
- Model: actual CryoCast `EncodeProcessDecode`, naive linear encoder/decoder,
  eight-channel `UNetProcessor`, BatchNorm, physical-space autoregression,
  and an additive persistence skip. It has 172,831 parameters. A zero
  decoder readout makes the epoch-zero prediction exactly persistence.
  No repository model presets or training defaults were changed.
- Both arms use two history states. Daily spacing reads origin-1 day and
  origin; weekly spacing reads origin-7 days and origin. **Weekly means
  seven-day-spaced daily snapshots, not seven-day averages.**
- Training: 106 common origins from 8 January to 23 April 2023. Each arm
  learns one step of its own cadence. Their target horizons therefore
  differ by design; neither model uses future observations at inference.
- Validation: 10 common origins from 8-17 January 2024; checkpoint selection
  minimises masked MSE jointly at physical leads 7 and 14 days. No test
  scores are used for training, stopping, or checkpoint selection.
- Testing: 15 common origins from 1-15 February 2024, with targets at
  seven and fourteen days, all within February. January observations may
  supply past history for early February origins; they are not test labels.
  Overlapping forecasts are correlated, not 15 independent weekly trials.
- Seeds: 496, 497, 498. Matching seeds share identical initial weights and
  epoch-wise origin order. Both arms train 60 epochs, batch size 8, 840
  Adam updates, learning rate 0.001, weight decay 0.0001 and gradient-norm
  clipping at 1.0. Validation is checked every five epochs, including zero.
- Persistence always repeats the **last observed issue-time field** at both
  horizons. It is never aged back to the first weekly history state. Metrics
  use the identical per-origin/per-lead valid mask for every method, requiring
  valid history and target observations. There are 53,760 pixel comparisons
  per seed and lead.

## Checkpoints and optimisation limits

| Seed | Arm | Selected epoch | Updates trained | Training seconds |
| --- | --- | ---: | ---: | ---: |
CHECKPOINTS

Validation curves are volatile and do not demonstrate convergence. The
study uses one regional crop, one input variable and a short winter test
period, not the project's full encoder/data combinations. The reduced
model and exploratory checkpoint choices must not be presented as full
CryoCast release performance. No confidence intervals, hypothesis tests,
or operational forecast-skill claim are made.

![Daily validation history](figures/validation_stride1.svg)

![Weekly validation history](figures/validation_stride7.svg)

## Verification and reproduction

The independent auditor reconstructs truth, persistence and masks directly
from the original observed-data bundle, recomputes all 180 case-level error
sums without the training runner's metric helper, checks the frozen weights,
and verifies matched initial states and update counts. The full case table,
per-seed metrics, training curves, dates and checksum manifests accompany
this report: [individual cases](evidence/per_case.csv),
[per-seed metrics](evidence/metrics.csv), [forecast dates](evidence/dates.json),
[training histories](evidence/frozen_checkpoints.json), and
[checksums](evidence/SHA256.json). The actual forecasts, model weights,
prepared Anemoi store and complete execution logs remain on external
research storage.

Run from the repository root with the source data from the frozen
[earlier observed-data package](https://github.com/sylvesterkaczmarek/icenet-mp/tree/139bd3f5b573fc61bef7d561fafb9beb7f75142c/validation/x0scale-478-expanded):

```bash
PYTHONPATH=. WANDB_MODE=disabled python validation/weekly_496/run.py \\
  --source /path/to/x0scale-478-expanded-20261011 --out /path/to/new-run
python validation/weekly_496/audit.py --source /path/to/source \\
  --experiment /path/to/new-run --output /path/to/audit
python validation/weekly_496/report.py --audit /path/to/audit/audit.json \\
  --output /path/to/report
```

The source bundle SHA-256 is `SOURCE_HASH`. The frozen runner's revision is
`REVISION`, with script SHA-256 `SCRIPT_HASH`. Results use the exact CPU,
Python and PyTorch environment recorded in `evidence/protocol.json`;
bitwise reproducibility across environments is not promised.

The observed fields are [OSI SAF OSI-438](https://osi-saf.eumetsat.int/products/osi-438),
DOI 10.15770/EUM_SAF_OSI_0024, produced by the Norwegian Meteorological
Institute under CC BY 4.0. SIC values were converted from percentages to
fractions and cropped; no new data were downloaded for this experiment.
The product is reprocessed and nominally delayed, so this is an analysis-time
hindcast: issue-time operational availability and data latency are not simulated.

## PR correction found during validation

The original NetCDF writer used native daily frequency for its `lead_time`
coordinate even when forecast `valid_time` was spaced weekly. PR #496 now
multiplies lead durations by the temporal stride. Tests independently verify
`valid_time = forecast_reference_time + lead_time` for strides 1, 7 and 30;
the daily fallback remains unchanged. The actual configuration key is
`window.step_stride`, not the earlier PR-description wording `predict.step_stride`.
"""
    text = text.replace("TABLE", "\n".join(rows)).replace(
        "CHECKPOINTS", "\n".join(checkpoint_rows)
    )
    protocol = audit["protocol"]
    text = (
        text.replace("SOURCE_HASH", protocol["source_sha256"])
        .replace("REVISION", protocol["source_revision"])
        .replace("SCRIPT_HASH", protocol["script_sha256"])
    )
    path.write_text(text)


def main() -> None:
    """Build the report using the independent audit output only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "passed":
        msg = "An independent audit must pass before figures are generated."
        raise ValueError(msg)
    args.output.mkdir(parents=True, exist_ok=True)
    render(audit, args.output / "figures")
    write_report(audit, args.output / "README.md")


if __name__ == "__main__":
    main()
