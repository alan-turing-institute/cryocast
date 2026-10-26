"""Render a concise, source-backed comparison from independently audited results."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)
LEADS = (1, 7, 14)
ARMS = {"huber": "Huber", "weighted": "Uncertainty-weighted Huber"}


def stats(
    rows: list[dict[str, Any]], selection: str, arm: str, lead: int, key: str
) -> tuple[float, float]:
    """Return mean and sample SD across seeds, never a confidence interval."""
    values = [
        r[key]
        for r in rows
        if r["selection"] == selection and r["arm"] == arm and r["lead_days"] == lead
    ]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def save_figure(fig: Figure, path: Path) -> None:
    """Write whitespace-clean SVG and a PNG used to inspect the exact plot."""
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".svg"))
    target = path.with_suffix(".svg")
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def figures(data: dict[str, Any], out: Path) -> None:
    """Show both error metrics, matched persistence and the full selection history."""
    root = out / "figures"
    root.mkdir(parents=True)
    rows = data["metrics"]
    for key, label in (("rmse_pp", "RMSE"), ("mae_pp", "MAE")):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for arm, name in ARMS.items():
            estimates = [stats(rows, "selected", arm, d, key) for d in LEADS]
            mean, sd = zip(*estimates, strict=True)
            ax.errorbar(
                LEADS,
                mean,
                yerr=sd,
                marker="o" if arm == "huber" else "s",
                capsize=4,
                label=name,
            )
        baseline = [
            next(r["persistence_" + key] for r in rows if r["lead_days"] == d)
            for d in LEADS
        ]
        ax.plot(LEADS, baseline, "--^", label="Matched persistence")
        ax.set(
            xlabel="Forecast lead (days)",
            ylabel=f"{label} (percentage points of SIC)",
            title="Matched forecasts with and without uncertainty weighting",
            xticks=LEADS,
            ylim=(0, None),
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.text(
            0.5,
            0.005,
            "15 February 2024 origins; error bars are SD over 5 seeds, not confidence intervals",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        save_figure(fig, root / key)
    ymax = (
        max(
            h["validation_mse"]
            for r in data["frozen_checkpoints"]
            for h in r["history"]
        )
        * 1.08
    )
    for arm, name in ARMS.items():
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for r in data["frozen_checkpoints"]:
            if r["arm"] == arm:
                ax.plot(
                    [h["epoch"] for h in r["history"]],
                    [h["validation_mse"] for h in r["history"]],
                    label=f"Seed {r['seed']}",
                )
        ax.set(
            title=name + " validation history",
            xlabel="Completed training epochs",
            ylabel="Unweighted January validation MSE (fraction squared)",
            ylim=(0, ymax),
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        save_figure(fig, root / ("validation_" + arm))


def result_table(rows: list[dict[str, Any]], selection: str) -> str:
    """Format every arm/lead combination, without choosing favourable metrics."""
    lines = [
        "| Method | Lead (days) | MAE (pp) | RMSE (pp) | MSE skill vs persistence |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in ARMS.items():
        for d in LEADS:
            mae, msd = stats(rows, selection, arm, d, "mae_pp")
            rmse, rsd = stats(rows, selection, arm, d, "rmse_pp")
            skill, _ = stats(rows, selection, arm, d, "mse_skill")
            lines.append(
                f"| {label} | {d} | {mae:.3f} ± {msd:.3f} | {rmse:.3f} ± {rsd:.3f} | {100 * skill:+.2f}% |"
            )
    for d in LEADS:
        r = next(r for r in rows if r["lead_days"] == d)
        lines.append(
            f"| Persistence | {d} | {r['persistence_mae_pp']:.3f} | {r['persistence_rmse_pp']:.3f} | 0.00% |"
        )
    return "\n".join(lines)


def render(data: dict[str, Any]) -> str:
    """Generate the report from audited metrics and explicit protocol limitations."""
    rows = data["metrics"]
    comparisons = []
    for d in LEADS:
        h = stats(rows, "selected", "huber", d, "rmse_pp")[0]
        w = stats(rows, "selected", "weighted", d, "rmse_pp")[0]
        pairs: dict[int, dict[str, float]] = {}
        for r in rows:
            if r["selection"] == "selected" and r["lead_days"] == d:
                pairs.setdefault(r["seed"], {})[r["arm"]] = r["rmse_pp"]
        wins = sum(p["weighted"] < p["huber"] for p in pairs.values())
        ties = sum(p["weighted"] == p["huber"] for p in pairs.values())
        comparisons.append(
            f"- At {d} {'day' if d == 1 else 'days'}: weighted mean RMSE is {abs(100 * (w / h - 1)):.1f}% {'lower' if w < h else 'higher'} than Huber, with {wins}/5 paired-seed wins and {ties}/5 ties."
        )
    selection_rows = [
        "| Seed | Loss | Selected epoch | Updates executed | Best January MSE |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    zero = 0
    for r in data["frozen_checkpoints"]:
        zero += r["best_epoch"] == 0
        selection_rows.append(
            f"| {r['seed']} | {ARMS[r['arm']]} | {r['best_epoch']} | {r['updates']} | {r['best_validation_mse']:.6f} |"
        )
    p = data["protocol"]
    support = data["training_support"]
    return f"""# Uncertainty-weighted Huber on matched forecasts

This retrospective regional experiment compares PR #458's actual
`UncertaintyWeightedLoss` with ordinary `torch.nn.HuberLoss`, changing
**only the training objective**. It uses real OSI SAF observation uncertainties,
not generated uncertainty labels. It supplies exploratory evidence for #303
and #243, without claiming full-release forecasting skill.

## Selected-checkpoint results

Mean ± sample SD across five paired seeds. MAE and RMSE are in percentage
points of sea ice concentration (SIC). MSE skill is computed per seed as
`1 - model_MSE / persistence_MSE`, then averaged. The RMSE mean is the mean
of per-seed RMSE values, not a square root of pooled MSE.

{result_table(rows, "selected")}

{chr(10).join(comparisons)}

The comparison with ordinary Huber and the comparison with persistence are
separate. All absolute errors and both signs of skill are retained. The data
and run count do not support a broad claim that weighting genuinely improves
forecasting accuracy; keep it opt-in pending independent seasonal evaluation.

![Unweighted forecast RMSE](figures/rmse_pp.svg)

![Unweighted forecast MAE](figures/mae_pp.svg)

## Matched protocol

- Five seeds, {p["seeds"]}, with two loss arms each. The same small real
  CryoCast `EncodeProcessDecode` UNet is used: 172,831 parameters, width 8,
  BatchNorm processor, two observed SIC history frames, daily physical-space
  autoregressive rollout and an additive persistence residual. This is a
  reduced-capacity SIC-only model, not the release model or all its inputs.
- Every pair has identical initial weights, model architecture, input tensors,
  training masks, shuffled origin order, Adam settings and compute budget.
  Each arm executes 120 epochs, **1,800 updates**, batch size 8, LR 0.0001,
  weight decay 0.0001, clip gradient norm at 1.0, on CPU with two threads.
  This protocol was frozen before scoring, with no subsequent rerun or tuning
  based on the held-out results.
- Both losses use Huber delta 0.5. Weighted loss applies inverse variance
  `1/max(sigma, 0.01)^2`, normalised by the batch's sum of weights. Nonpositive,
  nonfinite and sigma > 1 are invalid. Both arms use the **same valid-sigma
  training pixels**, so dropping missing uncertainty is not an arm difference.
- Uncertainty is a **future-target loss label only**. It is absent from the
  model input variable list. Input SIC and target SIC use physical fractions;
  sigma is read in physical fractions after CF decoding and division by 100.
- Training is January-April 2023 (118 eligible origins). January 2024 validation
  (16 origins) selects the minimum **unweighted** 1/7/14-day MSE checkpoint,
  checked every five epochs. February 2024 has 15 identical test origins,
  February 1-15, and the same one-, seven-, and fourteen-day targets for both
  arms and persistence. Two history frames end at each issue date; no future
  observations or future sigma enter a forecast.
- Test masks require the training-derived ocean mask plus valid history and
  target SIC; **test uncertainty does not filter or weight the metrics**.
  Persistence repeats the last issue-time observation over precisely those
  truth dates and pixels. All ten checkpoints were frozen before test scoring.

## Weight distribution and checkpoint selection

Training has {support["eligible_pixels"]:,} common valid-sigma pixels out of
{support["valid_sic_pixels"]:,} otherwise-valid SIC pixels across origins.
The global inverse-weight effective sample fraction is
{100 * support["global_weight_effective_sample_fraction"]:.1f}% (diagnostic
`sum(w)^2 / (N sum(w^2))`; the actual loss is normalised within each batch).
This weighting changes which locations dominate the objective; the
uncertainty-weighted training loss must not be compared numerically with
an unweighted Huber loss as though they measured the same objective.

{chr(10).join(selection_rows)}

The identical epoch-zero model is explicitly the persistence anchor and is
eligible for selection; {zero}/10 selected models stayed at that anchor.
These ties are retained. Selection uses the same unweighted criterion for
both arms. Volatile validation curves and the bounded budget do not establish
convergence. Final-budget performance is also shown below, without replacing
or selecting results using the test set.

![Huber validation history](figures/validation_huber.svg)

![Weighted validation history](figures/validation_weighted.svg)

## Fixed-budget final-checkpoint sensitivity

The following scores use epoch 120 for every arm and seed, irrespective of
validation selection. They are a disclosed secondary sensitivity analysis,
not an alternative selected after seeing which outcome was preferable.

{result_table(rows, "last")}

## Scope and limitations

The observed fields and actual `total_standard_uncertainty` come from
[OSI SAF OSI-438](https://osi-saf.eumetsat.int/products/osi-438), AMSR-derived
25 km northern EASE2-grid NetCDFs, DOI 10.15770/EUM_SAF_OSI_0024, produced by
the Norwegian Meteorological Institute with attribution to EUMETSAT OSI SAF
under CC BY 4.0. They are **not SSMIS observations**. The loss accepts the same
physical sigma variable; this experiment does not validate a separate SSMIS
product. Original NetCDFs are retained and individually SHA-256 checked.

The 64x64 crop (rows 232:296, columns 206:270) and its 3,584-pixel training-only
ocean mask are reused from the earlier observed-data work. SIC status/validity
rules, including exclusion of status bit 128, are inherited symmetrically;
sigma adds a common training validity condition. Uncertainty represents the
provider's retrieval error estimate, not complete forecast-target uncertainty
or guaranteed independent Gaussian noise. Inverse-variance weighting of Huber
is a modelling choice, not a proof of statistical optimality.

This short winter holdout was **already explored in prior experiments**.
The 15 origins overlap, pixels are spatially correlated, and five random
initialisations are not five independent datasets. SD bars show seed variation,
not confidence intervals. No p-values, significance or operational skill claims
are made. This is an analysis-time hindcast using a reprocessed product; issue-time
data availability and latency are not simulated. The single-region, SIC-only,
small-model comparison cannot close #243 or justify changing the default loss.

## Source audit and reproduction

The independent auditor reconstructs the actual training inputs and masks,
checks raw uncertainty file hashes, rebuilds truth and persistence from original
observations, verifies frozen weights and paired initialization/order/update
counts, and recomputes all **900 per-case error sums** without importing the
training program's scoring helper. It checks both selected and final checkpoints.

Small records in [evidence](evidence/) contain the full case table, per-seed
metrics, dates, training curves, raw-file hashes and source checksums. Forecast
arrays, original NetCDFs, Anemoi data stores, checkpoint weights and execution
logs remain on external research storage, not in this PR.

The earlier [source-preparation package](https://github.com/sylvesterkaczmarek/icenet-mp/tree/139bd3f5b573fc61bef7d561fafb9beb7f75142c/validation/x0scale-478-expanded)
records the input archive and crop preparation. Reuse the checksum-identified
source bundle; do not substitute differently preprocessed observations.

```bash
PYTHONPATH=. WANDB_MODE=disabled python validation/uncertainty_458/run.py \\
  --source /path/to/x0scale-478-expanded-20261011 --out /path/to/new-experiment
python validation/uncertainty_458/audit.py --source /path/to/source \\
  --experiment /path/to/new-experiment --out /path/to/new-audit
python validation/uncertainty_458/report.py --audit /path/to/new-audit/audit.json \\
  --out /path/to/new-report
```

The frozen runner revision is `{p["source_revision"]}`, runner SHA-256
`{p["script_sha256"]}`; the input bundle SHA-256 is
`{p["source_npz_sha256"]}`. Exact environment versions are recorded in the
protocol. Cross-platform bitwise reproducibility is not promised.

## Defects corrected before the experiment

The data path checked sigma against the filtered model-input variables instead
of the underlying stored variables. This wrongly required sigma to appear among
inputs. It now reads sigma directly from disk as a separate physical-scale target
without exposing it to the model. A real SingleDataset regression covers this.

Half-precision reductions could overflow even on a 64x64 image; nonfinite
parameters were accepted, and excluded NaN targets could contaminate the Huber
loss and gradients through multiplication by zero. Relative bounded weights,
at-least-float32 arithmetic, explicit parameter validation and selection before
loss calculation fix these defects. Uniform-weight objective/gradient equivalence
and double precision are retained. Ordinary Huber defaults remain unchanged.
"""


def main() -> None:
    """Write the report and charts from one complete audit, without rerunning models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        msg = "Use a new report directory."
        raise ValueError(msg)
    data = json.loads(args.audit.read_text())
    args.out.mkdir(parents=True)
    figures(data, args.out)
    (args.out / "README.md").write_text(render(data))
    log.info("REPORT_WRITTEN %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
