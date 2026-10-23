"""Aggregate independently checked normalization forecasts and render PR evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
mpl.rcParams["svg.hashsalt"] = "cryocast-norm-594-longer"
from matplotlib import pyplot as plt  # noqa: E402 - headless backend selected first

MONTHS = ("2024-01", "2024-07", "combined")
LABELS = {
    "batchnorm": "BatchNorm",
    "groupnorm": "GroupNorm",
    "persistence": "Persistence",
}
SELECTIONS = ("initial", "selected", "final")
SEEDS = (575, 576, 577)
NORMS = ("batchnorm", "groupnorm")
LATE_LEAD_START = 10
EXPECTED_CASES = 7980


def check(condition: object, message: str) -> None:
    """Reject incomplete groups rather than averaging incomparable conditions."""
    if not condition:
        raise ValueError(message)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write all inspected values with stable column ordering."""
    check(rows, "Cannot write an empty evidence table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    """Require standards-compliant finite JSON."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    """Hash a small publishable result file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(root: Path) -> list[dict[str, Any]]:
    """Read original JSON or the exact compact public CSV representation."""
    original = root / "per_case.json"
    if original.exists():
        records: list[dict[str, Any]] = json.loads(original.read_text())
        return records
    records = []
    files = (
        [root / "per-case.csv"]
        if (root / "per-case.csv").exists()
        else sorted(root.glob("per-case-*.csv"))
    )
    check(files, "No public per-case evidence files found")
    for path in files:
        with path.open(newline="") as stream:
            for raw in csv.DictReader(stream):
                row: dict[str, Any] = {}
                for key, value in raw.items():
                    if key == "seed":
                        row[key] = None if value == "" else int(value)
                    elif key in {"lead", "count"}:
                        row[key] = int(value)
                    elif key in {"norm", "selection", "origin", "month"}:
                        row[key] = value
                    else:
                        row[key] = float(value)
                records.append(row)
    return records


def seed_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool error sums within each lead, then average the fourteen lead RMSEs."""
    summaries, daily = [], []
    groups = {(r["seed"], r["norm"], r["selection"]) for r in rows}
    for seed, norm, selection in sorted(groups, key=lambda x: (str(x[0]), x[1], x[2])):
        arm = [
            r
            for r in rows
            if (r["seed"], r["norm"], r["selection"]) == (seed, norm, selection)
        ]
        for month in MONTHS:
            current = (
                arm if month == "combined" else [r for r in arm if r["month"] == month]
            )
            n_windows = 30 if month == "combined" else 15
            check(len(current) == n_windows * 14, "Unmatched temporal support")
            per_lead = []
            for lead in range(1, 15):
                selected = [r for r in current if r["lead"] == lead]
                check(len(selected) == n_windows, "Missing lead/window records")
                count = sum(r["count"] for r in selected)
                value = {
                    "seed": seed,
                    "norm": norm,
                    "selection": selection,
                    "month": month,
                    "lead_days": lead,
                    "windows": n_windows,
                    "valid_cells": count,
                    "rmse": float(
                        np.sqrt(sum(r["squared_sum"] for r in selected) / count)
                    ),
                    "mae": sum(r["absolute_sum"] for r in selected) / count,
                    "mean_absolute_sie_km2": float(
                        np.mean([r["absolute_sie_km2"] for r in selected])
                    ),
                    "mean_signed_sie_km2": float(
                        np.mean([r["signed_sie_km2"] for r in selected])
                    ),
                }
                per_lead.append(value)
                daily.append(value)
            summaries.append(
                {
                    "seed": seed,
                    "norm": norm,
                    "selection": selection,
                    "month": month,
                    "windows": n_windows,
                    "mean_rmse": float(np.mean([v["rmse"] for v in per_lead])),
                    "late_rmse": float(
                        np.mean(
                            [
                                v["rmse"]
                                for v in per_lead
                                if v["lead_days"] >= LATE_LEAD_START
                            ]
                        )
                    ),
                    "mean_mae": float(np.mean([v["mae"] for v in per_lead])),
                    "mean_absolute_sie_km2": float(
                        np.mean([v["mean_absolute_sie_km2"] for v in per_lead])
                    ),
                    "late_absolute_sie_km2": float(
                        np.mean(
                            [
                                v["mean_absolute_sie_km2"]
                                for v in per_lead
                                if v["lead_days"] >= LATE_LEAD_START
                            ]
                        )
                    ),
                }
            )
    return summaries, daily


def cross_seed(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe training-seed variation, without treating windows as independent runs."""
    metrics = (
        "mean_rmse",
        "late_rmse",
        "mean_mae",
        "mean_absolute_sie_km2",
        "late_absolute_sie_km2",
    )
    output = []
    for month in MONTHS:
        for selection in SELECTIONS:
            for norm in (*NORMS, "persistence"):
                group = [
                    r
                    for r in summaries
                    if r["month"] == month
                    and r["norm"] == norm
                    and (r["selection"] == selection or norm == "persistence")
                ]
                check(
                    len(group) == (1 if norm == "persistence" else len(SEEDS)),
                    "Missing seed evidence",
                )
                item = {
                    "month": month,
                    "selection": selection,
                    "norm": norm,
                    "n_seeds": len(group),
                }
                for metric in metrics:
                    values = [r[metric] for r in group]
                    item[metric + "_mean"] = float(np.mean(values))
                    item[metric + "_sd"] = (
                        None if len(values) == 1 else float(np.std(values, ddof=1))
                    )
                output.append(item)
    return output


def paired_comparisons(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose every paired effect instead of reporting only a favorable overall mean."""
    output = []
    for month in MONTHS:
        persistence = next(
            r for r in summaries if r["month"] == month and r["norm"] == "persistence"
        )
        for selection in SELECTIONS:
            for seed in SEEDS:
                pair = {
                    r["norm"]: r
                    for r in summaries
                    if r["month"] == month
                    and r["selection"] == selection
                    and r["seed"] == seed
                }
                check(set(pair) == set(NORMS), "Missing normalization pair")
                bn, gn = pair["batchnorm"], pair["groupnorm"]
                output.append(
                    {
                        "month": month,
                        "selection": selection,
                        "seed": seed,
                        "groupnorm_minus_batchnorm_rmse": gn["mean_rmse"]
                        - bn["mean_rmse"],
                        "groupnorm_rmse_reduction_percent": 100
                        * (bn["mean_rmse"] - gn["mean_rmse"])
                        / bn["mean_rmse"],
                        "batchnorm_minus_persistence_rmse": bn["mean_rmse"]
                        - persistence["mean_rmse"],
                        "groupnorm_minus_persistence_rmse": gn["mean_rmse"]
                        - persistence["mean_rmse"],
                        "groupnorm_minus_batchnorm_sie_km2": gn["mean_absolute_sie_km2"]
                        - bn["mean_absolute_sie_km2"],
                    }
                )
    return output


def training_budget_effects(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare 96 and 288 updates on exactly the same expanded holdout support."""
    output = []
    for month in MONTHS:
        for norm in NORMS:
            for seed in SEEDS:
                pair = {
                    r["selection"]: r
                    for r in summaries
                    if r["month"] == month
                    and r["norm"] == norm
                    and r["seed"] == seed
                    and r["selection"] in {"initial", "final"}
                }
                check(
                    set(pair) == {"initial", "final"},
                    "Missing budget-matched checkpoint",
                )
                first, last = pair["initial"]["mean_rmse"], pair["final"]["mean_rmse"]
                output.append(
                    {
                        "month": month,
                        "norm": norm,
                        "seed": seed,
                        "initial_mean_rmse": first,
                        "final_mean_rmse": last,
                        "final_minus_initial_rmse": last - first,
                    }
                )
    return output


def budget_table(budget: list[dict[str, Any]]) -> str:
    """Show paired changes rather than comparing different test populations."""
    lines = [
        "| Test month | Method | 96-update mean RMSE | 288-update mean RMSE | Paired change ± seed SD | Seeds improving |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for month in MONTHS:
        for norm in NORMS:
            rows = [r for r in budget if r["month"] == month and r["norm"] == norm]
            changes = [r["final_minus_initial_rmse"] for r in rows]
            lines.append(
                f"| {month} | {LABELS[norm]} | {np.mean([r['initial_mean_rmse'] for r in rows]):.4f} | {np.mean([r['final_mean_rmse'] for r in rows]):.4f} | {np.mean(changes):+.4f} ± {np.std(changes, ddof=1):.4f} | {sum(v < 0 for v in changes)}/3 |"
            )
    return "\n".join(lines)


def plot_daily(
    daily: list[dict[str, Any]], month: str, metric: str, output: Path, preview: Path
) -> str:
    """Render separate month/metric charts with sample-SD whiskers across seeds."""
    title_month = "January 2024" if month == "2024-01" else "July 2024"
    scale = 1e6 if metric == "mean_absolute_sie_km2" else 1.0
    ylabel = (
        "Mean absolute extent error (million km²)"
        if scale > 1
        else "Sea-ice concentration RMSE (fraction)"
    )
    short = "sie" if scale > 1 else "rmse"
    filename = f"norm-594-longer-{month}-{short}.svg"
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for norm, marker, linestyle in (
        ("batchnorm", "o", "-"),
        ("groupnorm", "s", "--"),
        ("persistence", "^", ":"),
    ):
        values = [
            r
            for r in daily
            if r["month"] == month
            and r["norm"] == norm
            and (r["selection"] == "final" or norm == "persistence")
        ]
        means, deviations = [], []
        for lead in range(1, 15):
            raw = [r[metric] / scale for r in values if r["lead_days"] == lead]
            means.append(float(np.mean(raw)))
            deviations.append(float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0)
        ax.errorbar(
            range(1, 15),
            means,
            yerr=deviations,
            marker=marker,
            linestyle=linestyle,
            markersize=4,
            capsize=2,
            linewidth=1.5,
            label=LABELS[norm],
        )
    ax.set(
        title=f"{title_month} forecasts after 288 updates",
        xlabel="Forecast lead (days)",
        ylabel=ylabel,
        xlim=(0.7, 14.3),
        ylim=(0, None),
    )
    ax.set_xticks(range(1, 15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.text(
        0.5,
        0.02,
        "Three paired seeds; whiskers are sample SD, not confidence intervals. Fifteen overlapping windows.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / filename, bbox_inches="tight", metadata={"Date": None})
    svg_path = output / filename
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        preview / filename.replace(".svg", ".png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    return filename


def plot_validation(records: list[dict[str, Any]], output: Path, preview: Path) -> str:
    """Show validation curves without implying that a fixed budget established convergence."""
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for norm, marker, linestyle in (("batchnorm", "o", "-"), ("groupnorm", "s", "--")):
        selected = [r for r in records if r["norm"] == norm]
        values = np.asarray(
            [[h["validation_mse"] for h in r["validation_history"]] for r in selected]
        )
        ax.errorbar(
            [96, 144, 192, 240, 288],
            values.mean(axis=0),
            yerr=values.std(axis=0, ddof=1),
            marker=marker,
            linestyle=linestyle,
            capsize=3,
            label=LABELS[norm],
        )
    ax.set(
        title="Matched continuation validation curves",
        xlabel="Cumulative optimizer updates",
        ylabel="Validation sea-ice concentration MSE",
        ylim=(0, None),
    )
    ax.set_xticks([96, 144, 192, 240, 288])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.02,
        "Eight fixed windows from July 2022 and January 2023. Whiskers are sample SD across three seeds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    filename = "norm-594-longer-validation.svg"
    fig.savefig(output / filename, bbox_inches="tight", metadata={"Date": None})
    svg_path = output / filename
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        preview / filename.replace(".svg", ".png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    return filename


def formatted(value: float, sd: float | None, scale: float = 1.0) -> str:
    """Keep exact numerical lookup tables beside the figures."""
    return (
        f"{value / scale:.4f}"
        if sd is None
        else f"{value / scale:.4f} ± {sd / scale:.4f}"
    )


def metric_table(aggregate: list[dict[str, Any]], selection: str) -> str:
    """Use the same metric definition in every month and checkpoint table."""
    lines = [
        "| Test month | Method | Mean 14-day RMSE | Days 10-14 RMSE | Mean MAE | Mean absolute SIE (million km²) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for month in MONTHS:
        for norm in (*NORMS, "persistence"):
            row = next(
                r
                for r in aggregate
                if r["month"] == month
                and r["norm"] == norm
                and r["selection"] == selection
            )
            cells = [
                formatted(row[k + "_mean"], row[k + "_sd"], scale)
                for k, scale in (
                    ("mean_rmse", 1),
                    ("late_rmse", 1),
                    ("mean_mae", 1),
                    ("mean_absolute_sie_km2", 1e6),
                )
            ]
            lines.append("| " + " | ".join([month, LABELS[norm], *cells]) + " |")
    return "\n".join(lines)


def render_report(  # noqa: PLR0913 - explicit reviewed evidence inputs
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
    audit: dict[str, Any],
    aggregate: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    plots: list[str],
    *,
    budget: list[dict[str, Any]],
) -> str:
    """Describe the bounded result and all material comparison limitations."""
    final = [r for r in paired if r["selection"] == "final"]
    findings = []
    for month in MONTHS:
        group = [r for r in final if r["month"] == month]
        better = sum(r["groupnorm_minus_batchnorm_rmse"] < 0 for r in group)
        gn_better_persistence = sum(
            r["groupnorm_minus_persistence_rmse"] < 0 for r in group
        )
        bn_better_persistence = sum(
            r["batchnorm_minus_persistence_rmse"] < 0 for r in group
        )
        differences = ", ".join(
            f"{r['groupnorm_minus_batchnorm_rmse']:+.5f}" for r in group
        )
        findings.append(
            f"For **{month}**, GroupNorm has lower mean RMSE in **{better}/3** paired seeds; GroupNorm minus BatchNorm differences are {differences}. The numbers of seeds beating persistence are **{bn_better_persistence}/3** for BatchNorm and **{gn_better_persistence}/3** for GroupNorm."
        )
    selections = [
        "| Seed | Method | Validation-selected total updates | Initial validation MSE | Final validation MSE |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for r in records:
        selections.append(  # noqa: PERF401 - row-specific presentation
            f"| {r['seed']} | {LABELS[r['norm']]} | {r['selected_step']} | {r['validation_history'][0]['validation_mse']:.6f} | {r['validation_history'][-1]['validation_mse']:.6f} |"
        )
    figure_text = "\n\n".join(
        f"![{name.removesuffix('.svg').replace('-', ' ')}](../assets/{name})"
        for name in plots
    )
    selection_table = "\n".join(selections)
    return f"""# Longer matched normalization training

This follow-up to [PR #594](https://github.com/alan-turing-institute/cryocast/pull/594) continues the six recorded [96-update runs](norm-evaluation-575-extended.md). It adds training without changing their spatial resolution or capacity, and expands testing from ten January windows to all fifteen January and fifteen July 2024 windows.

## What was held constant

Both arms retain the CNN-UNet-CNN with 1,256,655 total parameters (1,256,337 trainable and 318 in the frozen, unused target encoder) with processor width 16, 432 x 432 physical grids, 144 x 144 latent grids, three observed history days and a full fourteen-day latent autoregressive forecast. ERA5 and OSI SAF inputs, Huber loss with delta 0.5, batch size one, decoder normalization and all model defaults are unchanged. Argo is excluded from both arms. These remain reduced-capacity models relative to the 44.7-million-parameter production configuration.

Three paired seeds, 575/576/577, restore their own exact 96-update model checkpoints and AdamW moments. Each receives **192 additional updates**, giving **288 total updates per run** and **1,152 newly executed optimizer updates**. Both arms of a seed receive byte-identical training arrays in the same order; all 192 newly selected windows are distinct. The earlier 96-update histories are inherited, not rerun or asserted to have identical training arrays retrospectively.

## Learning-rate and selection protocol

The prior one-epoch scheduler had already reduced the learning rate to 0.000001. Continuing at that terminal rate would answer a different question. This study preregisters an identical cosine restart from **0.001 to 0.00001** across the 192 additional updates, retaining AdamW moments, weight decay 0.01 and gradient clipping at norm one. This is a controlled continuation with an explicit restart, **not an uninterrupted replay of the old schedule** and not a new from-scratch 288-update experiment.

The primary comparison is the fixed final 288-update checkpoint. A secondary comparison uses the lowest common-mask validation MSE at 96, 144, 192, 240 or 288 updates, allowing the original checkpoint to remain selected. Eight fixed validation windows are balanced across July 2022 and January 2023. All six measured-run selections are frozen before their test forecasts are scored. A separate two-update plumbing smoke check reused the already-explored first January window before the measured run; its scores are excluded and did not change the fixed protocol. No model, seed, checkpoint or learning rate is selected using the new test results.

Training uses 2020-2021 windows. A calendar-level check includes every history and forecast date and finds no overlap with validation or test years. Existing source normalization statistics end in November 2021. A separate status-flag audit reconstructs masks from all 855 stored days and from the 731 training days alone: both reproduce the stored ocean and active masks exactly, with zero differing cells. This is a verified property of this fixed snapshot, not a general guarantee for every dataset. The legacy stores use the explicitly enabled in-memory Anemoi missing-date correction; source metadata, date indices, masks and original checkpoint hashes remain unchanged.

## Final 288-update results

RMSE is pooled over valid pixels and windows separately for each lead, then averaged over the fourteen leads. The late-horizon measure averages leads 10-14. SIE uses a strict 15% threshold and nominal 625 km² grid cells; its error is made absolute **per window before averaging**. January and July are reported separately and together. The thirty overlapping windows cover 56 distinct forecast-target dates; repeated appearances of a date are not independent observations. Values are mean ± sample standard deviation across three seeds; persistence has no seed variation.

{metric_table(aggregate, "final")}

{" ".join(findings)}

## Effect of the additional training

The initial 96-update checkpoints are replayed on the **same thirty windows** as their final states. This isolates the added-training comparison from the expansion of the January-only test set. Negative paired change means lower RMSE after continuation; the spread is sample SD of the three seed-wise changes.

{budget_table(budget)}

## Validation-selected secondary comparison

{metric_table(aggregate, "selected")}

{selection_table}

Including the starting checkpoint prevents a worse continuation from being presented as an improvement through forced final-step selection. Conversely, the primary fixed-budget table remains visible and is not replaced by a selectively favorable checkpoint table.

## Per-lead and validation figures

{figure_text}

Whiskers show sample standard deviation across three training seeds, **not confidence intervals**. Fifteen windows within each month share many observed and forecast dates; they are not fifteen independent seasonal replications.

## Independent verification

A separate verifier, which does not import the training runner's metric implementation, hashes and re-scores all **{audit["independently_rescored_cases"]:,} per-origin/per-lead records** from saved forecast arrays using NumPy. It verifies prediction-independent masks, finite predictions, pixel counts, squared and absolute errors, paired absolute extent errors, training-order/data hashes, safe checkpoint checksums and validation-only selection. It also checks the initial checkpoints against the original ten-window January results within an explicitly declared float32-versus-float64 reduction tolerance.

The frozen experiment source is `{protocol["source_revision"]}` and runner SHA-256 is `{protocol["runner_sha256"]}`. Reproduction code and compact numerical evidence are in [`validation/norm_594_longer`](https://github.com/sylvesterkaczmarek/icenet-mp/blob/research/groupnorm-unet-575/validation/norm_594_longer/README.md). Full forecast arrays, model weights and execution logs remain on the external research volume; none are committed as repository data.

## Decision and remaining limits

Keep the shipped BatchNorm default. This experiment adds a longer full-grid continuation and a July holdout to the existing evidence. It does not establish converged full-capacity training, general superiority across encoder/decoder families, Argo-conditioned performance or release-level seasonal reliability. January data have already been explored in the earlier study, and July covers only one month in one year. The protocol was fixed before continuation, but the study is a follow-up to known preliminary results rather than an independent confirmatory trial.

At batch size one, 288 total updates remain less than a full pass over the 715 eligible training windows; the inherited 96-update sample history can also overlap the new continuation windows. The existing preprocessing includes its NaN-to-zero SIC ingestion policy; this comparison does not reconstruct original missingness or introduce a new QC filter. Prepared finite values and the fixed ocean mask define scoring support. The training and validation curves are reported rather than used to assert convergence. Three seeds support descriptive paired differences only; no significance or population confidence interval is claimed. MPS execution reports a nondeterministic indexed-accumulation operation in backward; seeded schedules and saved evidence are reproducible, but bitwise-identical retraining on every machine is not promised.

No production defaults or trained normalization architectures are switched. BatchNorm and GroupNorm checkpoints remain architecture-specific. A from-scratch, full-capacity, independently selected study over more years and configurations is still required before any default change.
"""


def main() -> None:
    """Create the existing PR's static report, tables and reproducible evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.run / "independent-audit.json").read_text())
    check(
        audit["passed"] and audit["independently_rescored_cases"] == EXPECTED_CASES,
        "Independent audit is incomplete",
    )
    protocol = json.loads((args.run / "protocol.json").read_text())
    records = json.loads((args.run / "frozen_selections.json").read_text())["records"]
    rows = load_cases(args.run)
    summaries, daily = seed_summaries(rows)
    aggregate, paired = cross_seed(summaries), paired_comparisons(summaries)
    budget = training_budget_effects(summaries)
    raw = args.repository / "validation/norm_594_longer/raw"
    raw.mkdir(exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)
    assets = args.repository / "docs/src/assets"
    for filename in (
        "protocol.json",
        "windows.json",
        "frozen_selections.json",
        "independent-audit.json",
        "mask-provenance-audit.json",
        "complete.json",
        "forecast_manifest.json",
        "test_support.json",
    ):
        (raw / filename).write_bytes((args.run / filename).read_bytes())
    for seed, norm, selection in sorted(
        {(r["seed"], r["norm"], r["selection"]) for r in rows},
        key=lambda item: (str(item[0]), item[1], item[2]),
    ):
        subset = [
            r
            for r in rows
            if (r["seed"], r["norm"], r["selection"]) == (seed, norm, selection)
        ]
        write_csv(raw / f"per-case-{norm}-{seed}-{selection}.csv", subset)
    write_csv(raw / "per-seed.csv", summaries)
    write_csv(raw / "per-lead.csv", daily)
    write_csv(raw / "paired-effects.csv", paired)
    write_csv(raw / "training-budget-effects.csv", budget)
    write_json(raw / "aggregate.json", aggregate)
    summary = {
        "aggregate": aggregate,
        "paired": paired,
        "training_budget_effects": budget,
        "selected_steps": [
            {"seed": r["seed"], "norm": r["norm"], "step": r["selected_step"]}
            for r in records
        ],
        "raw_cases": len(rows),
    }
    # Rebuilding from the public bundle must not add an extra source artifact.
    if args.run.resolve() != raw.resolve():
        write_json(args.run / "summary.json", summary)
    plots = [
        plot_daily(daily, month, metric, assets, args.previews)
        for month in MONTHS[:2]
        for metric in ("rmse", "mean_absolute_sie_km2")
    ]
    plots.append(plot_validation(records, assets, args.previews))
    report = render_report(
        protocol, records, audit, aggregate, paired, plots, budget=budget
    )
    (args.repository / "docs/src/research/norm-evaluation-594-longer.md").write_text(
        report
    )
    write_json(
        raw / "manifest.json",
        {
            p.name: file_hash(p)
            for p in sorted(raw.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    )
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
