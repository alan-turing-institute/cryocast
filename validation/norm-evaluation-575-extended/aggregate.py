"""Aggregate paired raw trial JSON into CSV, summary and per-lead evidence figures."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAIN_UPDATES = 96
START_CHANNELS = 16
TEST_WINDOWS = 10
LEAD_DAYS = 14
BASELINE_TOLERANCE = 1e-5
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
RUNS = Path(os.environ.get("CRYOCAST_NORM_OUTPUT_DIR", ""))
ART = Path(__file__).resolve().parents[2] / "docs" / "src" / "assets"
if not os.environ.get("CRYOCAST_NORM_OUTPUT_DIR"):
    msg = "Set CRYOCAST_NORM_OUTPUT_DIR to the directory of six completed runs."
    raise RuntimeError(msg)
ART.mkdir(parents=True, exist_ok=True)
NORM_LABELS = {"batchnorm": "BatchNorm", "groupnorm": "GroupNorm"}
SEEDS = (575, 576, 577)
KEYS = {
    "rmse": "rmse_per_lead",
    "sie": "absolute_sie_error_km2_per_lead",
    "persistence_rmse": "persistence_rmse_per_lead",
    "persistence_sie": "persistence_absolute_sie_error_km2_per_lead",
}
runs = {}
for seed in SEEDS:
    for norm in NORM_LABELS:
        file = RUNS / f"{norm}-seed{seed}.json"
        if not file.exists():
            msg = f"Not all six trials completed; missing {file}"
            raise RuntimeError(msg)
        run = json.loads(file.read_text())
        if not (run["seed"] == seed and run["norm"] == norm):
            msg = f"Unexpected run identity in {file}"
            raise ValueError(msg)
        if not (
            run["optimizer_steps"] == TRAIN_UPDATES
            and run["start_out_channels"] == START_CHANNELS
        ):
            msg = f"Unexpected protocol in {file}"
            raise ValueError(msg)
        if run["test_samples"] != TEST_WINDOWS:
            msg = f"Unexpected test-sample count in {file}"
            raise ValueError(msg)
        if len(run["rmse_per_lead"]) != LEAD_DAYS:
            msg = f"Unexpected forecast horizon in {file}"
            raise ValueError(msg)
        runs[norm, seed] = run

sources = {run["source_sha"] for run in runs.values()}
scripts = {run["script_sha256"] for run in runs.values()}
if len(sources) != 1 or len(scripts) != 1:
    msg = "Mixed experiment versions"
    raise ValueError(msg)
for metric in ("persistence_rmse", "persistence_sie"):
    arr = np.array([runs[k][KEYS[metric]] for k in runs])
    if np.max(np.ptp(arr, axis=0)) >= BASELINE_TOLERANCE:
        msg = f"Test batches differ across runs for {metric}"
        raise ValueError(msg)
if len({r["parameters"] for r in runs.values()}) != 1:
    msg = "Mismatched model parameter counts"
    raise ValueError(msg)

perseed = []
for seed in SEEDS:
    for norm, label in NORM_LABELS.items():
        r = runs[norm, seed]
        perseed.append(
            {
                "seed": seed,
                "normalization": label,
                "training_updates": r["optimizer_steps"],
                "test_windows": r["test_samples"],
                "model_parameters": r["parameters"],
                "train_seconds": round(r["train_seconds"], 3),
                "mean_rmse": round(r["mean_rmse"], 7),
                "late_rmse_leads_10_14": round(r["late_rmse"], 7),
                "mean_absolute_sie_km2": round(r["mean_abs_sie_error_km2"], 3),
                "late_absolute_sie_km2": round(r["late_abs_sie_error_km2"], 3),
            }
        )
with (ART / "norm-evaluation-575-extended-seeds.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(perseed[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(perseed)

daily = []
arrays = {}
for kind in ("rmse", "sie"):
    for norm in NORM_LABELS:
        arrays[norm, kind] = np.stack([runs[norm, seed][KEYS[kind]] for seed in SEEDS])
    arrays["persistence", kind] = np.asarray(
        runs["batchnorm", 575][KEYS["persistence_" + kind]]
    )
for day in range(14):
    row: dict[str, int | float] = {"lead_day": day + 1}
    for kind in ("rmse", "sie"):
        for norm in NORM_LABELS:
            row[f"{norm}_{kind}_mean"] = round(
                float(arrays[norm, kind][:, day].mean()), 8 if kind == "rmse" else 2
            )
            row[f"{norm}_{kind}_sd"] = round(
                float(arrays[norm, kind][:, day].std(ddof=1)),
                8 if kind == "rmse" else 2,
            )
        row[f"persistence_{kind}"] = round(
            float(arrays["persistence", kind][day]), 8 if kind == "rmse" else 2
        )
    daily.append(row)
with (ART / "norm-evaluation-575-extended-daily.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(daily[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(daily)

x = np.arange(1, 15)
for kind, ylabel, filename in (
    ("rmse", "SIC RMSE (fraction)", "norm-evaluation-575-extended-rmse.svg"),
    ("sie", "Absolute SIE error (km²)", "norm-evaluation-575-extended-sie.svg"),
):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for norm in NORM_LABELS:
        a = arrays[norm, kind]
        mean, sd = a.mean(axis=0), a.std(axis=0, ddof=1)
        line = ax.plot(
            x, mean, marker="o", markersize=3, label=f"{NORM_LABELS[norm]} (n=3)"
        )[0]
        ax.fill_between(
            x, np.maximum(0, mean - sd), mean + sd, color=line.get_color(), alpha=0.16
        )
    ax.plot(
        x,
        arrays["persistence", kind],
        linestyle="--",
        label="Persistence",
        marker=".",
        markersize=3,
    )
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel(ylabel)
    ax.set_title("Reduced-capacity CNN-UNet-CNN: 96 updates, three seeds")
    ax.set_xticks(x[::2])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    target = ART / filename
    fig.savefig(target, format="svg")
    # Matplotlib's SVG path layout uses trailing spaces before line breaks.
    # Strip insignificant whitespace so git diff --check passes.
    target.write_text(
        "".join(
            line.rstrip(" \t\r") + "\n" for line in target.read_text().splitlines()
        ),
        encoding="utf-8",
    )
    plt.close(fig)

aggregate: dict[str, dict[str, object]] = {}
for norm in NORM_LABELS:
    aggregate[norm] = {}
    for key in (
        "mean_rmse",
        "late_rmse",
        "mean_abs_sie_error_km2",
        "late_abs_sie_error_km2",
        "train_seconds",
    ):
        a = np.asarray([runs[norm, seed][key] for seed in SEEDS])
        aggregate[norm][key] = {
            "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)),
            "values": [float(v) for v in a],
        }
aggregate["persistence"] = {
    "mean_rmse": float(np.mean(arrays["persistence", "rmse"])),
    "late_rmse": float(np.mean(arrays["persistence", "rmse"][-5:])),
    "mean_abs_sie_error_km2": float(np.mean(arrays["persistence", "sie"])),
    "late_abs_sie_error_km2": float(np.mean(arrays["persistence", "sie"][-5:])),
}
aggregate["protocol"] = {
    "steps": 96,
    "seeds": SEEDS,
    "source_sha": next(iter(sources)),
    "script_sha256": next(iter(scripts)),
    "test_samples": 10,
    "parameters": runs["batchnorm", 575]["parameters"],
}
(RUNS / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
log.info("Aggregation complete: %s", json.dumps(aggregate, indent=2))
