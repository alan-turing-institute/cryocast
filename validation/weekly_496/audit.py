"""Independently reconstruct matched-horizon masks, persistence and scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)
DAY = np.timedelta64(1, "D")
LEADS = (7, 14)
SEEDS = (496, 497, 498)
STRIDES = (1, 7)
EXPECTED_RUNS = 6
EXPECTED_ORIGINS = 15
EXPECTED_UPDATES = 840
SOURCE_SHA256 = "949a176a64a7e7dece6b8953cbb740fcaf38900bd58a57efe6aa6dc1961dd9b1"


def ensure(condition: object, message: str) -> None:
    """Fail instead of publishing incomplete or mismatched results."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Checksum exact bytes independently of the experiment's writer."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_source(
    source: Path, experiment: Path, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Rebuild true fields and masks from the original source, not stored scores."""
    ensure(
        digest(source / "observed_sic_paired.npz") == SOURCE_SHA256,
        "Source checksum changed",
    )
    date_record = json.loads((experiment / "dates.json").read_text())
    origins = date_record["test_origins"]
    ensure(len(origins) == EXPECTED_ORIGINS, "Wrong test-origin count")
    with np.load(source / "observed_sic_paired.npz", allow_pickle=False) as raw:
        fields = np.concatenate([raw[s] for s in ("train", "val", "test")])
        quality = np.concatenate([raw[s + "_valid"] for s in ("train", "val", "test")])
        dates = np.concatenate([raw[s + "_dates"] for s in ("train", "val", "test")])
        ocean = raw["mask"].astype(bool)
    lookup = {str(date): i for i, date in enumerate(dates)}
    truths, masks, persistent = [], [], []
    for origin in origins:
        day = np.datetime64(origin, "D")
        idx = lookup[origin]
        common = ocean.copy()
        for offset in (0, -1, -7):
            common &= quality[lookup[str(day + offset * DAY)]]
        truth, mask = [], []
        for lead in LEADS:
            target_date = str(day + lead * DAY)
            ensure(target_date.startswith("2024-02-"), "Target left held-out February")
            target_index = lookup[target_date]
            truth.append(
                np.where(quality[target_index] & ocean, fields[target_index], 0)
            )
            mask.append(common & quality[target_index])
        truths.append(np.stack(truth))
        masks.append(np.stack(mask))
        persistent.append(
            np.repeat(
                np.where(quality[idx] & ocean, fields[idx], 0)[None], len(LEADS), axis=0
            )
        )
    independent_truth = np.stack(truths)[:, :, None]
    independent_mask = np.stack(masks)[:, :, None]
    independent_persistence = np.stack(persistent)[:, :, None]
    np.testing.assert_array_equal(arrays["truth"], independent_truth)
    np.testing.assert_array_equal(arrays["mask"], independent_mask)
    np.testing.assert_array_equal(arrays["persistence"], independent_persistence)
    ensure(
        set(date_record["train_origins"]).isdisjoint(origins), "Training/test overlap"
    )
    ensure(
        all(d.startswith("2023-") for d in date_record["train_origins"]),
        "Training left 2023",
    )
    ensure(
        all(d.startswith("2024-01-") for d in date_record["validation_origins"]),
        "Validation left January",
    )
    return date_record


def validate_rows(
    experiment: Path, arrays: dict[str, np.ndarray], dates: dict[str, Any]
) -> list[dict[str, Any]]:
    """Recompute every case without using the experiment's metric function."""
    with (experiment / "per_case.csv").open() as stream:
        published = list(csv.DictReader(stream))
    ensure(
        len(published) == EXPECTED_RUNS * EXPECTED_ORIGINS * len(LEADS),
        "Incomplete case table",
    )
    independent = []
    for record in published:
        seed, stride, lead = (int(record[k]) for k in ("seed", "stride", "lead_days"))
        i, j = dates["test_origins"].index(record["origin"]), LEADS.index(lead)
        predicted = arrays[f"seed{seed}_stride{stride}"][i, j]
        truth = arrays["truth"][i, j]
        persist = arrays["persistence"][i, j]
        mask = arrays["mask"][i, j]
        ensure(bool(np.isfinite(predicted).all()), "Nonfinite forecast")
        delta = predicted[mask].astype(np.float64) - truth[mask].astype(np.float64)
        base_delta = persist[mask].astype(np.float64) - truth[mask].astype(np.float64)
        computed = {
            "n_pixels": int(mask.sum()),
            "model_abs_sum": float(np.abs(delta).sum()),
            "model_sq_sum": float(np.dot(delta, delta)),
            "persistence_abs_sum": float(np.abs(base_delta).sum()),
            "persistence_sq_sum": float(np.dot(base_delta, base_delta)),
        }
        for field, actual in computed.items():
            np.testing.assert_allclose(
                float(record[field]), actual, rtol=1e-12, atol=1e-10
            )
        expected_date = str(np.datetime64(record["origin"]) + lead * DAY)
        ensure(record["valid_date"] == expected_date, "Origin/lead/valid-date mismatch")
        independent.append(
            {
                "seed": seed,
                "stride": stride,
                "origin": record["origin"],
                "lead_days": lead,
                "valid_date": expected_date,
                **computed,
            }
        )
    return independent


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-pixel errors and compare with the same persistence denominator."""
    summaries = []
    for seed in SEEDS:
        for stride in STRIDES:
            for lead in LEADS:
                group = [
                    r
                    for r in rows
                    if (r["seed"], r["stride"], r["lead_days"]) == (seed, stride, lead)
                ]
                ensure(
                    len(group) == EXPECTED_ORIGINS, "Missing or duplicate group entries"
                )
                count = sum(r["n_pixels"] for r in group)
                absolute = sum(r["model_abs_sum"] for r in group)
                squared = sum(r["model_sq_sum"] for r in group)
                p_absolute = sum(r["persistence_abs_sum"] for r in group)
                p_squared = sum(r["persistence_sq_sum"] for r in group)
                ensure(p_absolute > 0 and p_squared > 0, "Undefined persistence skill")
                summaries.append(
                    {
                        "seed": seed,
                        "stride": stride,
                        "lead_days": lead,
                        "n_origins": EXPECTED_ORIGINS,
                        "n_pixels": count,
                        "mae_pp": 100 * absolute / count,
                        "rmse_pp": 100 * float(np.sqrt(squared / count)),
                        "persistence_mae_pp": 100 * p_absolute / count,
                        "persistence_rmse_pp": 100 * float(np.sqrt(p_squared / count)),
                        "mae_skill": 1 - absolute / p_absolute,
                        "mse_skill": 1 - squared / p_squared,
                    }
                )
    return summaries


def main() -> None:
    """Audit the immutable experiment and save a independently verified summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment
    complete = json.loads((experiment / "complete.json").read_text())
    for file, key in (
        ("forecasts.npz", "forecast_sha256"),
        ("per_case.csv", "per_case_sha256"),
        ("protocol.json", "protocol_sha256"),
    ):
        ensure(
            digest(experiment / file) == complete[key],
            "Experiment artifact checksum differs",
        )
    with np.load(experiment / "forecasts.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    dates = compare_source(args.source, experiment, arrays)
    rows = validate_rows(experiment, arrays, dates)
    records = json.loads((experiment / "frozen_checkpoints.json").read_text())
    ensure(len(records) == EXPECTED_RUNS, "Expected six frozen training records")
    for seed in SEEDS:
        group = [r for r in records if r["seed"] == seed]
        ensure(len(group) == len(STRIDES), "Seed pairing incomplete")
        ensure(
            group[0]["initial_state_sha256"] == group[1]["initial_state_sha256"],
            "Unmatched initialization",
        )
        for record in group:
            ensure(record["updates"] == EXPECTED_UPDATES, "Mismatched training budget")
            checkpoint = experiment / f"seed{seed}-stride{record['stride']}.pt"
            ensure(
                digest(checkpoint) == record["checkpoint_sha256"],
                "Frozen weights changed",
            )
            minimum = min(record["history"], key=lambda h: h["validation_mse"])
            ensure(
                record["best_epoch"] == minimum["epoch"],
                "Checkpoint selection did not follow validation",
            )
    summary = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "independent_checks": [
            "source checksum",
            "actual forecast dates",
            "180 case sums",
            "same origin persistence",
            "same pixel masks",
            "disjoint targets",
            "six locked checkpoints",
            "matched initial weights and updates",
        ],
        "summary": summary,
        "training": records,
        "dates": dates,
        "protocol": json.loads((experiment / "protocol.json").read_text()),
    }
    (args.output / "audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    log.info("INDEPENDENT AUDIT PASSED\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
