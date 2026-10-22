"""Independently rescore retained forecasts and audit the continuation experiment.

This verifier does not import the runner's scoring implementation. It reads every
saved forecast and recomputes its sufficient statistics directly with NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ICE_FRACTION_THRESHOLD = 0.15
CELL_AREA_KM2 = 625.0
EXPECTED_INITIAL_UPDATES = 96
EXPECTED_FINAL_UPDATES = 288
EXPECTED_RUNS = 6
EXPECTED_ADDITIONAL_UPDATES = 1152


class EvidenceError(ValueError):
    """Reject inconsistent or incomplete evidence."""


def check(condition: object, message: str) -> None:
    """Stop with a precise audit failure."""
    if not condition:
        raise EvidenceError(message)


def digest(path: Path) -> str:
    """Hash an evidence file without reading the whole file into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    """Ensure numerical evidence is the frozen file named by its manifest."""
    check(digest(path) == expected, f"Checksum mismatch for {path.name}")


def _date_set(windows: list[dict[str, Any]]) -> set[str]:
    """Include both observations and future targets in split-overlap checks."""
    return {
        date for window in windows for date in window["history"] + window["forecast"]
    }


def verify_protocol(root: Path) -> dict[str, Any]:  # noqa: PLR0915 - explicit independent evidence gates
    """Check paired schedules, split boundaries and validation-only selection."""
    protocol = json.loads((root / "protocol.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    selections = json.loads((root / "frozen_selections.json").read_text())
    windows = json.loads((root / "windows.json").read_text())
    support = json.loads((root / "test_support.json").read_text())
    check(not protocol["smoke"], "A smoke run is not experiment evidence")
    check(protocol["additional_updates"] == 192, "Unexpected continuation budget")  # noqa: PLR2004
    check(
        protocol["initial_updates"] == EXPECTED_INITIAL_UPDATES
        and protocol["total_updates"] == EXPECTED_FINAL_UPDATES,
        "Cumulative updates differ",
    )
    check(
        complete["runs"] == EXPECTED_RUNS
        and complete["added_optimizer_updates"] == EXPECTED_ADDITIONAL_UPDATES,
        "Incomplete matched run set",
    )
    check(
        complete["test_evaluated_after_all_selections_frozen"],
        "Test-selection gate missing",
    )
    verify_hash(root / "protocol.json", complete["protocol_sha256"])
    verify_hash(root / "frozen_selections.json", complete["frozen_selections_sha256"])
    verify_hash(root / "per_case.json", complete["per_case_sha256"])
    check(
        protocol["frozen_utc"] < selections["frozen_utc"],
        "Protocol was not frozen first",
    )
    check(
        {key: len(value) for key, value in windows.items()}
        == {"train": 715, "validation": 30, "test": 30},
        "Window count mismatch",
    )
    sets = {key: _date_set(value) for key, value in windows.items()}
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        check(
            not sets[first] & sets[second], "Calendar overlap across split boundaries"
        )
    for split in windows.values():
        for window in split:
            origin = np.datetime64(window["origin"], "D")
            expected_history = [
                str(origin + np.timedelta64(d, "D")) for d in (-2, -1, 0)
            ]
            expected_forecast = [
                str(origin + np.timedelta64(d, "D")) for d in range(1, 15)
            ]
            check(
                window["history"] == expected_history
                and window["forecast"] == expected_forecast,
                "Incorrect calendar lead or history",
            )
    check(
        support["test_windows"] == windows["test"],
        "Test windows changed or were subsampled",
    )
    check(
        protocol["validation_indices"] == [0, 5, 9, 14, 15, 20, 24, 29],
        "Validation selection differs",
    )
    check(
        protocol["test_indices"] == list(range(30)),
        "Not all January/July windows were scored",
    )
    check(
        {w["origin"][:7] for w in windows["test"]} == {"2024-01", "2024-07"},
        "Unexpected test months",
    )
    mask_provenance = json.loads((root / "mask-provenance-audit.json").read_text())
    check(
        mask_provenance["stored_hashes"]["land"] == protocol["ocean_mask_sha256"],
        "Mask provenance is for a different scoring mask",
    )
    check(
        all(
            v["stored_matches_training_only"]
            and v["stored_matches_all_period_reconstruction"]
            for v in mask_provenance["mask_results"].values()
        ),
        "Stored mask differs from training-only geography",
    )
    records = selections["records"]
    expected_keys = {
        (seed, norm) for seed in (575, 576, 577) for norm in ("batchnorm", "groupnorm")
    }
    check(
        {(r["seed"], r["norm"]) for r in records} == expected_keys,
        "Missing or repeated run",
    )
    for record in records:
        check(record["adam_moments_restored"], "Optimizer was not restored")
        check(record["inherited_lr"] == 1e-6, "Historical terminal rate changed")  # noqa: PLR2004
        updates = record["updates"]
        check(
            [u["step"] for u in updates] == list(range(97, 289)),
            "Update sequence has gaps",
        )
        order = np.random.default_rng(594_000 + record["seed"]).permutation(715)[:192]
        check(
            [u["dataset_index"] for u in updates] == order.tolist(),
            "Training windows differ from frozen permutation",
        )
        check(
            hashlib.sha256(order.tobytes()).hexdigest()
            == record["training_order_sha256"],
            "Training permutation checksum differs",
        )
        expected_lr = (
            1e-5 + (1e-3 - 1e-5) * (1 + np.cos(np.pi * np.arange(192) / 191)) / 2
        )
        np.testing.assert_allclose(
            [u["lr"] for u in updates], expected_lr, rtol=1e-13, atol=1e-15
        )
        check(
            all(
                np.isfinite(u["huber"]) and np.isfinite(u["gradient_norm"])
                for u in updates
            ),
            "Nonfinite training diagnostics",
        )
        history = record["validation_history"]
        check(
            [r["step"] for r in history] == [96, 144, 192, 240, 288],
            "Validation schedule changed",
        )
        selected = min(history, key=lambda item: item["validation_mse"])
        check(
            record["selected_step"] == selected["step"],
            "Selection was not the minimum validation MSE",
        )
        check(
            record["selected_validation_mse"] == selected["validation_mse"],
            "Selected score differs",
        )
        for artifact in record["checkpoints"].values():
            verify_hash(root / artifact["file"], artifact["sha256"])
    for seed in (575, 576, 577):
        pair = [r for r in records if r["seed"] == seed]
        for key in (
            "training_order_sha256",
            "training_values_sha256",
            "parameters",
            "total_steps",
        ):
            check(pair[0][key] == pair[1][key], f"Unmatched paired condition {key}")
    for name, source in protocol["source_data"].items():
        check(
            source["statistics_end_date"] < "2022",
            f"Held-out normalization data in {name}",
        )
        check(source["field_shape"] == [432, 432], "Spatial resolution changed")
    return {
        "matched_runs": len(records),
        "additional_optimizer_updates": 1152,
        "calendar_disjoint": True,
        "paired_training_arrays_identical": True,
        "validation_only_selection": True,
        "restored_optimizer_schedule_checked": True,
        "test_windows": 30,
        "test_leads": 14,
        "unique_test_target_dates": len(
            {d for window in windows["test"] for d in window["forecast"]}
        ),
        "stored_masks_equal_training_only_reconstruction": True,
        "test_months": ["2024-01", "2024-07"],
        "scope": "Continuation with an identical LR restart, not uninterrupted scheduler continuation or full-size convergence",
    }


def score_arrays(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    """Recompute one lead using a separate direct double-precision calculation."""
    check(prediction.shape == truth.shape == mask.shape, "Raw score arrays misalign")
    p = np.asarray(prediction)[mask].astype(np.float64)
    t = np.asarray(truth)[mask].astype(np.float64)
    check(
        p.size and np.isfinite(p).all() and np.isfinite(t).all(),
        "Invalid raw forecasts or support",
    )
    delta = p - t
    # The strict ice threshold uses the stored field dtype, matching the physical
    # field's representable threshold. Only error reductions promote to float64.
    predicted_ice = np.asarray(prediction)[mask] > np.asarray(
        ICE_FRACTION_THRESHOLD, dtype=prediction.dtype
    )
    target_ice = np.asarray(truth)[mask] > np.asarray(
        ICE_FRACTION_THRESHOLD, dtype=truth.dtype
    )
    ice_difference = int(np.count_nonzero(predicted_ice)) - int(
        np.count_nonzero(target_ice)
    )
    return {
        "count": int(p.size),
        "absolute_sum": float(np.sum(np.abs(delta))),
        "squared_sum": float(np.dot(delta, delta)),
        "signed_sie_km2": ice_difference * CELL_AREA_KM2,
        "absolute_sie_km2": abs(ice_difference) * CELL_AREA_KM2,
    }


def verify_forecasts(root: Path) -> dict[str, Any]:
    """Check every retained prediction against independently recomputed statistics."""
    support = json.loads((root / "test_support.json").read_text())
    manifests = json.loads((root / "forecast_manifest.json").read_text())
    rows = json.loads((root / "per_case.json").read_text())
    expected_shape = (30, 14, 1, 432, 432)
    for label in ("truth", "persistence", "valid"):
        verify_hash(root / "test" / (label + ".npy"), support[label + "_sha256"])
    truth = np.load(root / "test/truth.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(root / "test/valid.npy", mmap_mode="r", allow_pickle=False)
    check(
        truth.shape == mask.shape == expected_shape and mask.dtype == np.bool_,
        "Raw target shape or mask dtype differs",
    )
    check(len(manifests) == 18, "Missing initial/selected/final forecasts")  # noqa: PLR2004
    lookup = {}
    for row in rows:
        key = (row["seed"], row["norm"], row["selection"], row["origin"], row["lead"])
        check(key not in lookup, "Duplicate scored case")
        lookup[key] = row
    check(len(rows) == 7980, "Unexpected raw metric count")  # noqa: PLR2004
    checked = 0
    maximum_absolute_difference = 0.0
    for artifact in [
        {
            "seed": None,
            "norm": "persistence",
            "selection": "reference",
            "file": "persistence.npy",
            "sha256": support["persistence_sha256"],
        },
        *manifests,
    ]:
        path = root / "test" / artifact["file"]
        verify_hash(path, artifact["sha256"])
        predictions = np.load(path, mmap_mode="r", allow_pickle=False)
        check(
            predictions.shape == expected_shape,
            "Forecast shape differs between methods",
        )
        for index, window in enumerate(support["test_windows"]):
            for lead in range(14):
                expected = score_arrays(
                    predictions[index, lead], truth[index, lead], mask[index, lead]
                )
                key = (
                    artifact["seed"],
                    artifact["norm"],
                    artifact["selection"],
                    window["origin"],
                    lead + 1,
                )
                recorded = lookup[key]
                for metric, value in expected.items():
                    np.testing.assert_allclose(
                        recorded[metric],
                        value,
                        rtol=2e-12,
                        atol=2e-10,
                        err_msg=str(key) + metric,
                    )
                    maximum_absolute_difference = max(
                        maximum_absolute_difference, abs(recorded[metric] - value)
                    )
                checked += 1
        del predictions
    check(checked == len(rows), "Not all metric cases independently checked")
    return {
        "raw_forecast_arrays_verified": len(manifests),
        "reference_arrays_verified": 3,
        "independently_rescored_cases": checked,
        "maximum_absolute_numeric_difference": maximum_absolute_difference,
        "shared_truth_and_mask": True,
        "prediction_dependent_exclusion": False,
        "extent_absolute_before_averaging": True,
        "raw_forecasts_sha256_verified": True,
    }


def compare_historical_january(root: Path, previous: Path) -> list[dict[str, Any]]:
    """Check the initial 96-update checkpoint replay against the prior January result."""
    rows = json.loads((root / "per_case.json").read_text())
    origins = sorted({r["origin"] for r in rows if r["month"] == "2024-01"})[:10]
    comparisons = []
    for seed in (575, 576, 577):
        for norm in ("batchnorm", "groupnorm"):
            original = json.loads((previous / f"{norm}-seed{seed}.json").read_text())
            selected = [
                r
                for r in rows
                if r["seed"] == seed
                and r["norm"] == norm
                and r["selection"] == "initial"
                and r["origin"] in origins
            ]
            rmse = [
                np.sqrt(
                    sum(r["squared_sum"] for r in selected if r["lead"] == lead)
                    / sum(r["count"] for r in selected if r["lead"] == lead)
                )
                for lead in range(1, 15)
            ]
            old = np.asarray(original["rmse_per_lead"])
            # Old MPS float32 reductions versus independently rescored float64.
            np.testing.assert_allclose(rmse, old, rtol=2e-4, atol=2e-5)
            comparisons.append(
                {
                    "seed": seed,
                    "norm": norm,
                    "january_windows": 10,
                    "maximum_rmse_absolute_difference": float(
                        np.max(np.abs(np.asarray(rmse) - old))
                    ),
                    "old_checkpoint_predictions_reproduced_within_declared_tolerance": True,
                }
            )
    return comparisons


def main() -> None:
    """Audit all raw evidence before producing an analytical report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol(args.run)
    forecasts = verify_forecasts(args.run)
    historical = compare_historical_january(args.run, args.previous)
    result = {
        "passed": True,
        **protocol,
        **forecasts,
        "historical_replay": historical,
        "verifier_sha256": digest(Path(__file__)),
        "no_new_independent_replications_claimed": True,
    }
    (args.run / "independent-audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
