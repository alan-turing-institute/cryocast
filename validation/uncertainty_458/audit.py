"""Rebuild held-out scores independently of the training program's score helper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from netCDF4 import Dataset

log = logging.getLogger(__name__)
SEEDS = (458, 459, 460, 461, 462)
ARMS = ("huber", "weighted")
LEADS = (1, 7, 14)
EXPECTED_UPDATES = 1800
EXPECTED_PARAMETERS = 172831


def require(value: object, message: str) -> None:
    """Fail loudly on a mismatch rather than publish partial evidence."""
    if not value:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Compute the SHA-256 of an immutable experiment artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_state_hash(state: dict[str, torch.Tensor]) -> str:
    """Check tensor values, independent of their saved-container checksum."""
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_records(exp: Path) -> list[dict[str, Any]]:
    """Check selections, update counts and the two equal initial states per seed."""
    records = json.loads((exp / "frozen_checkpoints.json").read_text())
    require(len(records) == len(SEEDS) * len(ARMS), "Missing or extra completed runs")
    require(
        {(r["seed"], r["arm"]) for r in records}
        == {(s, a) for s in SEEDS for a in ARMS},
        "Invalid run identities",
    )
    for r in records:
        require(r["updates"] == EXPECTED_UPDATES, "Unequal training updates")
        require(r["parameters"] == EXPECTED_PARAMETERS, "Unexpected model capacity")
        best = min(r["history"], key=lambda h: h["validation_mse"])
        require(
            best["epoch"] == r["best_epoch"],
            "Selection is not minimum validation error",
        )
        require(
            best["validation_mse"] == r["best_validation_mse"],
            "Selection value mismatch",
        )
        for selection in ("selected", "last"):
            ckpt = exp / f"seed{r['seed']}-{r['arm']}-{selection}.pt"
            require(
                digest(ckpt) == r[selection + "_checkpoint_sha256"],
                "Changed checkpoint bytes",
            )
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            require(
                array_state_hash(state) == r[selection + "_state_sha256"],
                "Changed tensor weights",
            )
    for seed in SEEDS:
        a, b = [r for r in records if r["seed"] == seed]
        for field in (
            "initial_state_sha256",
            "training_order_sha256",
            "updates",
            "parameters",
        ):
            require(a[field] == b[field], "Training pairing mismatch: " + field)
    return records


def rebuild_reference_fields(
    source: Path, exp: Path, dates: dict[str, Any], support: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verify common training support and rebuild raw test reference fields."""
    with np.load(source / "observed_sic_paired.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    fields = np.concatenate([arrays[k] for k in ("train", "val", "test")])
    valid = np.concatenate([arrays[k + "_valid"] for k in ("train", "val", "test")])
    all_dates = np.concatenate([arrays[k + "_dates"] for k in ("train", "val", "test")])
    lookup = {str(d): i for i, d in enumerate(all_dates)}
    ocean = arrays["mask"].astype(bool)
    originals = json.loads((source / "download_manifest.json").read_text())
    raw_lookup = {d["date"]: d for d in originals["files"]}
    store = zarr.open_group(str(exp / "observed_sigma_north.zarr"), mode="r")
    stored_date_index = {
        str(d): i for i, d in enumerate(store["dates"][:].astype("datetime64[D]"))
    }
    require(
        store.attrs["variables"] == ["ice_conc", "total_standard_uncertainty"],
        "Stored variable order differs",
    )
    train_masks, inputs, targets = [], [], []
    for origin in dates["train_origins"]:
        d = np.datetime64(origin)
        previous, target_date = (
            str(d - np.timedelta64(1, "D")),
            str(d + np.timedelta64(1, "D")),
        )
        entry = raw_lookup[target_date]
        path = source / entry["file"]
        require(digest(path) == entry["sha256"], "Training uncertainty source changed")
        with Dataset(path) as nc:
            sigma = (
                np.ma.filled(
                    nc["total_standard_uncertainty"][0, 232:296, 206:270], np.nan
                ).astype(np.float32)
                * 0.01
            )
        stored_sigma = store["data"][stored_date_index[target_date], 1, 0].reshape(
            64, 64
        )
        require(
            np.allclose(stored_sigma, sigma, rtol=1e-6, atol=1e-8, equal_nan=True),
            "Raw uncertainty magnitudes differ from training store",
        )
        use = (
            ocean
            & valid[lookup[previous]]
            & valid[lookup[origin]]
            & valid[lookup[target_date]]
        )
        use &= np.isfinite(sigma) & (sigma > 0) & (sigma <= 1)
        train_masks.append(use)
        inputs.append(
            np.stack(
                [
                    np.where(valid[lookup[x]] & ocean, fields[lookup[x]], 0)
                    for x in (previous, origin)
                ]
            )[:, None]
        )
        targets.append(
            np.where(
                valid[lookup[target_date]] & ocean, fields[lookup[target_date]], 0
            )[None, None]
        )
    train_mask = np.stack(train_masks)[:, None, None]
    require(
        hashlib.sha256(train_mask.tobytes()).hexdigest() == support["mask_sha256"],
        "Training masks differ",
    )
    require(
        hashlib.sha256(np.stack(inputs).tobytes()).hexdigest()
        == support["input_sha256"],
        "Input tensors differ",
    )
    require(
        hashlib.sha256(np.stack(targets).tobytes()).hexdigest()
        == support["target_sha256"],
        "Target tensors differ",
    )
    require(
        int(train_mask.sum()) == support["eligible_pixels"], "Training support differs"
    )
    rebuilt_truth, rebuilt_mask, rebuilt_persistence = [], [], []
    for origin in dates["test_origins"]:
        ref = np.datetime64(origin)
        prior = str(ref - np.timedelta64(1, "D"))
        common = ocean & valid[lookup[prior]] & valid[lookup[origin]]
        truth_list, mask_list = [], []
        for lead in LEADS:
            ti = lookup[str(ref + np.timedelta64(lead, "D"))]
            truth_list.append(np.where(valid[ti] & ocean, fields[ti], 0)[None])
            mask_list.append((common & valid[ti])[None])
        rebuilt_truth.append(np.stack(truth_list))
        rebuilt_mask.append(np.stack(mask_list))
        anchor = np.where(ocean & valid[lookup[origin]], fields[lookup[origin]], 0)
        rebuilt_persistence.append(np.stack([anchor[None]] * len(LEADS)))
    truth = np.stack(rebuilt_truth)
    mask = np.stack(rebuilt_mask)
    persistence = np.stack(rebuilt_persistence)
    return truth, mask, persistence


def audit(source: Path, exp: Path) -> dict[str, Any]:
    """Reconstruct observations, uncertainty and support from independently read files."""
    protocol = json.loads((exp / "protocol.json").read_text())
    complete = json.loads((exp / "complete.json").read_text())
    require(
        digest(source / "observed_sic_paired.npz") == protocol["source_npz_sha256"],
        "Observation bundle changed",
    )
    for name, key in (
        ("forecasts.npz", "forecasts_sha256"),
        ("per_case.csv", "per_case_sha256"),
        ("protocol.json", "protocol_sha256"),
    ):
        require(
            digest(exp / name) == complete[key], "Changed completed output: " + name
        )
    require(
        protocol["seeds"] == list(SEEDS) and protocol["arms"] == list(ARMS),
        "Changed arm/seed definitions",
    )
    records = verify_records(exp)
    dates = json.loads((exp / "dates.json").read_text())
    support = json.loads((exp / "training-support.json").read_text())
    truth, mask, persistence = rebuild_reference_fields(source, exp, dates, support)
    with np.load(exp / "forecasts.npz", allow_pickle=False) as z:
        forecasts = {k: z[k].copy() for k in z.files}
    for name, values in (
        ("truth", truth),
        ("mask", mask),
        ("persistence", persistence),
    ):
        require(
            np.array_equal(forecasts[name], values),
            "Independent field reconstruction mismatch: " + name,
        )
    with (exp / "per_case.csv").open() as stream:
        reported = list(csv.DictReader(stream))
    reported_lookup = {
        (int(r["seed"]), r["arm"], r["selection"], r["origin"], int(r["lead_days"])): r
        for r in reported
    }
    require(
        len(reported_lookup)
        == len(reported)
        == len(SEEDS) * len(ARMS) * 2 * len(dates["test_origins"]) * len(LEADS),
        "Missing or duplicate cases",
    )
    summaries = []
    for record in records:
        seed, arm = record["seed"], record["arm"]
        for selection in ("selected", "last"):
            forecast = forecasts[f"seed{seed}_{arm}_{selection}"]
            require(
                forecast.shape == truth.shape and np.isfinite(forecast).all(),
                "Forecast invalid",
            )
            if record["best_epoch"] == 0 and selection == "selected":
                require(
                    np.array_equal(forecast, persistence),
                    "Epoch-zero anchor is not persistence",
                )
            for j, lead in enumerate(LEADS):
                sums = np.zeros(5, dtype=np.float64)
                for i, origin in enumerate(dates["test_origins"]):
                    m = mask[i, j]
                    error = forecast[i, j][m].astype(np.float64) - truth[i, j][
                        m
                    ].astype(np.float64)
                    pe = persistence[i, j][m].astype(np.float64) - truth[i, j][
                        m
                    ].astype(np.float64)
                    case_values = [
                        np.abs(error).sum(),
                        np.dot(error, error),
                        np.abs(pe).sum(),
                        np.dot(pe, pe),
                        int(m.sum()),
                    ]
                    row = reported_lookup[seed, arm, selection, origin, lead]
                    expected = [
                        float(row[k])
                        for k in (
                            "model_abs_sum",
                            "model_sq_sum",
                            "persistence_abs_sum",
                            "persistence_sq_sum",
                            "n_pixels",
                        )
                    ]
                    require(
                        np.allclose(case_values, expected, rtol=1e-12, atol=1e-10),
                        "Reported score mismatch",
                    )
                    require(
                        row["valid_date"]
                        == str(np.datetime64(origin) + np.timedelta64(lead, "D")),
                        "Calendar lead mismatch",
                    )
                    sums += case_values
                a, squared, pa, ps, count = sums
                summaries.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "selection": selection,
                        "lead_days": lead,
                        "n_origins": len(dates["test_origins"]),
                        "n_pixels": int(count),
                        "mae_pp": float(a / count * 100),
                        "rmse_pp": float(np.sqrt(squared / count) * 100),
                        "persistence_mae_pp": float(pa / count * 100),
                        "persistence_rmse_pp": float(np.sqrt(ps / count) * 100),
                        "mae_skill": float(1 - a / pa),
                        "mse_skill": float(1 - squared / ps),
                    }
                )
    return {
        "protocol": protocol,
        "verified_raw_source_and_training_support": True,
        "independent_case_recomputations": len(reported),
        "frozen_checkpoints": records,
        "training_support": support,
        "dates": dates,
        "metrics": summaries,
    }


def main() -> None:
    """Audit a completed experiment without changing any experiment output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Do not overwrite a previous audit")
    result = audit(args.source, args.experiment)
    args.out.mkdir(parents=True)
    (args.out / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(result["metrics"][0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(result["metrics"])
    log.info(
        "VERIFIED %s cases and ten paired training runs",
        result["independent_case_recomputations"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
