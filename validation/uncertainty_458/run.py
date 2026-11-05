"""Matched uncertainty-weighted Huber versus Huber on real OSI SAF observations.

Five paired seeds have identical input tensors, initial model states, training
pixels, minibatch orders and update budgets. Unweighted January validation
selects checkpoints before any February scoring. This previously explored
regional holdout is retrospective, not an untouched confirmatory test set.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from netCDF4 import Dataset
from omegaconf import DictConfig

from cryocast.data.combined_dataset import CombinedDataset
from cryocast.data.single_dataset import SingleDataset
from cryocast.models import EncodeProcessDecode

log = logging.getLogger(__name__)
SOURCE_SHA256 = "949a176a64a7e7dece6b8953cbb740fcaf38900bd58a57efe6aa6dc1961dd9b1"
SEEDS = (458, 459, 460, 461, 462)
ARMS = ("huber", "weighted")
LEADS = (1, 7, 14)
WIDTH, BATCH, EPOCHS, VALIDATE_EVERY = 8, 8, 120, 5
LR, WEIGHT_DECAY, DELTA = 1e-4, 1e-4, 0.5
N_OBSERVATIONS, N_OCEAN_PIXELS = 180, 3584
DAY = np.timedelta64(1, "D")
SIGMA_VARIABLE = "total_standard_uncertainty"


def require(condition: object, message: str) -> None:
    """Stop rather than silently accept invalid or unmatched evidence."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash source observations and frozen outputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    """Write finite JSON atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def make_store(source: Path, out: Path) -> tuple[SingleDataset, dict[str, Any]]:  # noqa: PLR0915 - explicit source checks and store creation
    """Read genuine uncertainty from checksum-verified raw NetCDFs, without downloads."""
    original = source / "observed_sic_paired.npz"
    require(sha(original) == SOURCE_SHA256, "Unexpected observation-bundle checksum")
    with np.load(original, allow_pickle=False) as bundle:
        data = {k: bundle[k].copy() for k in bundle.files}
    fields = np.concatenate([data[k] for k in ("train", "val", "test")])
    valid = np.concatenate([data[k + "_valid"] for k in ("train", "val", "test")])
    dates = np.concatenate(
        [data[k + "_dates"] for k in ("train", "val", "test")]
    ).astype("datetime64[D]")
    mask = data["mask"].astype(bool)
    require(len(set(dates)) == len(dates) == N_OBSERVATIONS, "Unexpected dates")
    require(int(mask.sum()) == N_OCEAN_PIXELS, "Ocean mask changed")
    manifest = json.loads((source / "download_manifest.json").read_text())
    by_date = {row["date"]: row for row in manifest["files"]}
    sigmas, checked = [], []
    lat, lon = None, None
    for index, date in enumerate(dates):
        entry = by_date[str(date)]
        path = source / entry["file"]
        require(sha(path) == entry["sha256"], "Raw source file checksum mismatch")
        with Dataset(path) as nc:
            variable = nc[SIGMA_VARIABLE]
            require(variable.units == "%", "Unexpected raw uncertainty units")
            sigma = (
                np.ma.filled(variable[0, 232:296, 206:270], np.nan).astype(np.float32)
                / 100
            )
            sic = (
                np.ma.filled(nc["ice_conc"][0, 232:296, 206:270], np.nan).astype(
                    np.float32
                )
                / 100
            )
            require(
                np.allclose(sic[valid[index]], fields[index][valid[index]], atol=1e-7),
                "Uncertainty and SIC source alignment differs",
            )
            if lat is None:
                lat = np.asarray(nc["lat"][:])[232:296, 206:270].reshape(-1)
                lon = np.asarray(nc["lon"][:])[232:296, 206:270].reshape(-1)
        sigmas.append(sigma)
        checked.append(
            {"date": str(date), "file": entry["file"], "sha256": entry["sha256"]}
        )
    uncertainty = np.stack(sigmas)
    lookup = {str(date): i for i, date in enumerate(dates)}
    calendar = np.arange(dates[0], dates[-1] + DAY, DAY)
    missing = [str(d) + "T12:00:00" for d in calendar if str(d) not in lookup]
    values = np.zeros((len(calendar), 2, 1, 4096), dtype=np.float32)
    for i, date in enumerate(calendar):
        idx = lookup.get(str(date))
        if idx is not None:
            values[i, 0, 0] = np.where(valid[idx] & mask, fields[idx], 0).ravel()
            values[i, 1, 0] = uncertainty[idx].ravel()
    store_path = out / "observed_sigma_north.zarr"
    require(not store_path.exists(), "Refusing to overwrite an existing store")
    store = zarr.open_group(str(store_path), mode="w")
    store.create_dataset("data", data=values, chunks=(1, 2, 1, 4096))
    store.create_dataset(
        "dates", data=calendar.astype("datetime64[s]") + np.timedelta64(12, "h")
    )
    store.create_dataset("latitudes", data=lat)
    store.create_dataset("longitudes", data=lon)
    # Only training observations contribute to statistics. Both channels are read
    # in physical units; these statistics cannot rescale or leak held-out targets.
    ntrain = len(data["train"])
    train_mask = valid[:ntrain] & mask
    train_sigma = uncertainty[:ntrain][train_mask & np.isfinite(uncertainty[:ntrain])]
    train_sic = fields[:ntrain][train_mask]
    for name, vals in {
        "mean": [train_sic.mean(), train_sigma.mean()],
        "stdev": [train_sic.std(), train_sigma.std()],
        "minimum": [0.0, 0.0],
        "maximum": [1.0, 1.0],
    }.items():
        store.create_dataset(name, data=np.array(vals, dtype=np.float64))
    store.attrs.update(
        {
            "field_shape": [64, 64],
            "frequency": "24h",
            "variables": ["ice_conc", SIGMA_VARIABLE],
            "missing_dates": missing,
            "flatten_grid": True,
            "ensemble_dimension": 2,
        }
    )
    # Sigma is deliberately absent from the selected model-input variables.
    single = SingleDataset(
        name="sic", input_files=[store_path], variables=["ice_conc"], normalise=False
    )
    require(single.variable_names == ["ice_conc"], "Uncertainty leaked into inputs")
    save_json(
        out / "raw-source-manifest.json",
        {
            "files": checked,
            "uncertainty_units": "fraction",
            "raw_units": "%",
            "conversion": "divide by 100 after NetCDF CF decoding",
        },
    )
    return single, {
        "fields": fields,
        "valid": valid,
        "dates": dates,
        "lookup": lookup,
        "mask": mask,
        "sigma": uncertainty,
    }


def combined(
    single: SingleDataset, period: tuple[str, str], forecasts: int
) -> CombinedDataset:
    """Exercise PR #458's physical uncertainty and selected-input data path."""
    return CombinedDataset(
        [single.subset(date_ranges=[{"start": period[0], "end": period[1]}])],
        target_group_name="sic",
        target_variables=["ice_conc"],
        n_history_steps=2,
        n_forecast_steps=forecasts,
        target_uncertainty_variable=SIGMA_VARIABLE,
    )


def origin_map(ds: CombinedDataset) -> dict[str, int]:
    """Identify forecast origins by the most recent observation."""
    return {
        str(ds.get_history_steps(d)[-1].astype("datetime64[D]")): i
        for i, d in enumerate(ds.dates)
    }


def extract(
    ds: CombinedDataset, origins: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load input, future SIC and future sigma separately through the actual PR."""
    mapping = origin_map(ds)
    batches = [ds[mapping[o]] for o in origins]
    require(
        all(set(b) == {"sic", "target", "target_uncertainty"} for b in batches),
        "Unexpected input channels",
    )
    return tuple(
        torch.from_numpy(np.stack([b[k] for b in batches]))
        for k in ("sic", "target", "target_uncertainty")
    )  # type: ignore[return-value]


def masks_for(
    info: dict[str, Any], origins: list[str], leads: tuple[int, ...]
) -> np.ndarray:
    """Use identical issue-time and truth validity masks, independent of predictions."""
    masks = []
    for origin in origins:
        date = np.datetime64(origin)
        history = info["mask"].copy()
        for lag in (-1, 0):
            history &= info["valid"][info["lookup"][str(date + lag * DAY)]]
        masks.append(
            np.stack(
                [
                    history & info["valid"][info["lookup"][str(date + lead * DAY)]]
                    for lead in leads
                ]
            )
        )
    return np.stack(masks)[:, :, None]


def model_for(seed: int, arm: str) -> EncodeProcessDecode:
    """Build identically initialised physical-rollout models; only the loss varies."""
    torch.manual_seed(seed)
    space = DictConfig({"name": "sic", "channels": 1, "shape": [64, 64]})
    model = EncodeProcessDecode(
        name="uncertainty_458_small_unet",
        hemisphere="north",
        input_spaces=[space],
        output_space=space,
        n_history_steps=2,
        n_forecast_steps=1,
        encoders=DictConfig(
            {
                "latent_space": [64, 64],
                "sic": {"_target_": "cryocast.models.encoders.NaiveLinearEncoder"},
            }
        ),
        processor=DictConfig(
            {
                "_target_": "cryocast.models.processors.UNetProcessor",
                "start_out_channels": WIDTH,
                "kernel_size": 3,
                "norm_type": "batchnorm",
            }
        ),
        decoder=DictConfig(
            {
                "_target_": "cryocast.models.decoders.NaiveLinearDecoder",
                "mask_type": "none",
                "restrict_range": "none",
                "skip_connection": {"method": "additive"},
            }
        ),
        target_variable_indices=[0],
        rollout_space="physical",
        metrics=[],
        optimizer=DictConfig({}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
        loss=DictConfig(
            {"_target_": "cryocast.losses.UncertaintyWeightedLoss", "delta": DELTA}
            if arm == "weighted"
            else {"_target_": "torch.nn.HuberLoss", "delta": DELTA}
        ),
    )
    # Both arms start from the same explicit persistence residual anchor.
    layers = model.decoder.model
    if not isinstance(layers, torch.nn.Sequential):
        msg = "Unexpected decoder structure."
        raise TypeError(msg)
    readout = layers[0]
    if not isinstance(readout, torch.nn.Conv2d) or readout.bias is None:
        msg = "Expected a biased convolutional decoder readout."
        raise TypeError(msg)
    torch.nn.init.zeros_(readout.weight)
    torch.nn.init.zeros_(readout.bias)
    return model


def weights_hash(model: EncodeProcessDecode) -> str:
    """Hash the entire state, including batch-normalisation buffers."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict(model: EncodeProcessDecode, x: torch.Tensor) -> np.ndarray:
    """Roll out without future observations, selecting matching calendar leads."""
    previous = model.n_forecast_steps
    model.n_forecast_steps = max(LEADS)
    model.eval()
    with torch.inference_mode():
        output = torch.cat([model({"sic": batch}) for batch in x.split(BATCH)], dim=0)
    model.n_forecast_steps = previous
    selected = output[:, [lead - 1 for lead in LEADS]].cpu().numpy()
    require(
        bool(np.isfinite(selected).all()),
        "Nonfinite forecast; do not hide failed predictions",
    )
    return selected


def scores(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[float, float, int]:
    """Return unweighted absolute and squared error sums on common pixels."""
    require(pred.shape == target.shape == mask.shape, "Misaligned score arrays")
    difference = pred.astype(np.float64)[mask] - target.astype(np.float64)[mask]
    require(
        len(difference) > 0 and np.isfinite(difference).all(),
        "Invalid predictions or masks",
    )
    return (
        float(np.abs(difference).sum()),
        float(np.square(difference).sum()),
        len(difference),
    )


def train_one(
    seed: int,
    arm: str,
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
    val: tuple[torch.Tensor, np.ndarray, np.ndarray],
    out: Path,
) -> dict[str, Any]:
    """Run an equal budget, selecting on identical unweighted validation MSE."""
    model = model_for(seed, arm)
    initial = weights_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x, y, sigma = train
    vx, vy, vm = val
    best, best_epoch, updates = float("inf"), -1, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.perf_counter()
    order_digest = hashlib.sha256()
    for epoch in range(EPOCHS + 1):
        training_loss = None
        if epoch:
            model.train()
            order = torch.randperm(
                len(x), generator=torch.Generator().manual_seed(seed * 1000 + epoch)
            )
            order_digest.update(order.numpy().tobytes())
            loss_sum = 0.0
            for ids in order.split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                pred = model({"sic": x[ids]})
                use = mask[ids]
                # Both arms see exactly the same pixels; only the relative sigma
                # weights differ. Future sigma is loss-only, never a model input.
                loss = model.loss(pred[use], y[ids][use], sigma[ids][use])
                require(bool(torch.isfinite(loss)), "Nonfinite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                loss_sum += float(loss.detach())
                updates += 1
            training_loss = loss_sum / len(order.split(BATCH))
        if epoch % VALIDATE_EVERY == 0:
            forecast = predict(model, vx)
            _, squared, count = scores(forecast, vy, vm)
            mse = squared / count
            history.append(
                {
                    "epoch": epoch,
                    "updates": updates,
                    "validation_mse": mse,
                    "training_objective": training_loss,
                }
            )
            if mse < best:
                best, best_epoch = mse, epoch
                best_state = copy.deepcopy(model.state_dict())
            log.info("seed=%s arm=%s epoch=%s val_mse=%.7f", seed, arm, epoch, mse)
    last_state_sha = weights_hash(model)
    last_path = out / f"seed{seed}-{arm}-last.pt"
    torch.save(model.state_dict(), last_path)
    model.load_state_dict(best_state)
    path = out / f"seed{seed}-{arm}-selected.pt"
    torch.save(best_state, path)
    return {
        "seed": seed,
        "arm": arm,
        "parameters": sum(p.numel() for p in model.parameters()),
        "updates": updates,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_mse": best,
        "initial_state_sha256": initial,
        "training_order_sha256": order_digest.hexdigest(),
        "selected_state_sha256": weights_hash(model),
        "last_state_sha256": last_state_sha,
        "selected_checkpoint_sha256": sha(path),
        "last_checkpoint_sha256": sha(last_path),
        "seconds": time.perf_counter() - started,
        "history": history,
    }


def evaluate_frozen(
    records: list[dict[str, Any]],
    out: Path,
    ds: CombinedDataset,
    origins: list[str],
    info: dict[str, Any],
) -> None:
    """Score all already-frozen checkpoints with unweighted metrics and persistence."""
    x, target, _sigma = extract(ds, origins)
    truth = target[:, [d - 1 for d in LEADS]].numpy()
    mask = masks_for(info, origins, LEADS)
    persistence = np.repeat(x[:, -1:].numpy(), len(LEADS), axis=1)
    forecasts: dict[str, Any] = {
        "truth": truth,
        "mask": mask,
        "persistence": persistence,
    }
    rows = []
    for r in records:
        seed, arm = r["seed"], r["arm"]
        for selection in ("selected", "last"):
            model = model_for(seed, arm)
            path = out / f"seed{seed}-{arm}-{selection}.pt"
            require(
                sha(path) == r[selection + "_checkpoint_sha256"],
                "Checkpoint changed after selection",
            )
            model.load_state_dict(torch.load(path, weights_only=True))
            require(
                weights_hash(model) == r[selection + "_state_sha256"],
                "Weight content mismatch",
            )
            pred = predict(model, x)
            forecasts[f"seed{seed}_{arm}_{selection}"] = pred
            for i, origin in enumerate(origins):
                for j, lead in enumerate(LEADS):
                    absolute, squared, count = scores(
                        pred[i, j], truth[i, j], mask[i, j]
                    )
                    pa, ps, pc = scores(persistence[i, j], truth[i, j], mask[i, j])
                    require(pc == count, "Unequal pixel support")
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "selection": selection,
                            "origin": origin,
                            "lead_days": lead,
                            "valid_date": str(np.datetime64(origin) + lead * DAY),
                            "n_pixels": count,
                            "model_abs_sum": absolute,
                            "model_sq_sum": squared,
                            "persistence_abs_sum": pa,
                            "persistence_sq_sum": ps,
                        }
                    )
    with (out / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "forecasts.npz", **forecasts)
    save_json(
        out / "complete.json",
        {
            "runs": len(records),
            "selections_per_run": 2,
            "cases": len(rows),
            "forecasts_sha256": sha(out / "forecasts.npz"),
            "per_case_sha256": sha(out / "per_case.csv"),
            "protocol_sha256": sha(out / "protocol.json"),
        },
    )
    log.info("COMPLETE: ten matched runs, selected and final checkpoints, 15 origins")


def main() -> None:
    """Freeze the protocol, train all models, then inspect test forecasts once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(
        not out.exists(), "Use a fresh output directory; previous runs are immutable"
    )
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(mode=True)
    git = shutil.which("git")
    require(git, "Git is required for source provenance")
    revision = subprocess.check_output(  # noqa: S603 - fixed local git arguments
        [str(git), "rev-parse", "HEAD"], text=True
    ).strip()
    save_json(
        out / "protocol.json",
        {
            "source_revision": revision,
            "script_sha256": sha(Path(__file__)),
            "source_npz_sha256": SOURCE_SHA256,
            "seeds": SEEDS,
            "arms": ARMS,
            "leads_days": LEADS,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "width": WIDTH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "delta": DELTA,
            "sigma_min": 0.01,
            "sigma_max": 1.0,
            "sigma_power": 2.0,
            "selection": "minimum unweighted January 1/7/14-day MSE, including epoch-zero persistence",
            "selection_frozen_before_test": True,
            "test_previously_explored": True,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": 2,
        },
    )
    single, info = make_store(args.source, out)
    training = combined(single, ("2023-01-01", "2023-04-30"), 1)
    validation = combined(single, ("2024-01-01", "2024-01-31"), max(LEADS))
    testing = combined(single, ("2024-01-31", "2024-02-29"), max(LEADS))
    train_origins = sorted(origin_map(training))
    val_origins = sorted(origin_map(validation))
    test_origins = sorted(o for o in origin_map(testing) if o >= "2024-02-01")
    require(
        (len(train_origins), len(val_origins), len(test_origins)) == (118, 16, 15),
        "Unexpected split eligibility",
    )
    train = extract(training, train_origins)
    sigma = train[2]
    common_mask = (
        torch.from_numpy(masks_for(info, train_origins, (1,)))
        & torch.isfinite(sigma)
        & (sigma > 0)
        & (sigma <= 1)
    )
    require(
        bool(torch.all(common_mask.sum(dim=(1, 2, 3, 4)) > 0)), "Empty training sample"
    )
    vx, vy, _ = extract(validation, val_origins)
    val = (
        vx,
        vy[:, [d - 1 for d in LEADS]].numpy(),
        masks_for(info, val_origins, LEADS),
    )
    weight = sigma[common_mask].clamp_min(0.01).pow(-2).double().numpy()
    save_json(
        out / "dates.json",
        {
            "train_origins": train_origins,
            "val_origins": val_origins,
            "test_origins": test_origins,
            "history_offsets_days": [-1, 0],
            "leads_days": LEADS,
        },
    )
    save_json(
        out / "training-support.json",
        {
            "eligible_pixels": int(common_mask.sum()),
            "valid_sic_pixels": int(masks_for(info, train_origins, (1,)).sum()),
            "global_weight_effective_sample_fraction": float(
                weight.sum() ** 2 / (np.square(weight).sum() * len(weight))
            ),
            "weight_p01_p50_p99": np.quantile(weight, [0.01, 0.5, 0.99]).tolist(),
            "input_sha256": hashlib.sha256(train[0].numpy().tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(train[1].numpy().tobytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(common_mask.numpy().tobytes()).hexdigest(),
        },
    )
    records = []
    for seed in SEEDS:
        for arm in ARMS:
            r = train_one(seed, arm, train, common_mask, val, out)
            records.append(r)
            save_json(out / f"train-{seed}-{arm}.json", r)
        a, b = records[-2:]
        for key in (
            "initial_state_sha256",
            "training_order_sha256",
            "updates",
            "parameters",
        ):
            require(a[key] == b[key], "Unmatched paired training condition: " + key)
    save_json(out / "frozen_checkpoints.json", records)
    evaluate_frozen(records, out, testing, test_origins, info)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
