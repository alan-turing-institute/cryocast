"""Compare daily and weekly CryoCast models at identical physical lead times.

Use only the checksum-identified, previously published OSI SAF crop. Models
are selected using January validation targets; February labels are evaluated
only after every trained checkpoint has been frozen. No cloud jobs are used.
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
N_OBSERVATIONS = 180
N_OCEAN_PIXELS = 3584
SEEDS = (496, 497, 498)
LEADS = (7, 14)
STRIDES = (1, 7)
HISTORY = 2
WIDTH = 8
BATCH = 8
EPOCHS = 60
VALIDATE_EVERY = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
DAY = np.timedelta64(1, "D")


def require(condition: object, message: str) -> None:
    """Reject incomplete, inconsistent or accidentally mismatched evidence."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Return an input or output file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    """Atomically write a finite JSON record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_store(source: Path, out: Path) -> tuple[SingleDataset, dict[str, Any]]:
    """Represent actual observed dates in an Anemoi store, including native gaps."""
    original = source / "observed_sic_paired.npz"
    require(sha(original) == SOURCE_SHA256, "Unexpected source observations checksum")
    with np.load(original, allow_pickle=False) as bundle:
        data = {key: bundle[key].copy() for key in bundle.files}
    fields = np.concatenate([data[key] for key in ("train", "val", "test")])
    valid = np.concatenate([data[key + "_valid"] for key in ("train", "val", "test")])
    dates = np.concatenate(
        [data[key + "_dates"] for key in ("train", "val", "test")]
    ).astype("datetime64[D]")
    require(
        len(dates) == len(set(dates)) == N_OBSERVATIONS, "Unexpected observation dates"
    )
    require(
        bool(np.all(np.diff(dates) > np.timedelta64(0, "D"))), "Unsorted observations"
    )
    mask = data["mask"].astype(bool)
    require(
        mask.shape == (64, 64) and int(mask.sum()) == N_OCEAN_PIXELS,
        "Wrong training-derived ocean mask",
    )
    require(bool(np.isfinite(fields).all()), "Nonfinite prepared SIC")
    lookups = {str(date): index for index, date in enumerate(dates)}
    calendar = np.arange(dates[0], dates[-1] + DAY, DAY)
    missing = [str(date) + "T12:00:00" for date in calendar if str(date) not in lookups]
    store_path = out / "observed_north.zarr"
    require(
        not store_path.exists(),
        "Output store already exists; use a fresh output directory",
    )
    physical = np.zeros((len(calendar), 1, 1, 4096), dtype=np.float32)
    for offset, date in enumerate(calendar):
        index = lookups.get(str(date))
        if index is not None:
            physical[offset, 0, 0] = np.where(
                valid[index] & mask, fields[index], 0
            ).reshape(-1)
    raw = source / "raw/training/ice_conc_nh_ease2-250_icdr-v3p0-amsr_202301011200.nc"
    with Dataset(raw) as netcdf:
        lat = np.asarray(netcdf.variables["lat"][:])[232:296, 206:270].reshape(-1)
        lon = np.asarray(netcdf.variables["lon"][:])[232:296, 206:270].reshape(-1)
    store = zarr.open_group(str(store_path), mode="w")
    store.create_dataset("data", data=physical, chunks=(1, 1, 1, 4096))
    store.create_dataset(
        "dates", data=calendar.astype("datetime64[s]") + np.timedelta64(12, "h")
    )
    store.create_dataset("latitudes", data=lat)
    store.create_dataset("longitudes", data=lon)
    training = data["train"][data["train_valid"] & mask]
    for name, value in {
        "mean": training.mean(),
        "stdev": training.std(),
        "minimum": 0.0,
        "maximum": 1.0,
    }.items():
        store.create_dataset(name, data=np.asarray([value], dtype=np.float64))
    store.attrs.update(
        {
            "field_shape": [64, 64],
            "frequency": "24h",
            "variables": ["ice_conc"],
            "missing_dates": missing,
            "flatten_grid": True,
            "ensemble_dimension": 2,
        }
    )
    single = SingleDataset(name="sic", input_files=[store_path], normalise=False)
    require(
        len(single.dates) == N_OBSERVATIONS, "Native missing dates were not excluded"
    )
    information = {
        "fields": fields,
        "valid": valid,
        "dates": dates,
        "lookup": lookups,
        "mask": mask,
    }
    return single, information


def combined(
    single: SingleDataset, stride: int, period: tuple[str, str], forecasts: int
) -> CombinedDataset:
    """Use the PR's actual date-selection and data-reading path."""
    return CombinedDataset(
        [single.subset(date_ranges=[{"start": period[0], "end": period[1]}])],
        target_group_name="sic",
        target_variables=["ice_conc"],
        n_history_steps=HISTORY,
        n_forecast_steps=forecasts,
        step_stride=stride,
    )


def origin_map(dataset: CombinedDataset) -> dict[str, int]:
    """Map the last observed date, rather than the first history date, to samples."""
    return {
        str(dataset.get_history_steps(date)[-1].astype("datetime64[D]")): i
        for i, date in enumerate(dataset.dates)
    }


def extract(
    dataset: CombinedDataset, origins: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read identical origin lists through the real CombinedDataset interface."""
    mapping = origin_map(dataset)
    batches = [dataset[mapping[origin]] for origin in origins]
    return torch.from_numpy(np.stack([b["sic"] for b in batches])), torch.from_numpy(
        np.stack([b["target"] for b in batches])
    )


def masks_for(
    info: dict[str, Any], origins: list[str], leads: tuple[int, ...]
) -> np.ndarray:
    """Require the same valid history and truth pixels for every compared method."""
    masks = []
    for origin in origins:
        reference = np.datetime64(origin)
        history_mask = info["mask"].copy()
        for lag in (0, -1, -7):
            history_mask &= info["valid"][info["lookup"][str(reference + lag * DAY)]]
        masks.append(
            np.stack(
                [
                    history_mask
                    & info["valid"][info["lookup"][str(reference + lead * DAY)]]
                    for lead in leads
                ]
            )
        )
    result = np.stack(masks)[:, :, None]
    require(bool(np.all(result.sum(axis=(-1, -2, -3)) > 0)), "Empty comparison mask")
    return result


def model_for(seed: int) -> EncodeProcessDecode:
    """Build the same small physical-rollout CryoCast model for both stride arms."""
    torch.manual_seed(seed)
    space = DictConfig({"name": "sic", "channels": 1, "shape": [64, 64]})
    model = EncodeProcessDecode(
        name="weekly_496_small_unet",
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
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
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


def predict(model: EncodeProcessDecode, x: torch.Tensor, stride: int) -> np.ndarray:
    """Roll out without future observations, selecting matching calendar leads."""
    previous = model.n_forecast_steps
    model.n_forecast_steps = max(LEADS) // stride
    model.eval()
    with torch.inference_mode():
        output = torch.cat([model({"sic": batch}) for batch in x.split(BATCH)], dim=0)
    model.n_forecast_steps = previous
    selected = output[:, [lead // stride - 1 for lead in LEADS]].cpu().numpy()
    require(
        bool(np.isfinite(selected).all()),
        "Nonfinite forecast; do not hide failed predictions",
    )
    return selected


def scores(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> tuple[float, float, int]:
    """Calculate independently interpretable pixel sums on an identical mask."""
    require(
        prediction.shape == truth.shape == mask.shape, "Metric-array shape mismatch"
    )
    difference = prediction.astype(np.float64)[mask] - truth.astype(np.float64)[mask]
    return (
        float(np.abs(difference).sum()),
        float(np.square(difference).sum()),
        len(difference),
    )


def train_one(
    seed: int,
    stride: int,
    train: tuple[torch.Tensor, torch.Tensor],
    train_mask: torch.Tensor,
    val: tuple[torch.Tensor, np.ndarray, np.ndarray],
    out: Path,
) -> dict[str, Any]:
    """Train a fixed equal update budget; use validation only to select checkpoints."""
    model = model_for(seed)
    initial = weights_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x, y = train
    val_x, val_y, val_mask = val
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    history = []
    update = 0
    started = time.perf_counter()
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            order = torch.randperm(
                len(x), generator=torch.Generator().manual_seed(seed * 1000 + epoch)
            )
            train_sum, train_count = 0.0, 0
            for indices in order.split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                prediction = model({"sic": x[indices]})
                use = train_mask[indices]
                loss = torch.square(prediction - y[indices])[use].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_sum += float(loss.detach()) * int(use.sum())
                train_count += int(use.sum())
                update += 1
        if epoch % VALIDATE_EVERY == 0:
            result = predict(model, val_x, stride)
            _, square, count = scores(result, val_y, val_mask)
            loss_value = square / count
            history.append(
                {"epoch": epoch, "updates": update, "validation_mse": loss_value}
            )
            if loss_value < best_loss:
                best_loss, best_epoch = loss_value, epoch
                best_state = copy.deepcopy(model.state_dict())
            log.info(
                "seed=%s stride=%s epoch=%s validation_mse=%.7f",
                seed,
                stride,
                epoch,
                loss_value,
            )
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = out / f"seed{seed}-stride{stride}.pt"
    torch.save(best_state, checkpoint)
    return {
        "seed": seed,
        "stride": stride,
        "parameters": sum(p.numel() for p in model.parameters()),
        "updates": update,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "initial_state_sha256": initial,
        "best_state_sha256": weights_hash(model),
        "checkpoint_sha256": sha(checkpoint),
        "train_seconds": time.perf_counter() - started,
        "history": history,
    }


def evaluate_frozen(
    records: list[dict[str, Any]],
    out: Path,
    testing: dict[int, CombinedDataset],
    test_origins: list[str],
    info: dict[str, Any],
    eligibility: dict[str, Any],
) -> None:
    """Evaluate locked checkpoints against unchanged, matched persistence fields."""
    # Test extraction/scoring occurs after all checkpoint choices are frozen.
    test_data = {s: extract(testing[s], test_origins) for s in STRIDES}
    truth = test_data[7][1].numpy()
    require(
        bool(np.array_equal(test_data[1][1][:, [6, 13]].numpy(), truth)),
        "Test targets differ",
    )
    require(
        bool(
            np.array_equal(
                test_data[1][0][:, -1].numpy(), test_data[7][0][:, -1].numpy()
            )
        ),
        "Different persistence origins",
    )
    mask = masks_for(info, test_origins, LEADS)
    persistence = np.repeat(test_data[7][0][:, -1:].numpy(), len(LEADS), axis=1)
    rows = []
    predictions: dict[str, Any] = {}
    for record in records:
        seed, stride = record["seed"], record["stride"]
        model = model_for(seed)
        checkpoint = out / f"seed{seed}-stride{stride}.pt"
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        require(
            weights_hash(model) == record["best_state_sha256"],
            "Checkpoint hash differs",
        )
        forecast = predict(model, test_data[stride][0], stride)
        predictions[f"seed{seed}_stride{stride}"] = forecast
        for i, origin in enumerate(test_origins):
            for j, lead in enumerate(LEADS):
                absolute, square, count = scores(
                    forecast[i, j], truth[i, j], mask[i, j]
                )
                p_absolute, p_square, p_count = scores(
                    persistence[i, j], truth[i, j], mask[i, j]
                )
                require(count == p_count, "Persistence mask differs")
                rows.append(
                    {
                        "seed": seed,
                        "stride": stride,
                        "origin": origin,
                        "lead_days": lead,
                        "valid_date": eligibility["test_targets"][i][j],
                        "n_pixels": count,
                        "model_abs_sum": absolute,
                        "model_sq_sum": square,
                        "persistence_abs_sum": p_absolute,
                        "persistence_sq_sum": p_square,
                    }
                )
    with (out / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out / "forecasts.npz",
        **predictions,
        truth=truth,
        mask=mask,
        persistence=persistence,
    )
    save_json(
        out / "complete.json",
        {
            "runs": len(records),
            "cases_per_run": len(test_origins) * len(LEADS),
            "forecast_sha256": sha(out / "forecasts.npz"),
            "per_case_sha256": sha(out / "per_case.csv"),
            "protocol_sha256": sha(out / "protocol.json"),
        },
    )
    log.info("COMPLETE: six matched runs, 15 test origins, 7/14-day leads")


def main() -> None:
    """Freeze the protocol, run both arms and only then evaluate held-out targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    require(
        not out.exists(), "Use a new output directory to preserve prior experiments"
    )
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(mode=True)
    git = shutil.which("git")
    if git is None:
        msg = "Git is required to identify the source revision."
        raise RuntimeError(msg)
    source_revision = subprocess.check_output(  # noqa: S603 - fixed local git query
        [git, "rev-parse", "HEAD"], text=True
    ).strip()
    protocol = {
        "source_sha256": SOURCE_SHA256,
        "script_sha256": sha(Path(__file__)),
        "source_revision": source_revision,
        "seeds": SEEDS,
        "strides": STRIDES,
        "physical_leads_days": LEADS,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "optimizer": "Adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "width": WIDTH,
        "checkpoint_selection": "minimum equal-mask January 7/14-day MSE including epoch-zero persistence anchor",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "zarr": zarr.__version__,
        "device": "cpu",
        "threads": 2,
        "no_test_selection": True,
    }
    save_json(out / "protocol.json", protocol)
    single, info = make_store(args.source, out)
    training = {
        s: combined(single, s, ("2023-01-01", "2023-04-30"), 1) for s in STRIDES
    }
    validation = {
        s: combined(single, s, ("2024-01-01", "2024-01-31"), max(LEADS) // s)
        for s in STRIDES
    }
    testing = {
        s: combined(single, s, ("2024-01-25", "2024-02-29"), max(LEADS) // s)
        for s in STRIDES
    }
    train_origins = sorted(set(origin_map(training[1])) & set(origin_map(training[7])))
    val_origins = sorted(
        set(origin_map(validation[1])) & set(origin_map(validation[7]))
    )
    test_origins = sorted(
        date
        for date in set(origin_map(testing[1])) & set(origin_map(testing[7]))
        if date >= "2024-02-01"
    )
    require(
        (len(train_origins), len(val_origins), len(test_origins)) == (106, 10, 15),
        "Unexpected split/date eligibility",
    )
    train_data = {s: extract(training[s], train_origins) for s in STRIDES}
    val_data = {s: extract(validation[s], val_origins) for s in STRIDES}
    # Truth is shared at matching leads, irrespective of model state frequency.
    val_truth = val_data[7][1].numpy()
    require(
        bool(np.array_equal(val_data[1][1][:, [6, 13]].numpy(), val_truth)),
        "Validation targets differ",
    )
    val_mask = masks_for(info, val_origins, LEADS)
    common_train_mask = masks_for(info, train_origins, (1, 7)).all(
        axis=1, keepdims=True
    )
    train_mask = torch.from_numpy(common_train_mask)
    eligibility = {
        "train_origins": train_origins,
        "validation_origins": val_origins,
        "test_origins": test_origins,
        "history_offsets": {"daily": [-1, 0], "weekly": [-7, 0]},
        "forecast_offsets": LEADS,
        "test_targets": [
            [str(np.datetime64(o) + lead * DAY) for lead in LEADS] for o in test_origins
        ],
    }
    save_json(out / "dates.json", eligibility)
    records = []
    for seed in SEEDS:
        for stride in STRIDES:
            record = train_one(
                seed,
                stride,
                train_data[stride],
                train_mask,
                (val_data[stride][0], val_truth, val_mask),
                out,
            )
            records.append(record)
            save_json(out / f"training-seed{seed}-stride{stride}.json", record)
        pair = records[-2:]
        require(
            pair[0]["initial_state_sha256"] == pair[1]["initial_state_sha256"],
            "Unmatched initialization",
        )
        require(pair[0]["updates"] == pair[1]["updates"], "Unmatched compute budget")
    save_json(out / "frozen_checkpoints.json", records)
    evaluate_frozen(records, out, testing, test_origins, info, eligibility)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
