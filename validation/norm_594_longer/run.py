"""Continue the six frozen 96-update normalization runs on full hemispheric grids.

This is a paired continuation study, not a from-scratch convergence experiment.
Weights and AdamW moments are restored. A predeclared cosine learning-rate restart
is applied identically to both arms because the original one-epoch scheduler has
already reduced its learning rate to 1e-6. Test data are scored only after every
run's validation selection is frozen. Original datasets/checkpoints are read-only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anemoi.datasets
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from cryocast.data.combined_dataset import CombinedDataset
from cryocast.model_service import ModelService
from cryocast.models import EncodeProcessDecode
from validation.norm_594_longer.metrics import (
    EXTRA_STEPS,
    FINAL_LR,
    INITIAL_STEPS,
    LEADS,
    NORMS,
    RESTART_LR,
    SEEDS,
    VALIDATE_EVERY,
    continuation_lr,
    per_lead_metrics,
    require,
    training_order,
)

log = logging.getLogger(__name__)
REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTED_PARAMETERS = 1_256_655
HISTORICAL_TERMINAL_LR = 1e-6
SHAPE = (432, 432)
VALIDATION_POSITIONS = (0, 5, 9, 14)


def sha(path: Path) -> str:
    """Hash a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    """Write finite JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config_for(data_root: Path, seed: int, norm: str) -> DictConfig:
    """Retain the previous full-grid model, inputs, loss and fourteen-day rollout."""
    with initialize_config_dir(
        config_dir=str(REPOSITORY / "cryocast/config"), version_base=None
    ):
        cfg = compose(
            config_name="baseline/cnn_unet_cnn",
            overrides=[
                f"base_path={data_root}",
                "data=sample_south",
                "variables=sic-osisaf",
                "window=forecast-14-history-3",
                "window.batch_size=1",
                f"model.processor.norm_type={norm}",
            ],
        )
    OmegaConf.set_struct(cfg, False)  # noqa: FBT003 - OmegaConf API
    del cfg.data.datasets["samp-floatsouth-argo-25p0km-2020-2024-24h-v3"]
    del cfg.variables.input["float-argo"]
    del cfg.model.encoders["float-argo"]
    cfg.model.processor.start_out_channels = 16
    cfg.random.seed = seed
    cfg.reporting.metrics = []
    cfg.reporting.loggers = {}
    cfg.train.callbacks = {}
    cfg.train.trainer.accelerator = "mps"
    cfg.train.trainer.devices = 1
    return cfg


def build_datasets(service: ModelService) -> dict[str, CombinedDataset]:
    """Build actual CryoCast split windows, omitting unused climatology generation."""
    dm = service.data_module
    return {
        name: CombinedDataset(
            [single.subset(date_ranges=periods) for single in dm.datasets.values()],
            target_group_name=dm.target_group_name,
            target_variables=dm.target_variables,
            n_history_steps=3,
            n_forecast_steps=LEADS,
        )
        for name, periods in (
            ("train", dm.train_periods),
            ("validation", dm.val_periods),
            ("test", dm.test_periods),
        )
    }


def describe_windows(dataset: CombinedDataset) -> list[dict[str, Any]]:
    """Record every observed and forecast date instead of relying on split labels."""
    return [
        {
            "index": i,
            "history": [
                str(d.astype("datetime64[D]")) for d in dataset.get_history_steps(start)
            ],
            "forecast": [
                str(d.astype("datetime64[D]"))
                for d in dataset.get_forecast_steps(start)
            ],
            "origin": str(dataset.get_history_steps(start)[-1].astype("datetime64[D]")),
        }
        for i, start in enumerate(dataset.dates)
    ]


def data_manifest(data_root: Path) -> dict[str, Any]:
    """Audit source calendar, normalization interval, dimensions and metadata hashes."""
    names = (
        "samp-sicsouth-osisaf-25p0km-2020-2024-24h-v1",
        "samp-weathersouth-era5-25p0km-2020-2024-24h-v4",
    )
    records = {}
    for name in names:
        root = data_root / "data/anemoi" / (name + ".zarr")
        attrs = json.loads((root / ".zattrs").read_text())
        require(attrs["field_shape"] == list(SHAPE), "Source grid changed")
        require(
            attrs["statistics_end_date"] < "2022",
            "Normalization includes held-out years",
        )
        paths = [root / ".zattrs", root / ".zgroup"]
        for directory in (
            "dates",
            "minimum",
            "maximum",
            "mean",
            "stdev",
            "latitudes",
            "longitudes",
        ):
            paths.extend(
                p
                for p in (root / directory).rglob("*")
                if p.is_file() and not p.name.startswith("._")
            )
        records[name] = {
            "metadata_sha256": {
                str(p.relative_to(root)): sha(p) for p in sorted(paths) if p.is_file()
            },
            "statistics_start_date": attrs["statistics_start_date"],
            "statistics_end_date": attrs["statistics_end_date"],
            "field_shape": attrs["field_shape"],
            "shape": attrs["shape"],
            "missing_dates": len(attrs["missing_dates"]),
        }
    return records


def batch_at(dataset: CombinedDataset, index: int) -> dict[str, np.ndarray]:
    """Read a single checked history/future window through the production loader."""
    batch = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in dataset[index].items()
        if k != "climatology"
    }
    require(set(batch) == {"era5", "sic-osisaf", "target"}, "Unexpected model inputs")
    require(batch["target"].shape == (LEADS, 1, *SHAPE), "Forecast shape changed")
    require(
        all(np.isfinite(v).all() for v in batch.values()),
        "Nonfinite prepared source data",
    )
    return batch


def move_inputs(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Move only observed input groups to MPS; future truth remains loss-only."""
    return {
        k: torch.from_numpy(v).unsqueeze(0).to("mps")
        for k, v in batch.items()
        if k != "target"
    }


def state_snapshot(model: EncodeProcessDecode) -> dict[str, torch.Tensor]:
    """Detach all weights and buffers onto CPU for safe checkpoint storage."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash state names, shapes, dtypes and numeric content."""
    digest = hashlib.sha256()
    for name, value in state.items():
        data = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(data.shape).encode())
        digest.update(str(data.dtype).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def validation_mse(
    model: EncodeProcessDecode,
    dataset: CombinedDataset,
    indices: list[int],
    ocean: np.ndarray,
) -> float:
    """Select using the same eight balanced validation windows, never test scores."""
    model.eval()
    square_sum, count = 0.0, 0
    with torch.inference_mode():
        for index in indices:
            batch = batch_at(dataset, index)
            prediction = model(move_inputs(batch))[0].cpu().numpy()
            truth = batch["target"]
            valid = np.broadcast_to(ocean, truth.shape) & np.isfinite(truth)
            for record in per_lead_metrics(prediction, truth, valid):
                square_sum += record["squared_sum"]
                count += record["count"]
    require(count > 0, "Empty validation score")
    return square_sum / count


def train_one(  # noqa: PLR0913, PLR0915 - explicit protocol execution
    *,
    data_root: Path,
    previous: Path,
    checkpoint: dict[str, Any],
    datasets: dict[str, CombinedDataset],
    validation_indices: list[int],
    ocean: np.ndarray,
    output: Path,
    steps: int,
) -> dict[str, Any]:
    """Restore one recorded arm and run a fixed matched continuation budget."""
    seed, norm = checkpoint["seed"], checkpoint["norm"]
    path = previous / checkpoint["checkpoint"]
    require(
        sha(path) == checkpoint["checkpoint_sha256"], "Historical checkpoint changed"
    )
    require(
        not torch.serialization.get_unsafe_globals_in_checkpoint(path),
        "Unexpected checkpoint globals",
    )
    original = torch.load(path, weights_only=True, map_location="cpu")
    require(original["global_step"] == INITIAL_STEPS, "Incorrect starting update count")
    cfg = config_for(data_root, seed, norm)
    service = ModelService.from_config(cfg)
    model = service.model
    if not isinstance(model, EncodeProcessDecode):
        msg = "Expected ordinary encode/process/decode"
        raise TypeError(msg)
    parameters = sum(p.numel() for p in model.parameters())
    require(
        parameters == EXPECTED_PARAMETERS, "Model capacity differs from the prior study"
    )
    model.load_state_dict(original["state_dict"], strict=True)
    initial_state = state_snapshot(model)
    model.to("mps")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=RESTART_LR,
        weight_decay=0.01,
    )
    optimizer.load_state_dict(original["optimizer_states"][0])
    inherited_lr = float(optimizer.param_groups[0]["lr"])
    require(
        inherited_lr == HISTORICAL_TERMINAL_LR,
        "Unexpected historical scheduler endpoint",
    )
    require(
        all(
            int(state["step"].item()) == INITIAL_STEPS
            for state in optimizer.state.values()
        ),
        "Adam steps disagree with checkpoint",
    )
    # Preserve restored moments; override only the preregistered LR schedule.
    order = training_order(seed, len(datasets["train"]), steps)
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    input_digest = hashlib.sha256()
    best_state, best_step = initial_state, INITIAL_STEPS
    best_mse = validation_mse(model, datasets["validation"], validation_indices, ocean)
    history = [
        {"step": INITIAL_STEPS, "validation_mse": best_mse, "training_huber_mean": None}
    ]
    losses: list[float] = []
    updates: list[dict[str, Any]] = []
    started = time.perf_counter()
    log.info("START seed=%s norm=%s val_mse=%.7f", seed, norm, best_mse)
    for index, dataset_index in enumerate(order):
        model.train()
        batch = batch_at(datasets["train"], int(dataset_index))
        for key in sorted(batch):
            input_digest.update(key.encode())
            input_digest.update(batch[key].tobytes())
        learning_rate = continuation_lr(index, steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        prediction = model(move_inputs(batch))
        target = torch.from_numpy(batch["target"]).unsqueeze(0).to("mps")
        loss = model.loss(prediction, target)
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        loss.backward()
        norm_value = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        total_step = INITIAL_STEPS + index + 1
        updates.append(
            {
                "step": total_step,
                "dataset_index": int(dataset_index),
                "huber": loss_value,
                "lr": learning_rate,
                "gradient_norm": float(norm_value.detach().cpu()),
            }
        )
        if (index + 1) % 24 == 0:
            log.info(
                "PROGRESS seed=%s norm=%s total_step=%s loss=%.7f",
                seed,
                norm,
                total_step,
                loss_value,
            )
        if (index + 1) % VALIDATE_EVERY == 0 or index + 1 == steps:
            score = validation_mse(
                model, datasets["validation"], validation_indices, ocean
            )
            history.append(
                {
                    "step": total_step,
                    "validation_mse": score,
                    "training_huber_mean": float(np.mean(losses)),
                }
            )
            losses.clear()
            if score < best_mse:
                best_mse, best_step = score, total_step
                best_state = state_snapshot(model)
            log.info(
                "VALIDATION seed=%s norm=%s step=%s mse=%.7f",
                seed,
                norm,
                total_step,
                score,
            )
    require(
        all(
            int(state["step"].item()) == INITIAL_STEPS + steps
            for state in optimizer.state.values()
        ),
        "Incorrect continued Adam step count",
    )
    last_state = state_snapshot(model)
    checkpoint_hashes = {}
    for selection, state in (
        ("initial", initial_state),
        ("selected", best_state),
        ("final", last_state),
    ):
        destination = output / f"{norm}-seed{seed}-{selection}.pt"
        torch.save(state, destination)
        checkpoint_hashes[selection] = {
            "file": destination.name,
            "sha256": sha(destination),
            "state_sha256": state_hash(state),
        }
    require(
        sha(path) == checkpoint["checkpoint_sha256"], "Source checkpoint was modified"
    )
    result = {
        "seed": seed,
        "norm": norm,
        "parameters": parameters,
        "initial_steps": INITIAL_STEPS,
        "additional_steps": steps,
        "total_steps": INITIAL_STEPS + steps,
        "inherited_lr": inherited_lr,
        "restart_lr": RESTART_LR,
        "final_lr": FINAL_LR,
        "adam_moments_restored": True,
        "selected_step": best_step,
        "selected_validation_mse": best_mse,
        "validation_history": history,
        "updates": updates,
        "training_order_sha256": order_hash,
        "training_values_sha256": input_digest.hexdigest(),
        "original_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoints": checkpoint_hashes,
        "seconds": time.perf_counter() - started,
    }
    save_json(output / f"train-{norm}-seed{seed}.json", result)
    del optimizer, model, service, original, initial_state, best_state, last_state
    gc.collect()
    torch.mps.empty_cache()
    log.info("TRAIN_DONE seed=%s norm=%s selected_step=%s", seed, norm, best_step)
    return result


def evaluate_test(
    records: list[dict[str, Any]],
    data_root: Path,
    test: CombinedDataset,
    ocean: np.ndarray,
    output: Path,
    test_indices: list[int],
) -> None:
    """Score frozen initial, selected and final checkpoints on all fixed test windows."""
    test_dir = output / "test"
    test_dir.mkdir()
    shape = (len(test_indices), LEADS, 1, *SHAPE)
    truth_array = np.lib.format.open_memmap(
        test_dir / "truth.npy", mode="w+", dtype=np.float32, shape=shape
    )
    persistence_array = np.lib.format.open_memmap(
        test_dir / "persistence.npy", mode="w+", dtype=np.float32, shape=shape
    )
    valid_array = np.lib.format.open_memmap(
        test_dir / "valid.npy", mode="w+", dtype=np.bool_, shape=shape
    )
    windows = describe_windows(test)
    for row, index in enumerate(test_indices):
        batch = batch_at(test, index)
        truth_array[row] = batch["target"]
        persistence_array[row] = np.broadcast_to(
            batch["sic-osisaf"][-1:, :1], (LEADS, 1, *SHAPE)
        )
        valid_array[row] = (
            np.broadcast_to(ocean, batch["target"].shape)
            & np.isfinite(batch["target"])
            & np.isfinite(persistence_array[row])
        )
    for array in (truth_array, persistence_array, valid_array):
        array.flush()
    case_rows = []
    for row, index in enumerate(test_indices):
        for metric in per_lead_metrics(
            persistence_array[row], truth_array[row], valid_array[row]
        ):
            case_rows.append(  # noqa: PERF401 - explicit evidence records
                {
                    "seed": None,
                    "norm": "persistence",
                    "selection": "reference",
                    "origin": windows[index]["origin"],
                    "month": windows[index]["origin"][:7],
                    **metric,
                }
            )
    manifests = []
    for record in records:
        seed, norm = record["seed"], record["norm"]
        service = ModelService.from_config(config_for(data_root, seed, norm))
        model = service.model.to("mps").eval()
        for selection in ("initial", "selected", "final"):
            artifact = record["checkpoints"][selection]
            path = output / artifact["file"]
            require(sha(path) == artifact["sha256"], "Frozen checkpoint was changed")
            state = torch.load(path, map_location="cpu", weights_only=True)
            require(
                state_hash(state) == artifact["state_sha256"],
                "Checkpoint content hash differs",
            )
            model.load_state_dict(state, strict=True)
            forecast_path = test_dir / f"{norm}-seed{seed}-{selection}.npy"
            predictions = np.lib.format.open_memmap(
                forecast_path, mode="w+", dtype=np.float32, shape=shape
            )
            with torch.inference_mode():
                for row, index in enumerate(test_indices):
                    batch = batch_at(test, index)
                    require(
                        np.array_equal(batch["target"], truth_array[row]),
                        "Truth changed between arms",
                    )
                    prediction = model(move_inputs(batch))[0].cpu().numpy()
                    predictions[row] = prediction
                    for metric in per_lead_metrics(
                        prediction, truth_array[row], valid_array[row]
                    ):
                        case_rows.append(  # noqa: PERF401 - explicit evidence records
                            {
                                "seed": seed,
                                "norm": norm,
                                "selection": selection,
                                "origin": windows[index]["origin"],
                                "month": windows[index]["origin"][:7],
                                **metric,
                            }
                        )
            predictions.flush()
            del predictions
            manifests.append(
                {
                    "seed": seed,
                    "norm": norm,
                    "selection": selection,
                    "file": forecast_path.name,
                    "sha256": sha(forecast_path),
                }
            )
            log.info("TEST_FROZEN seed=%s norm=%s selection=%s", seed, norm, selection)
        del model, service
        gc.collect()
        torch.mps.empty_cache()
    save_json(output / "per_case.json", case_rows)
    save_json(output / "forecast_manifest.json", manifests)
    save_json(
        output / "test_support.json",
        {
            "test_windows": [windows[i] for i in test_indices],
            "shape": shape,
            "truth_sha256": sha(test_dir / "truth.npy"),
            "persistence_sha256": sha(test_dir / "persistence.npy"),
            "valid_sha256": sha(test_dir / "valid.npy"),
        },
    )


def main() -> None:  # noqa: PLR0915 - one frozen experiment with explicit audit gates
    """Freeze design, train all six runs, then evaluate common held-out months once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()
    require(not output.exists(), "Use a fresh output folder")
    require(
        torch.backends.mps.is_available(), "This frozen protocol requires Apple MPS"
    )
    require(
        os.environ.get("ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL") == "1",
        "Legacy stores require explicit read-only indexing compatibility",
    )
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    previous = args.previous.resolve()
    manifest = json.loads((previous / "training-checkpoint-manifest.json").read_text())
    expected = {(s, n) for s in SEEDS for n in NORMS}
    require(
        {(r["seed"], r["norm"]) for r in manifest} == expected,
        "Historical checkpoint set differs",
    )
    if args.smoke:
        manifest = manifest[:1]
    for record in manifest:
        require(
            sha(previous / record["checkpoint"]) == record["checkpoint_sha256"],
            "Invalid historical checkpoint provenance",
        )
    steps = 2 if args.smoke else EXTRA_STEPS
    service = ModelService.from_config(config_for(args.data_root, SEEDS[0], NORMS[0]))
    datasets = build_datasets(service)
    windows = {k: describe_windows(v) for k, v in datasets.items()}
    require(
        {k: len(v) for k, v in windows.items()}
        == {"train": 715, "validation": 30, "test": 30},
        "Prepared split windows changed",
    )
    date_sets = {
        k: {d for w in v for d in w["history"] + w["forecast"]}
        for k, v in windows.items()
    }
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        require(
            not date_sets[first] & date_sets[second], "Calendar overlap across splits"
        )
    validation_indices = [
        offset + position for offset in (0, 15) for position in VALIDATION_POSITIONS
    ]
    test_indices = list(range(30))
    if args.smoke:
        validation_indices, test_indices = [0], [0]
    ocean_path = service.data_module.mask_directory / "land_mask.npy"
    ocean = np.load(ocean_path).astype(bool)
    require(ocean.shape == SHAPE and ocean.any(), "Incorrect ocean mask")
    source_manifest = data_manifest(args.data_root)
    git = shutil.which("git")
    require(git is not None, "Git is required for exact provenance")
    revision = subprocess.check_output(  # noqa: S603 - fixed Git arguments
        [str(git), "rev-parse", "HEAD"], text=True, cwd=REPOSITORY
    ).strip()
    protocol = {
        "source_revision": revision,
        "runner_sha256": sha(Path(__file__)),
        "metrics_sha256": sha(Path(__file__).with_name("metrics.py")),
        "frozen_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "seeds": list(SEEDS),
        "norms": list(NORMS),
        "initial_updates": INITIAL_STEPS,
        "additional_updates": steps,
        "total_updates": INITIAL_STEPS + steps,
        "budget": "same full-grid reduced-capacity 1.26M model; not full-size production",
        "grid": SHAPE,
        "latent_grid": [144, 144],
        "history_days": 3,
        "forecast_days": LEADS,
        "batch_size": 1,
        "loss": "unchanged full-grid Huber delta=0.5",
        "optimizer": "restore AdamW moments; weight_decay=0.01; clip_norm=1",
        "restart_lr": RESTART_LR,
        "final_lr": FINAL_LR,
        "scheduler": "predeclared per-update cosine restart; old terminal LR is 1e-6",
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "selection": "minimum equal-support validation MSE at 96/144/192/240/288; includes original state",
        "primary_comparison": "fixed final 288-update states; validation-selected states secondary",
        "test_usage": "January previously explored; July added now; all final selections frozen before test",
        "uncertainty": "three paired training seeds; overlapping windows are not independent replicates",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "anemoi": anemoi.datasets.__version__,
        "device": "mps",
        "determinism": "warn-only; MPS index_put accumulation is not guaranteed bitwise reproducible",
        "checkpoints": manifest,
        "source_data": source_manifest,
        "ocean_mask_sha256": sha(ocean_path),
    }
    save_json(output / "protocol.json", protocol)
    save_json(output / "windows.json", windows)
    # The protocol and selected windows exist before any continuation update.
    log.info("PROTOCOL_FROZEN %s", sha(output / "protocol.json"))
    records = []
    for checkpoint in manifest:
        records.append(  # noqa: PERF401 - sequential stateful training
            train_one(
                data_root=args.data_root,
                previous=previous,
                checkpoint=checkpoint,
                datasets=datasets,
                validation_indices=validation_indices,
                ocean=ocean,
                output=output,
                steps=steps,
            )
        )
    if not args.smoke:
        for seed in SEEDS:
            pair = [r for r in records if r["seed"] == seed]
            for key in (
                "training_order_sha256",
                "training_values_sha256",
                "parameters",
                "total_steps",
            ):
                require(
                    pair[0][key] == pair[1][key], "Unmatched paired condition: " + key
                )
    save_json(
        output / "frozen_selections.json",
        {"frozen_utc": datetime.now(UTC).isoformat(), "records": records},
    )
    evaluate_test(
        records, args.data_root, datasets["test"], ocean, output, test_indices
    )
    require(
        data_manifest(args.data_root) == source_manifest,
        "Source metadata/date indices changed",
    )
    require(sha(ocean_path) == protocol["ocean_mask_sha256"], "Original mask changed")
    for checkpoint in manifest:
        require(
            sha(previous / checkpoint["checkpoint"]) == checkpoint["checkpoint_sha256"],
            "Original checkpoint changed",
        )
    save_json(
        output / "complete.json",
        {
            "runs": len(records),
            "added_optimizer_updates": len(records) * steps,
            "protocol_sha256": sha(output / "protocol.json"),
            "frozen_selections_sha256": sha(output / "frozen_selections.json"),
            "per_case_sha256": sha(output / "per_case.json"),
            "original_checkpoint_and_metadata_unchanged": True,
            "test_evaluated_after_all_selections_frozen": True,
        },
    )
    log.info(
        "EXPERIMENT_COMPLETE runs=%s added_updates=%s",
        len(records),
        len(records) * steps,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
