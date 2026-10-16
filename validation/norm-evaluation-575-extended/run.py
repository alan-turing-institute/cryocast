"""Reproduce paired BatchNorm vs GroupNorm sea-ice forecast evidence (issue #575).

Source code belongs in the PR; output JSON and large data stay on external storage.
Requires CRYOCAST_NORM_DATA_ROOT, CRYOCAST_NORM_OUTPUT_DIR and the explicit
ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL=1 for existing legacy Zarr.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import anemoi.datasets
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from cryocast.model_service import ModelService
from cryocast.model_service import model_service as model_service_module

# Keep this bounded local benchmark single-process so worker teardown cannot dominate.
model_service_module.suggested_max_num_workers = lambda local_world_size: 0  # noqa: ARG005

CONFIG_DIR = str(
    (Path(__file__).resolve().parents[2] / "cryocast" / "config").resolve()
)
BASE = os.environ.get("CRYOCAST_NORM_DATA_ROOT", "")
OUT = Path(os.environ.get("CRYOCAST_NORM_OUTPUT_DIR", ""))
SEEDS = (575, 576, 577)
NORMS = ("batchnorm", "groupnorm")
TRAIN_BATCHES = 96
TEST_BATCHES = 10
PIXEL_AREA_KM2 = 25.0**2
SIC_THRESHOLD = 0.15
log = logging.getLogger(__name__)
START_CHANNELS = 16


def config_for(seed: int, norm: str) -> DictConfig:
    """Construct the matched training and evaluation configuration."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="baseline/cnn_unet_cnn",
            overrides=[
                f"base_path={BASE}",
                "data=sample_south",
                "variables=sic-osisaf",
                "window=forecast-14-history-3",
                "window.batch_size=1",
                f"model.processor.norm_type={norm}",
            ],
        )
    OmegaConf.set_struct(cfg, False)  # noqa: FBT003 - OmegaConf API
    # Argo excluded consistently from both arms to isolate the normalization change.
    # The legacy OSI SAF/ERA5 stores are read through Anemoi in-memory date fix.
    del cfg.data.datasets["samp-floatsouth-argo-25p0km-2020-2024-24h-v3"]
    del cfg.variables.input["float-argo"]
    del cfg.model.encoders["float-argo"]
    cfg.model.processor.start_out_channels = START_CHANNELS
    cfg.random.seed = seed
    cfg.train.trainer.max_epochs = 1
    cfg.train.trainer.limit_train_batches = TRAIN_BATCHES
    cfg.train.trainer.limit_val_batches = 0
    cfg.train.trainer.num_sanity_val_steps = 0
    cfg.train.trainer.accelerator = "mps"
    cfg.train.trainer.devices = 1
    cfg.train.trainer.enable_model_summary = False
    cfg.train.trainer.log_every_n_steps = 10
    cfg.train.trainer.enable_progress_bar = False
    cfg.train.callbacks = {}
    cfg.reporting.loggers = {}
    return cfg


def eval_model(service: ModelService) -> dict[str, object]:
    """Measure the same 14-day forecast dates for all arms and seeds."""
    model = service.model.to("mps")
    model.eval()
    service.data_module.assign_workers(0)
    loader = service.data_module.test_dataloader()

    land_mask = torch.as_tensor(
        np.load(service.data_module.mask_directory / "land_mask.npy"),
        dtype=torch.bool,
        device="mps",
    )
    sic_index = service.data_module.datasets["sic-osisaf"].variable_names.index(
        "ice_conc"
    )

    model_sq = torch.zeros(14, device="mps", dtype=torch.float32)
    persistence_sq = torch.zeros(14, device="mps", dtype=torch.float32)
    model_sie = torch.zeros(14, device="mps", dtype=torch.float32)
    model_sie_abs = torch.zeros(14, device="mps", dtype=torch.float32)
    persistence_sie = torch.zeros(14, device="mps", dtype=torch.float32)
    persistence_sie_abs = torch.zeros(14, device="mps", dtype=torch.float32)
    count = torch.zeros(14, device="mps", dtype=torch.float32)
    sample_count = 0

    start = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= TEST_BATCHES:
                break
            moved = {
                key: value.to("mps") if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            target = moved["target"]
            inputs = {k: v for k, v in moved.items() if k != "target"}
            pred = model(inputs)
            last_sic = moved["sic-osisaf"][:, -1:, sic_index : sic_index + 1]
            persistence = last_sic.expand(-1, target.shape[1], -1, -1, -1)

            mask = land_mask.view(1, 1, 1, *land_mask.shape)
            n_valid = float(land_mask.sum().item()) * target.shape[0]
            pred_err = torch.where(mask, (pred - target).float().square(), 0.0)
            pers_err = torch.where(mask, (persistence - target).float().square(), 0.0)
            model_sq += pred_err.sum(dim=(0, 2, 3, 4))
            persistence_sq += pers_err.sum(dim=(0, 2, 3, 4))
            count += n_valid

            target_ice = (target > SIC_THRESHOLD) & mask
            pred_ice = (pred > SIC_THRESHOLD) & mask
            pers_ice = (persistence > SIC_THRESHOLD) & mask
            model_sie_batch = (
                pred_ice.sum(dim=(2, 3, 4)) - target_ice.sum(dim=(2, 3, 4))
            ).float() * PIXEL_AREA_KM2
            persistence_sie_batch = (
                pers_ice.sum(dim=(2, 3, 4)) - target_ice.sum(dim=(2, 3, 4))
            ).float() * PIXEL_AREA_KM2
            model_sie += model_sie_batch.sum(dim=0)
            model_sie_abs += model_sie_batch.abs().sum(dim=0)
            persistence_sie += persistence_sie_batch.sum(dim=0)
            persistence_sie_abs += persistence_sie_batch.abs().sum(dim=0)
            sample_count += target.shape[0]

    elapsed = time.perf_counter() - start
    rmse = torch.sqrt(model_sq / count).cpu().tolist()
    persistence_rmse = torch.sqrt(persistence_sq / count).cpu().tolist()
    sie = (model_sie / sample_count).cpu().tolist()
    sie_abs = (model_sie_abs / sample_count).cpu().tolist()
    persistence_sie_mean = (persistence_sie / sample_count).cpu().tolist()
    persistence_sie_abs_mean = (persistence_sie_abs / sample_count).cpu().tolist()
    return {
        "test_samples": sample_count,
        "inference_seconds": elapsed,
        "rmse_per_lead": rmse,
        "persistence_rmse_per_lead": persistence_rmse,
        "sie_error_km2_per_lead": sie,
        "absolute_sie_error_km2_per_lead": sie_abs,
        "persistence_sie_error_km2_per_lead": persistence_sie_mean,
        "persistence_absolute_sie_error_km2_per_lead": persistence_sie_abs_mean,
        "mean_rmse": float(np.mean(rmse)),
        "late_rmse": float(np.mean(rmse[-5:])),
        "persistence_mean_rmse": float(np.mean(persistence_rmse)),
        "persistence_late_rmse": float(np.mean(persistence_rmse[-5:])),
        "mean_abs_sie_error_km2": float(np.mean(sie_abs)),
        "late_abs_sie_error_km2": float(np.mean(sie_abs[-5:])),
        "persistence_mean_abs_sie_error_km2": float(np.mean(persistence_sie_abs_mean)),
        "note_sie_error": "Absolute sea-ice-extent error is averaged per sample, before aggregating leads/seeds.",
    }


def run_one(seed: int, norm: str) -> dict[str, object]:
    """Fit one seeded normalization arm and evaluate its held-out rollouts."""
    cfg = config_for(seed, norm)
    start = time.perf_counter()
    service = ModelService.from_config(cfg)
    params = sum(p.numel() for p in service.model.parameters())
    trainer = service.train()
    train_seconds = time.perf_counter() - start
    result = eval_model(service)
    result.update(
        {
            "seed": seed,
            "norm": norm,
            "parameters": params,
            "train_seconds": train_seconds,
            "optimizer_steps": trainer.global_step,
            "start_out_channels": START_CHANNELS,
            "test_batches_limit": TEST_BATCHES,
            "train_batches_limit": TRAIN_BATCHES,
        }
    )
    if trainer.global_step != TRAIN_BATCHES:
        msg = f"Expected {TRAIN_BATCHES} updates, got {trainer.global_step}"
        raise RuntimeError(msg)
    del trainer, service
    gc.collect()
    torch.mps.empty_cache()
    return result


def main() -> None:
    """Run or resume all paired seeds; write each completed result atomically."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not BASE or not Path(BASE).is_dir():
        msg = "Provide CRYOCAST_NORM_DATA_ROOT pointing to existing datasets."
        raise RuntimeError(msg)
    if "CRYOCAST_NORM_OUTPUT_DIR" not in os.environ:
        msg = "Provide CRYOCAST_NORM_OUTPUT_DIR on external storage."
        raise RuntimeError(msg)
    if os.environ.get("ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL") != "1":
        msg = "Legacy stores require the explicit in-memory Anemoi date fix."
        raise RuntimeError(msg)
    OUT.mkdir(parents=True, exist_ok=True)
    git_bin = shutil.which("git")
    if git_bin is None:
        msg = "Git is required to record the exact source revision."
        raise RuntimeError(msg)
    source_sha = subprocess.check_output(  # noqa: S603 - fixed git arguments
        [git_bin, "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    provenance = {
        "source_sha": source_sha,
        "script_sha256": script_sha256,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "anemoi_version": anemoi.datasets.__version__,
        "platform": platform.platform(),
        "data_root": BASE,
        "legacy_date_fix_in_memory": True,
    }
    runs = []
    for seed in SEEDS:
        for norm in NORMS:
            dest = OUT / f"{norm}-seed{seed}.json"
            if dest.is_file():
                result = json.loads(dest.read_text(encoding="utf-8"))
                if (
                    result.get("optimizer_steps") != TRAIN_BATCHES
                    or result.get("start_out_channels") != START_CHANNELS
                    or result.get("test_batches_limit") != TEST_BATCHES
                    or result.get("source_sha") != source_sha
                    or result.get("script_sha256") != script_sha256
                ):
                    msg = f"Existing result does not match protocol: {dest}"
                    raise ValueError(msg)
                log.info("REUSE verified run seed=%s norm=%s", seed, norm)
            else:
                log.info("RUN seed=%s norm=%s", seed, norm)
                result = run_one(seed, norm)
                result.update(provenance)
                temp = dest.with_suffix(".json.tmp")
                temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
                temp.replace(dest)
                log.info(
                    "DONE seed=%s norm=%s train_seconds=%.1f mean_rmse=%.6f",
                    seed,
                    norm,
                    result["train_seconds"],
                    result["mean_rmse"],
                )
            runs.append(result)
            (OUT / "partial-summary.json").write_text(
                json.dumps(
                    {
                        "protocol": {
                            "steps": TRAIN_BATCHES,
                            "seeds": SEEDS,
                            "channels": START_CHANNELS,
                        },
                        "runs": runs,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    summary = {
        "protocol": {
            "model": "cnn_unet_cnn",
            "history_steps": 3,
            "forecast_steps": 14,
            "batch_size": 1,
            "train_batches": TRAIN_BATCHES,
            "epochs": 1,
            "optimizer_steps_per_run": TRAIN_BATCHES,
            "start_out_channels": START_CHANNELS,
            "parameter_capacity": "Reduced-capacity CNN-UNet-CNN experiment; not the 44.7M production setting",
            "test_batches": TEST_BATCHES,
            "seeds": list(SEEDS),
            "inputs": ["ERA5", "OSI SAF SIC"],
            "excluded_input": "Argo excluded from both arms to isolate the normalization comparison; updated Argo experiments are separate",
            "note": "Extended reduced-capacity training pilot, NOT converged or full-size production-skill evidence. The full-size earlier 20-step experiment is separate.",
        },
        "runs": runs,
        "provenance": provenance,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
