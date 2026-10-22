"""Check whether the retained status masks add information from held-out years."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import zarr

from cryocast.utils import mask_dir

DATASET = "samp-sicsouth-osisaf-25p0km-2020-2024-24h-v1"
TRAINING_END = "2021-12-31"
LAND_BIT = 1
INACTIVE_BIT = 128


def file_hash(path: Path) -> str:
    """Hash the small stored masks before and after read-only verification."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data_root: Path) -> dict[str, object]:
    """Compare all-period and training-only masks using actual stored status flags."""
    path = data_root / "data/anemoi" / f"{DATASET}.zarr"
    attrs = json.loads((path / ".zattrs").read_text())
    source = zarr.open(str(path), mode="r")
    dates = np.asarray(source["dates"][:]).astype("datetime64[D]")
    variable_index = attrs["variables"].index("status_flag")
    masks_root = mask_dir(data_root, DATASET)
    stored = {
        name: np.load(masks_root / f"{name}_mask.npy").astype(bool)
        for name in ("land", "active")
    }
    hashes = {name: file_hash(masks_root / f"{name}_mask.npy") for name in stored}
    shape = stored["land"].shape
    all_land, all_active, train_land, train_active = (
        np.zeros(shape, dtype=bool) for _ in range(4)
    )
    training_rows = 0
    for index, date in enumerate(dates):
        flags = np.asarray(
            source["data"][index, variable_index, 0], dtype=np.uint8
        ).reshape(shape)
        land, active = (flags & LAND_BIT) != 0, (flags & INACTIVE_BIT) == 0
        all_land |= land
        all_active |= active
        if date <= np.datetime64(TRAINING_END):
            train_land |= land
            train_active |= active
            training_rows += 1
    all_masks = {"land": ~all_land, "active": all_active & ~all_land}
    training_masks = {"land": ~train_land, "active": train_active & ~train_land}
    results = {}
    for name, mask in stored.items():
        results[name] = {
            "stored_matches_all_period_reconstruction": bool(
                np.array_equal(mask, all_masks[name])
            ),
            "stored_matches_training_only": bool(
                np.array_equal(mask, training_masks[name])
            ),
            "different_cells_from_training_only": int(
                np.count_nonzero(mask != training_masks[name])
            ),
            "stored_valid_cells": int(mask.sum()),
            "training_only_valid_cells": int(training_masks[name].sum()),
        }
        if file_hash(masks_root / f"{name}_mask.npy") != hashes[name]:
            msg = "Original mask changed during its read-only audit."
            raise ValueError(msg)
    return {
        "dataset": DATASET,
        "status_variable_index": variable_index,
        "total_rows": len(dates),
        "training_rows": training_rows,
        "training_end": TRAINING_END,
        "stored_hashes": hashes,
        "source_policy": "status masks generated across all available source dates",
        "mask_results": results,
        "scope": "Snapshot equivalence only; this does not prove all-period masking safe for every future dataset.",
    }


def main() -> None:
    """Emit a reproducible mask-provenance receipt without rewriting source data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.out.write_text(rendered)
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
