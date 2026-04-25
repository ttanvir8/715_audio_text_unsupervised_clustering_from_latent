from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_latent_bundle(
    output_dir: str | Path,
    mu: np.ndarray,
    logvar: np.ndarray,
    row_indices: list[int],
    labels,
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    output_path = Path(output_dir)
    expected_files = [
        output_path / "mu.npy",
        output_path / "logvar.npy",
        output_path / "row_indices.npy",
        output_path / "labels.parquet",
        output_path / "latent_export_metadata.json",
    ]
    existing = [path for path in expected_files if path.exists()]
    if existing and not overwrite:
        first = existing[0]
        raise FileExistsError(f"Refusing to overwrite existing latent export: {first}")

    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "mu.npy", mu)
    np.save(output_path / "logvar.npy", logvar)
    np.save(output_path / "row_indices.npy", np.asarray(row_indices, dtype=np.int64))
    pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), output_path / "labels.parquet")
    save_json(metadata, output_path / "latent_export_metadata.json")
