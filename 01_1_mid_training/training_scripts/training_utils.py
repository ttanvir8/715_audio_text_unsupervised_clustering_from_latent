from __future__ import annotations

import csv
from itertools import islice
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. "
            "Run on a CUDA-enabled machine, or set training.device: cpu for a debug run."
        )
    return torch.device(device_name)


def prepare_batch(batch: dict[str, Any], device: torch.device, normalizer) -> dict[str, Any]:
    mel = batch["melspectrogram"].to(device, non_blocking=True)
    lyrics = batch["lyrics_input"].to(device, non_blocking=True)
    mel_norm = normalizer.normalize(mel).to(device)
    return {
        "mel_norm": mel_norm,
        "lyrics": lyrics,
        "row_index": batch["row_index"],
    }


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def append_metrics_csv(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_metrics_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@torch.no_grad()
def export_latents(
    model,
    loader,
    normalizer,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    model.eval()
    mu_values = []
    row_indices: list[int] = []
    iterable = islice(loader, max_batches) if max_batches is not None else loader
    for batch in iterable:
        prepared = prepare_batch(batch, device, normalizer)
        mu, _ = model.encode(prepared["mel_norm"], prepared["lyrics"])
        mu_values.append(mu.detach().cpu().numpy())
        row_indices.extend(int(index) for index in batch["row_index"].tolist())

    if not mu_values:
        raise ValueError("No latent values were exported.")

    return np.concatenate(mu_values, axis=0), row_indices

