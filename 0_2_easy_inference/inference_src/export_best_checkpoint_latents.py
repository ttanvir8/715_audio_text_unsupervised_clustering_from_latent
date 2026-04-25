from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "0_1_easy_training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from dataloader.dataset import LyricsVAEDataset, lyrics_collate_fn  # noqa: E402
from dataloader.splits import load_label_frame, load_split  # noqa: E402
from training_scripts.config_utils import project_path  # noqa: E402
from training_scripts.model import build_model  # noqa: E402
from training_scripts.training_utils import prepare_batch, select_device  # noqa: E402


def log(message: str) -> None:
    print(f"[easy_export_latents] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export encoder latents from a trained easy lyrics VAE checkpoint."
    )
    parser.add_argument(
        "--experiment-dir",
        default="0_1_easy_training/experiments/lyrics_vae_run_001",
        help="Training experiment directory containing best_checkpoint/model.pt.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which split to export. full means the selected easy train+val rows.",
    )
    parser.add_argument("--device", default="auto", help="Device for encoder inference.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader workers; defaults to the training data config.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per scope for smoke tests.",
    )
    parser.add_argument(
        "--output-root",
        default="0_2_easy_inference/latents_by_best_checkpoints",
        help="Root directory where experiment latent folders are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing latent export for the selected scope.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def scope_indices(scope: str, split: dict[str, Any]) -> list[int]:
    if scope == "validation":
        return [int(index) for index in split["val_indices"]]
    if scope == "full":
        return [int(index) for index in split["selected_indices"]]
    raise ValueError(f"Unsupported scope: {scope}")


def labels_for_rows(labels, row_indices: list[int]):
    return labels.set_index("row_index").loc[row_indices].reset_index()


def build_loader(
    dataset: LyricsVAEDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lyrics_collate_fn,
    )


@torch.no_grad()
def export_scope_latents(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    model.eval()
    mu_values = []
    logvar_values = []
    row_indices: list[int] = []
    iterable = islice(loader, max_batches) if max_batches is not None else loader
    total = min(len(loader), max_batches) if max_batches is not None else len(loader)

    for batch in tqdm(iterable, desc="latent export", total=total, leave=False):
        prepared = prepare_batch(batch, device)
        mu, logvar = model.encode(prepared["lyrics"])
        mu_values.append(mu.detach().cpu().numpy())
        logvar_values.append(logvar.detach().cpu().numpy())
        row_indices.extend(int(index) for index in batch["row_index"].tolist())

    if not mu_values:
        raise ValueError("No latents were exported. Check dataset scope and max-batches.")

    return (
        np.concatenate(mu_values, axis=0),
        np.concatenate(logvar_values, axis=0),
        row_indices,
    )


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
        raise FileExistsError(f"Refusing to overwrite existing latent export: {existing[0]}")

    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "mu.npy", mu)
    np.save(output_path / "logvar.npy", logvar)
    np.save(output_path / "row_indices.npy", np.asarray(row_indices, dtype=np.int64))
    pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), output_path / "labels.parquet")
    save_json(metadata, output_path / "latent_export_metadata.json")


def main() -> None:
    args = parse_args()
    experiment_dir = project_path(args.experiment_dir)
    resolved_config_path = experiment_dir / "resolved_config.yaml"
    checkpoint_path = experiment_dir / "best_checkpoint" / "model.pt"
    split_path = experiment_dir / "split_indices.json"

    log(f"Experiment directory: {experiment_dir}")
    log(f"Loading resolved config: {resolved_config_path}")
    config = load_yaml(resolved_config_path)
    data_config = config["data"]
    model_config = config["vae"]["model"]
    num_workers = (
        int(data_config.get("num_workers", 0))
        if args.num_workers is None
        else int(args.num_workers)
    )
    device = select_device(args.device)
    pin_memory = device.type == "cuda"
    log(
        "Inference settings: "
        f"scope={args.scope}, batch_size={args.batch_size}, "
        f"num_workers={num_workers}, device={device}"
    )

    lyrics_path = project_path(data_config["lyrics_path"])
    input_embedding_column = str(data_config.get("input_embedding_column", "lyrics_e5_large_embedding"))
    language_column = str(data_config.get("language_column", "music_lang"))

    log(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split = load_split(split_path)
    labels = load_label_frame(lyrics_path, language_column=language_column)
    output_root = project_path(args.output_root) / experiment_dir.name

    for scope in scopes_from_arg(args.scope):
        indices = scope_indices(scope, split)
        dataset = LyricsVAEDataset(
            lyrics_path=lyrics_path,
            input_embedding_column=input_embedding_column,
            indices=indices,
        )
        loader = build_loader(dataset, args.batch_size, num_workers, pin_memory)
        log(f"Exporting {scope}: rows={len(dataset)}, batches={len(loader)}")
        mu, logvar, row_indices = export_scope_latents(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )
        scope_labels = labels_for_rows(labels, row_indices)
        output_dir = output_root / scope
        metadata = {
            "experiment_dir": str(experiment_dir),
            "checkpoint_path": str(checkpoint_path),
            "scope": scope,
            "requested_rows": len(dataset),
            "exported_rows": int(mu.shape[0]),
            "max_batches": args.max_batches,
            "mu_shape": list(mu.shape),
            "logvar_shape": list(logvar.shape),
            "row_indices_shape": [len(row_indices)],
            "input_embedding_column": input_embedding_column,
            "language_column": language_column,
            "split_strategy": split.get("split_strategy"),
            "model_config": model_config,
        }
        save_latent_bundle(
            output_dir=output_dir,
            mu=mu,
            logvar=logvar,
            row_indices=row_indices,
            labels=scope_labels,
            metadata=metadata,
            overwrite=args.overwrite,
        )
        log(f"Saved {scope}: output_dir={output_dir}, mu={mu.shape}, logvar={logvar.shape}")

    log(f"Latent export complete: {output_root}")


if __name__ == "__main__":
    main()

