from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "01_1_mid_training"
HELPER_ROOT = PROJECT_ROOT / "01_2_inference" / "extra_helper_functions"
for path in (TRAINING_ROOT, HELPER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataloader.dataset import AudioLyricsVAEDataset, audio_lyrics_collate_fn  # noqa: E402
from dataloader.splits import load_label_frame, load_split  # noqa: E402
from inference_io import load_yaml, save_latent_bundle  # noqa: E402
from training_scripts.config_utils import project_path  # noqa: E402
from training_scripts.model import build_model  # noqa: E402
from training_scripts.normalization import MelNormalizer  # noqa: E402
from training_scripts.training_utils import prepare_batch, select_device  # noqa: E402


def log(message: str) -> None:
    print(f"[mid_export_latents] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export encoder latents from a trained medium audio+lyrics VAE checkpoint."
    )
    parser.add_argument(
        "--experiment-dir",
        default="01_1_mid_training/experiments/audio_lyrics_vae_run_001",
        help="Training experiment directory containing best_checkpoint/model.pt.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which split to export. full means the selected medium train+val rows.",
    )
    parser.add_argument("--device", default="auto", help="Device for encoder inference.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
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
        default="01_2_inference/latents_by_best_checkpoints",
        help="Root directory where experiment latent folders are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing latent export for the selected scope.",
    )
    return parser.parse_args()


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
    dataset: AudioLyricsVAEDataset,
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
        collate_fn=audio_lyrics_collate_fn,
    )


@torch.no_grad()
def export_scope_latents(
    model,
    loader: DataLoader,
    normalizer: MelNormalizer,
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
        prepared = prepare_batch(batch, device, normalizer)
        mu, logvar = model.encode(prepared["mel_norm"], prepared["lyrics"])
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


def main() -> None:
    args = parse_args()
    experiment_dir = project_path(args.experiment_dir)
    resolved_config_path = experiment_dir / "resolved_config.yaml"
    checkpoint_path = experiment_dir / "best_checkpoint" / "model.pt"
    normalizer_path = experiment_dir / "mel_normalization.json"
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

    model_input_path = project_path(data_config["model_input_path"])
    mel_tensor_dir = project_path(data_config["mel_tensor_dir"])
    label_path = project_path(data_config["label_path"])
    mel_cache_chunks = int(data_config.get("mel_cache_chunks", 2))
    input_embedding_column = str(data_config.get("input_embedding_column", "lyrics_e5_large_embedding"))
    language_column = str(data_config.get("language_column", "music_lang"))

    log(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = MelNormalizer.from_file(normalizer_path)
    split = load_split(split_path)

    all_dataset = AudioLyricsVAEDataset(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        input_embedding_column=input_embedding_column,
        mel_cache_chunks=mel_cache_chunks,
    )
    labels = load_label_frame(
        label_path,
        language_column=language_column,
        row_count=all_dataset.full_length,
        model_input_path=model_input_path,
    )
    all_dataset.close()

    output_root = project_path(args.output_root) / experiment_dir.name
    for scope in scopes_from_arg(args.scope):
        indices = scope_indices(scope, split)
        dataset = AudioLyricsVAEDataset(
            model_input_path=model_input_path,
            mel_tensor_dir=mel_tensor_dir,
            input_embedding_column=input_embedding_column,
            mel_cache_chunks=mel_cache_chunks,
            indices=indices,
        )
        loader = build_loader(dataset, args.batch_size, num_workers, pin_memory)
        log(f"Exporting {scope}: rows={len(dataset)}, batches={len(loader)}")
        mu, logvar, row_indices = export_scope_latents(
            model=model,
            loader=loader,
            normalizer=normalizer,
            device=device,
            max_batches=args.max_batches,
        )
        scope_labels = labels_for_rows(labels, row_indices)
        output_dir = output_root / scope
        metadata = {
            "experiment_dir": str(experiment_dir),
            "checkpoint_path": str(checkpoint_path),
            "normalizer_path": str(normalizer_path),
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
            "mel_normalization": normalizer.to_dict(),
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
        dataset.close()

    log(f"Latent export complete: {output_root}")


if __name__ == "__main__":
    main()
