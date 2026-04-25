from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from dataloader.dataset import MusicVAEDataset, music_collate_fn  # noqa: E402
from dataloader.splits import create_train_val_split, load_label_frame  # noqa: E402
from training_scripts.config_utils import (  # noqa: E402
    load_experiment_config,
    project_path,
)
from training_scripts.losses import build_loss_config, compute_vae_loss  # noqa: E402
from training_scripts.model import build_model  # noqa: E402
from training_scripts.normalization import compute_mel_normalizer  # noqa: E402
from training_scripts.training_utils import prepare_batch, select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-batch VAE pipeline sanity check.")
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment_config)
    data_config = config["data"]
    vae_config = config["vae"]
    training_config = vae_config["training"]

    model_input_path = project_path(data_config["model_input_path"])
    mel_tensor_dir = project_path(data_config["mel_tensor_dir"])
    label_path = project_path(data_config["label_path"])

    dataset = MusicVAEDataset(model_input_path=model_input_path, mel_tensor_dir=mel_tensor_dir)
    labels = load_label_frame(label_path, row_count=dataset.full_length)
    split = create_train_val_split(
        labels,
        train_ratio=float(data_config.get("train_ratio", 0.85)),
        val_ratio=float(data_config.get("val_ratio", 0.15)),
        seed=int(data_config.get("seed", 751)),
        stratify_by=str(data_config.get("stratify_by", "genre_language")),
    )
    train_dataset = MusicVAEDataset(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        indices=split["train_indices"][: max(args.batch_size * 2, 4)],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=music_collate_fn,
    )

    normalizer = compute_mel_normalizer(loader, max_batches=1)
    device = select_device(str(training_config.get("device", "auto")))
    model = build_model(vae_config["model"]).to(device)
    loss_config = build_loss_config(vae_config["loss"])

    batch = next(iter(loader))
    prepared = prepare_batch(batch, device, normalizer)
    outputs = model(
        prepared["mel_norm"],
        prepared["lyrics"],
        prepared["metadata"],
        prepared["condition"],
    )
    loss, metrics = compute_vae_loss(outputs, prepared, beta=0.0, config=loss_config)
    loss.backward()

    expected_shapes = {
        "mel_recon_norm": (args.batch_size, 1, 128, 1292),
        "lyrics_recon": (args.batch_size, 1024),
        "metadata_recon": (args.batch_size, 40),
        "mu": (args.batch_size, 64),
        "logvar": (args.batch_size, 64),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(outputs[key].shape)
        if actual != expected:
            raise AssertionError(f"{key} shape mismatch: expected {expected}, got {actual}")

    if not math.isfinite(float(loss.detach().cpu())):
        raise AssertionError("Loss is not finite.")
    for key, value in metrics.items():
        if not math.isfinite(value):
            raise AssertionError(f"Metric {key} is not finite: {value}")

    print("VAE sanity check passed.")
    for key, expected in expected_shapes.items():
        print(f"{key}: {expected}")
    print(f"loss: {float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
