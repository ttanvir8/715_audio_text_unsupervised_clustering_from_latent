from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from dataloader.dataset import LyricsVAEDataset, lyrics_collate_fn  # noqa: E402
from dataloader.splits import create_language_ratio_split, load_label_frame  # noqa: E402
from training_scripts.config_utils import load_experiment_config, project_path  # noqa: E402
from training_scripts.losses import beta_for_step, build_loss_config, compute_vae_loss  # noqa: E402
from training_scripts.model import build_model  # noqa: E402
from training_scripts.training_utils import prepare_batch, select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-batch lyrics VAE sanity check.")
    parser.add_argument(
        "--experiment-config",
        default="0_1_easy_training/configs/experiments/run_001.yaml",
        help="Path to the experiment manifest YAML.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device from the VAE config, for example cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment_config)
    data_config = config["data"]
    vae_config = config["vae"]
    model_config = dict(vae_config["model"])
    loss_config = build_loss_config(vae_config["loss"])

    seed = int(config["experiment"].get("seed", data_config.get("seed", 751)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    lyrics_path = project_path(data_config["lyrics_path"])
    input_embedding_column = str(data_config.get("input_embedding_column", "lyrics_e5_large_embedding"))
    language_column = str(data_config.get("language_column", "music_lang"))
    labels = load_label_frame(lyrics_path, language_column=language_column)
    split = create_language_ratio_split(
        labels,
        train_ratio=float(data_config.get("train_ratio", 0.85)),
        val_ratio=float(data_config.get("val_ratio", 0.15)),
        seed=seed,
        language_column=language_column,
        max_rows_per_language=data_config.get("max_rows_per_language"),
        min_rows_per_language=int(data_config.get("min_rows_per_language", 2)),
    )

    dataset = LyricsVAEDataset(
        lyrics_path=lyrics_path,
        input_embedding_column=input_embedding_column,
        indices=split["train_indices"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lyrics_collate_fn,
    )
    batch = next(iter(loader))
    model_config["input_dim"] = dataset.input_dim

    device_name = args.device or str(vae_config["training"].get("device", "auto"))
    device = select_device(device_name)
    model = build_model(model_config).to(device)
    prepared = prepare_batch(batch, device)
    outputs = model(prepared["lyrics"])
    beta = beta_for_step(0, 1, loss_config)
    loss, metrics = compute_vae_loss(outputs, prepared, beta, loss_config)
    loss.backward()

    print("Lyrics VAE sanity check passed.")
    print(f"lyrics_input: {tuple(prepared['lyrics'].shape)}")
    print(f"lyrics_recon: {tuple(outputs['lyrics_recon'].shape)}")
    print(f"mu: {tuple(outputs['mu'].shape)}")
    print(f"logvar: {tuple(outputs['logvar'].shape)}")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"split languages: {split['per_language']}")


if __name__ == "__main__":
    main()

