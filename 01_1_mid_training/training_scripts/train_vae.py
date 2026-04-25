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

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from dataloader.dataset import AudioLyricsVAEDataset, audio_lyrics_collate_fn  # noqa: E402
from dataloader.splits import (  # noqa: E402
    create_language_ratio_split,
    export_validation_labels,
    load_label_frame,
    save_split,
)
from training_scripts.config_utils import (  # noqa: E402
    load_experiment_config,
    project_path,
    resolve_output_dir,
    save_yaml,
)
from training_scripts.losses import (  # noqa: E402
    beta_for_step,
    build_loss_config,
    compute_vae_loss,
)
from training_scripts.model import build_model  # noqa: E402
from training_scripts.normalization import compute_mel_normalizer  # noqa: E402
from training_scripts.training_utils import (  # noqa: E402
    append_metrics_csv,
    append_metrics_rows_csv,
    average_metrics,
    export_latents,
    prepare_batch,
    save_json,
    select_device,
)


def log(message: str) -> None:
    print(f"[train_audio_lyrics_vae] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a non-conditional audio+lyrics VAE.")
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to the experiment manifest YAML.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit training batches per epoch for smoke runs.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation/export batches for smoke runs.",
    )
    parser.add_argument(
        "--max-stat-batches",
        type=int,
        default=None,
        help="Limit mel-stat batches; defaults to max-train-batches when set.",
    )
    return parser.parse_args()


def make_loader(
    dataset: AudioLyricsVAEDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=audio_lyrics_collate_fn,
    )


def make_run_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "training_logs": output_dir / "training_logs",
        "graphs": output_dir / "graphs",
        "checkpoints": output_dir / "checkpoints_of_each_epochs",
        "best": output_dir / "best_checkpoint",
        "latents": output_dir / "latent_exports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def metric_row(epoch: int, split: str, metrics: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "split": split}
    row.update(metrics)
    return row


def step_metric_row(
    epoch: int,
    split: str,
    global_step: int,
    batch_in_epoch: int,
    batch_size: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "split": split,
        "global_step": global_step,
        "batch_in_epoch": batch_in_epoch,
        "batch_size": batch_size,
    }
    row.update(metrics)
    return row


def is_metric_improved(value: float, best_value: float, mode: str) -> bool:
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError(f"Unsupported checkpoint_mode: {mode}")


def require_metric(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"Metric '{metric_name}' is not available. Available metrics: {available}")
    return float(metrics[metric_name])


def format_epoch_metrics(metrics: dict[str, float]) -> str:
    return (
        f"loss={metrics.get('loss', float('nan')):.6f}, "
        f"recon={metrics.get('recon_loss', float('nan')):.6f}, "
        f"mel={metrics.get('mel_recon_loss', float('nan')):.6f}, "
        f"lyrics={metrics.get('lyrics_recon_loss', float('nan')):.6f}, "
        f"raw_kl={metrics.get('raw_kl_loss', float('nan')):.6f}, "
        f"freebits_kl={metrics.get('kl_loss', float('nan')):.6f}, "
        f"active_dims={metrics.get('active_kl_dims', float('nan')):.2f}, "
        f"mu_std={metrics.get('mu_std_mean', float('nan')):.6f}, "
        f"beta={metrics.get('beta', float('nan')):.6f}"
    )


def run_epoch(
    model,
    loader,
    normalizer,
    loss_config,
    device: torch.device,
    total_steps: int,
    global_step: int,
    optimizer=None,
    grad_clip_norm: float = 1.0,
    mixed_precision: bool = False,
    max_batches: int | None = None,
    epoch: int = 0,
    split: str = "",
    step_log_every: int = 0,
) -> tuple[dict[str, float], int, list[dict[str, Any]]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    metrics: list[dict[str, float]] = []
    step_metrics: list[dict[str, Any]] = []
    amp_enabled = bool(mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    limited_loader = islice(loader, max_batches) if max_batches is not None else loader
    progress_total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    iterable = tqdm(
        limited_loader,
        desc="train" if train_mode else "val",
        leave=False,
        total=progress_total,
    )
    for batch_in_epoch, batch in enumerate(iterable, start=1):
        prepared = prepare_batch(batch, device, normalizer)
        beta = beta_for_step(global_step, total_steps, loss_config)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(prepared["mel_norm"], prepared["lyrics"])
                loss, batch_metrics = compute_vae_loss(outputs, prepared, beta, loss_config)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

        metrics.append(batch_metrics)
        if step_log_every > 0 and (
            batch_in_epoch == 1
            or batch_in_epoch % step_log_every == 0
            or batch_in_epoch == progress_total
        ):
            step_metrics.append(
                step_metric_row(
                    epoch=epoch,
                    split=split,
                    global_step=global_step,
                    batch_in_epoch=batch_in_epoch,
                    batch_size=int(prepared["lyrics"].shape[0]),
                    metrics=batch_metrics,
                )
            )
        iterable.set_postfix(loss=f"{batch_metrics['loss']:.4f}", beta=f"{beta:.3f}")

    return average_metrics(metrics), global_step, step_metrics


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    config: dict[str, Any],
    normalizer,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "mel_normalization": normalizer.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def main() -> None:
    args = parse_args()
    log(f"Loading experiment config: {args.experiment_config}")
    config = load_experiment_config(args.experiment_config)
    output_dir = resolve_output_dir(config)
    log(f"Preparing output directory: {output_dir}")
    run_paths = make_run_dirs(output_dir)

    data_config = config["data"]
    vae_config = config["vae"]
    model_config = dict(vae_config["model"])
    training_config = dict(vae_config["training"])
    loss_config = build_loss_config(vae_config["loss"])

    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    config["vae"]["training"] = training_config

    seed = int(config["experiment"].get("seed", data_config.get("seed", 751)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    log(f"Seed set to {seed}")

    model_input_path = project_path(data_config["model_input_path"])
    mel_tensor_dir = project_path(data_config["mel_tensor_dir"])
    label_path = project_path(data_config["label_path"])
    input_embedding_column = str(data_config.get("input_embedding_column", "lyrics_e5_large_embedding"))
    language_column = str(data_config.get("language_column", "music_lang"))
    log(f"Model input parquet: {model_input_path}")
    log(f"Mel tensor directory: {mel_tensor_dir}")
    log(f"Label parquet: {label_path}")
    log(f"Lyrics embedding column: {input_embedding_column}")
    log(f"Language split column: {language_column}")

    log("Loading full audio+lyrics dataset")
    all_dataset = AudioLyricsVAEDataset(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        input_embedding_column=input_embedding_column,
        mel_cache_chunks=int(data_config.get("mel_cache_chunks", 2)),
    )
    log(
        "Dataset rows available: "
        f"{all_dataset.full_length}, lyrics_dim={all_dataset.input_dim}, "
        f"mel_shape={all_dataset.mel_shape}"
    )

    log("Loading row-aligned labels for per-language split")
    labels = load_label_frame(
        label_path,
        language_column=language_column,
        row_count=all_dataset.full_length,
        model_input_path=model_input_path,
    )
    split = create_language_ratio_split(
        labels,
        train_ratio=float(data_config.get("train_ratio", 0.85)),
        val_ratio=float(data_config.get("val_ratio", 0.15)),
        seed=seed,
        language_column=language_column,
        max_rows_per_language=_optional_int(data_config.get("max_rows_per_language")),
        min_rows_per_language=int(data_config.get("min_rows_per_language", 2)),
    )
    save_split(split, output_dir / "split_indices.json")
    log(
        "Saved split_indices.json: "
        f"selected={split['row_count']}, "
        f"train={len(split['train_indices'])}, "
        f"val={len(split['val_indices'])}, "
        f"strategy={split['split_strategy']}"
    )
    for language, counts in split["per_language"].items():
        log(
            f"Language {language}: "
            f"selected={counts['selected']}, train={counts['train']}, val={counts['val']}"
        )

    batch_size = int(training_config.get("batch_size", 16))
    num_workers = int(data_config.get("num_workers", 0))
    log(f"Selecting device from setting: {training_config.get('device', 'auto')}")
    device = select_device(str(training_config.get("device", "auto")))
    log(f"Using device: {device}")
    pin_memory = device.type == "cuda"

    log("Building train dataset")
    train_dataset = AudioLyricsVAEDataset(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        input_embedding_column=input_embedding_column,
        mel_cache_chunks=int(data_config.get("mel_cache_chunks", 2)),
        indices=split["train_indices"],
    )
    log("Building validation dataset")
    val_dataset = AudioLyricsVAEDataset(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        input_embedding_column=input_embedding_column,
        mel_cache_chunks=int(data_config.get("mel_cache_chunks", 2)),
        indices=split["val_indices"],
    )

    configured_input_dim = model_config.get("lyrics_input_dim", "auto")
    if configured_input_dim in {None, "auto"}:
        model_config["lyrics_input_dim"] = train_dataset.input_dim
    elif int(configured_input_dim) != train_dataset.input_dim:
        raise ValueError(
            f"Config model.lyrics_input_dim={configured_input_dim} does not match "
            f"{input_embedding_column} dim={train_dataset.input_dim}."
        )
    config["vae"]["model"] = model_config

    log(
        "Creating dataloaders: "
        f"batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}"
    )
    train_loader = make_loader(train_dataset, batch_size, True, num_workers, pin_memory)
    val_loader = make_loader(val_dataset, batch_size, False, num_workers, pin_memory)
    stat_loader = make_loader(train_dataset, batch_size, False, num_workers, pin_memory)
    log(f"Train batches per epoch: {len(train_loader)}")
    log(f"Validation batches per epoch: {len(val_loader)}")

    save_yaml(config, output_dir / "resolved_config.yaml")
    save_yaml(
        {
            "run_name": config["experiment"]["run_name"],
            "output_dir": str(output_dir),
            "config_paths": config["config_paths"],
        },
        output_dir / "run_manifest.yaml",
    )
    log("Saved resolved_config.yaml and run_manifest.yaml")

    stat_batches = args.max_stat_batches
    if stat_batches is None:
        stat_batches = args.max_train_batches
    if stat_batches is None:
        log(
            "Computing mel normalization stats from full train split "
            "with a non-shuffled sequential loader"
        )
    else:
        log(
            f"Computing mel normalization stats from first {stat_batches} train batches "
            "with a non-shuffled sequential loader"
        )
    normalizer = compute_mel_normalizer(stat_loader, max_batches=stat_batches)
    normalizer.save(output_dir / "mel_normalization.json")
    log(
        "Saved mel_normalization.json: "
        f"mean={normalizer.mean:.6f}, std={normalizer.std:.6f}"
    )

    log("Building AudioLyricsConcatVAE model")
    model = build_model(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log(f"Model loaded to {device} with {parameter_count:,} parameters")
    log("Creating AdamW optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 2e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
        betas=(0.9, 0.999),
    )

    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()
        log(f"Removed existing metrics file: {metrics_path}")
    step_metrics_path = run_paths["training_logs"] / "step_metrics.csv"
    if step_metrics_path.exists():
        step_metrics_path.unlink()
        log(f"Removed existing step metrics file: {step_metrics_path}")

    epochs = int(training_config.get("epochs", 60))
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    train_steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        train_steps_per_epoch = min(train_steps_per_epoch, max_train_batches)
    total_steps = max(1, epochs * train_steps_per_epoch)
    log(
        "Training schedule: "
        f"epochs={epochs}, train_steps_per_epoch={train_steps_per_epoch}, "
        f"total_steps={total_steps}, max_train_batches={max_train_batches}, "
        f"max_val_batches={max_val_batches}"
    )
    checkpoint_metric = str(training_config.get("checkpoint_metric", "recon_loss"))
    checkpoint_mode = str(training_config.get("checkpoint_mode", "min")).lower()
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("training.checkpoint_mode must be either 'min' or 'max'.")
    patience = int(training_config.get("early_stopping_patience", 10))
    log(
        "Best-checkpoint/early-stopping metric: "
        f"val/{checkpoint_metric} ({checkpoint_mode}), patience={patience}"
    )
    step_log_every = int(training_config.get("step_log_every", 1))
    val_step_log_every = int(training_config.get("val_step_log_every", step_log_every))
    log(
        "Step metrics logging: "
        f"train_every={step_log_every}, val_every={val_step_log_every}, "
        f"path={step_metrics_path}"
    )

    best_metric_value = float("inf") if checkpoint_mode == "min" else -float("inf")
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        log(f"Starting epoch {epoch}/{epochs}")
        train_metrics, global_step, train_step_metrics = run_epoch(
            model,
            train_loader,
            normalizer,
            loss_config,
            device,
            total_steps,
            global_step,
            optimizer=optimizer,
            grad_clip_norm=float(training_config.get("gradient_clip_norm", 1.0)),
            mixed_precision=bool(training_config.get("mixed_precision", True)),
            max_batches=max_train_batches,
            epoch=epoch,
            split="train",
            step_log_every=step_log_every,
        )
        log(f"Epoch {epoch} train complete: {format_epoch_metrics(train_metrics)}")
        val_metrics, global_step, val_step_metrics = run_epoch(
            model,
            val_loader,
            normalizer,
            loss_config,
            device,
            total_steps,
            global_step,
            optimizer=None,
            mixed_precision=False,
            max_batches=max_val_batches,
            epoch=epoch,
            split="val",
            step_log_every=val_step_log_every,
        )
        log(f"Epoch {epoch} validation complete: {format_epoch_metrics(val_metrics)}")

        append_metrics_csv(metrics_path, metric_row(epoch, "train", train_metrics))
        append_metrics_csv(metrics_path, metric_row(epoch, "val", val_metrics))
        log(f"Appended metrics for epoch {epoch}: {metrics_path}")
        append_metrics_rows_csv(step_metrics_path, train_step_metrics + val_step_metrics)
        log(
            "Appended step metrics for epoch "
            f"{epoch}: rows={len(train_step_metrics) + len(val_step_metrics)}, "
            f"path={step_metrics_path}"
        )

        checkpoint_metrics = {"train": train_metrics, "val": val_metrics}
        if bool(training_config.get("save_every_epoch", True)):
            checkpoint_path = run_paths["checkpoints"] / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                config,
                normalizer,
                checkpoint_metrics,
            )
            log(f"Saved epoch checkpoint: {checkpoint_path}")

        metric_value = require_metric(val_metrics, checkpoint_metric)
        if is_metric_improved(metric_value, best_metric_value, checkpoint_mode):
            best_metric_value = metric_value
            epochs_without_improvement = 0
            best_path = run_paths["best"] / "model.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                config,
                normalizer,
                checkpoint_metrics,
            )
            save_json(model_config, run_paths["best"] / "model_config.json")
            log(
                f"New best checkpoint saved: {best_path} "
                f"(val/{checkpoint_metric}={best_metric_value:.6f})"
            )
        else:
            epochs_without_improvement += 1
            log(
                "No validation improvement: "
                f"val/{checkpoint_metric}={metric_value:.6f}, "
                f"best={best_metric_value:.6f}, "
                f"epochs_without_improvement={epochs_without_improvement}"
            )

        if epochs_without_improvement >= patience:
            log(f"Early stopping triggered at epoch {epoch} with patience={patience}")
            break

    log("Loading best checkpoint for latent export")
    best_checkpoint = torch.load(run_paths["best"] / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    log("Exporting validation latent embeddings")
    mu, row_indices = export_latents(
        model,
        val_loader,
        normalizer,
        device,
        max_batches=max_val_batches,
    )
    np.save(run_paths["latents"] / "val_mu.npy", mu)
    np.save(run_paths["latents"] / "val_row_indices.npy", np.array(row_indices, dtype=np.int64))
    log(f"Saved latent array: val_mu={mu.shape}")
    export_validation_labels(labels, row_indices, run_paths["latents"] / "val_labels.parquet")
    log(f"Saved validation labels with {len(row_indices)} rows")
    save_json(
        {
            "val_mu_shape": list(mu.shape),
            "val_label_rows": len(row_indices),
            "row_order_source": "validation dataloader row_index order",
            "input_embedding_column": input_embedding_column,
            "language_column": language_column,
            "split_strategy": split["split_strategy"],
            "mel_normalization": normalizer.to_dict(),
        },
        run_paths["latents"] / "latent_export_metadata.json",
    )
    log("Saved latent_export_metadata.json")

    log(f"Training complete. Output directory: {output_dir}")
    log(f"Best val/{checkpoint_metric}: {best_metric_value:.6f}")


if __name__ == "__main__":
    main()

