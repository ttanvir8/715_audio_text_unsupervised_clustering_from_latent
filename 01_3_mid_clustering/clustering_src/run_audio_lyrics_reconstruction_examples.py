from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "01_1_mid_training"
HELPER_ROOT = PROJECT_ROOT / "01_3_mid_clustering" / "extra_helper_functions"
for path in (TRAINING_ROOT, HELPER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataloader.dataset import AudioLyricsVAEDataset, audio_lyrics_collate_fn  # noqa: E402
from mid_clustering_utils import project_path, save_json, write_dataframe  # noqa: E402
from training_scripts.model import build_model  # noqa: E402
from training_scripts.normalization import MelNormalizer  # noqa: E402
from training_scripts.training_utils import prepare_batch, select_device  # noqa: E402


def log(message: str) -> None:
    print(f"[mid_recon_examples] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render VAE reconstruction examples from cluster-representative medium audio+lyrics latents."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default="01_1_mid_training/experiments/audio_lyrics_vae_run_001",
        help="Training experiment directory containing best_checkpoint/model.pt and mel_normalization.json.",
    )
    parser.add_argument(
        "--latent-root",
        default="01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001",
        help="Root containing validation/full latent exports from 01_2_inference.",
    )
    parser.add_argument(
        "--clustering-root",
        default="01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001",
        help="Root containing validation/full clustering outputs.",
    )
    parser.add_argument(
        "--output-root",
        default="01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001",
        help="Root where reconstruction artifacts will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which scope to render.",
    )
    parser.add_argument("--device", default="auto", help="Device for decoder inference.")
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=1,
        help="How many centroid-nearest examples to render per cluster.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Maximum number of clusters to render per scope.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed.")
    return parser.parse_args()


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def assignment_column(assignments: pd.DataFrame, metrics: pd.DataFrame | None = None) -> str:
    if metrics is not None and "assignment" in metrics.columns:
        candidates = [str(value) for value in metrics["assignment"].dropna().unique().tolist()]
        if candidates:
            return candidates[0]
    candidates = [column for column in assignments.columns if column.startswith("kmeans__")]
    if not candidates:
        raise ValueError("No K-Means assignment column found in cluster_assignments.parquet.")
    return candidates[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def select_representative_rows(
    latents: np.ndarray,
    metadata: pd.DataFrame,
    cluster_column: str,
    examples_per_cluster: int,
    max_clusters: int,
) -> pd.DataFrame:
    standardized_latents = StandardScaler().fit_transform(latents).astype(np.float32, copy=False)
    selections: list[pd.DataFrame] = []
    unique_clusters = sorted(metadata[cluster_column].dropna().astype(int).unique().tolist())[:max_clusters]

    for cluster_id in unique_clusters:
        cluster_mask = metadata[cluster_column].to_numpy() == cluster_id
        cluster_indices = np.flatnonzero(cluster_mask)
        if len(cluster_indices) == 0:
            continue
        cluster_values = standardized_latents[cluster_indices]
        centroid = cluster_values.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(cluster_values - centroid, axis=1)
        order = np.argsort(distances)[: max(1, int(examples_per_cluster))]
        selected_indices = cluster_indices[order]
        selected = metadata.iloc[selected_indices].copy()
        selected["latent_position"] = selected_indices
        selected["cluster_distance"] = distances[order]
        selected["cluster_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
        selections.append(selected)

    if not selections:
        raise ValueError("No representative rows were selected for reconstruction.")
    return pd.concat(selections, ignore_index=True)


def load_scope_inputs(
    latent_scope_dir: Path,
    clustering_scope_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    mu = np.load(latent_scope_dir / "mu.npy").astype(np.float32, copy=False)
    labels = pq.read_table(latent_scope_dir / "labels.parquet").to_pandas()
    assignments = pq.read_table(clustering_scope_dir / "cluster_assignments.parquet").to_pandas()
    metrics_path = clustering_scope_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    cluster_col = assignment_column(assignments, metrics)

    if mu.shape[0] != len(labels) or mu.shape[0] != len(assignments):
        raise ValueError(
            "Scope inputs are not row-aligned: "
            f"mu={mu.shape[0]}, labels={len(labels)}, assignments={len(assignments)}"
        )
    if "row_index" in labels.columns and not labels["row_index"].equals(assignments["row_index"]):
        raise ValueError("labels.parquet and cluster_assignments.parquet row_index columns differ.")

    metadata = labels.copy()
    metadata[cluster_col] = assignments[cluster_col].to_numpy()
    return mu, metadata, cluster_col


def build_loader(
    dataset: AudioLyricsVAEDataset,
    batch_size: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=audio_lyrics_collate_fn,
    )


def mel_image(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(values, a_min=0.0, a_max=None))


def plot_example_row(
    row: pd.Series,
    original_mel: np.ndarray,
    recon_mel: np.ndarray,
    error_mel: np.ndarray,
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})

    images = [
        axes[0].imshow(mel_image(original_mel), aspect="auto", origin="lower", cmap="magma"),
        axes[1].imshow(mel_image(recon_mel), aspect="auto", origin="lower", cmap="magma"),
        axes[2].imshow(error_mel, aspect="auto", origin="lower", cmap="viridis"),
    ]
    titles = ["Original Mel", "Decoded From Latent", "Absolute Error"]
    for axis, image, title in zip(axes[:3], images, titles):
        axis.set_title(title)
        axis.set_xlabel("time")
        axis.set_ylabel("mel bin")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axes[3].axis("off")
    info_lines = [
        f"music_name: {row['music_name']}",
        f"music_id: {row['music_id']}",
        f"artist: {row['art_name']}",
        f"language: {row['music_lang']}",
        f"genre: {row['main_genre']}",
        f"cluster: {int(row['cluster'])}",
        f"cluster_rank: {int(row['cluster_rank'])}",
        f"cluster_distance: {float(row['cluster_distance']):.4f}",
        f"mel_mse: {float(row['mel_mse']):.6f}",
        f"lyrics_mse: {float(row['lyrics_mse']):.6f}",
        f"lyrics_cosine: {float(row['lyrics_cosine']):.6f}",
    ]
    axes[3].text(0.0, 1.0, "\n".join(info_lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(
        f"Cluster {int(row['cluster'])} representative reconstruction",
        fontsize=14,
    )
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_gallery(
    summary: pd.DataFrame,
    originals: list[np.ndarray],
    reconstructions: list[np.ndarray],
    errors: list[np.ndarray],
    output_path: str | Path,
) -> None:
    row_count = len(summary)
    fig, axes = plt.subplots(row_count, 4, figsize=(18, max(4.5, 3.8 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, (_, row) in enumerate(summary.iterrows()):
        panels = [originals[row_index], reconstructions[row_index], errors[row_index]]
        cmaps = ["magma", "magma", "viridis"]
        titles = ["Original Mel", "Decoded From Latent", "Absolute Error"]
        for column_index, (panel, cmap, title) in enumerate(zip(panels, cmaps, titles)):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                mel_image(panel) if column_index < 2 else panel,
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            axis.set_title(title)
            axis.set_xlabel("time")
            axis.set_ylabel("mel bin")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        info_axis = axes[row_index, 3]
        info_axis.axis("off")
        info_axis.text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"music_name: {row['music_name']}",
                    f"language: {row['music_lang']}",
                    f"genre: {row['main_genre']}",
                    f"cluster: {int(row['cluster'])}",
                    f"mel_mse: {float(row['mel_mse']):.6f}",
                    f"lyrics_cosine: {float(row['lyrics_cosine']):.6f}",
                ]
            ),
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    fig.suptitle("VAE reconstruction examples from cluster-representative latents", fontsize=15)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_scope(args: argparse.Namespace, scope: str) -> dict[str, Any]:
    experiment_dir = project_path(args.experiment_dir)
    latent_scope_dir = project_path(args.latent_root) / scope
    clustering_scope_dir = project_path(args.clustering_root) / scope
    output_dir = project_path(args.output_root) / scope / "reconstruction_examples"
    per_example_dir = output_dir / "per_example"

    config = load_yaml(experiment_dir / "resolved_config.yaml")
    data_config = config["data"]
    model_config = config["vae"]["model"]
    model_input_path = project_path(data_config["model_input_path"])
    mel_tensor_dir = project_path(data_config["mel_tensor_dir"])
    input_embedding_column = str(data_config.get("input_embedding_column", "lyrics_e5_large_embedding"))

    mu, metadata, cluster_col = load_scope_inputs(latent_scope_dir, clustering_scope_dir)
    metadata = metadata.copy()
    metadata = metadata.rename(columns={cluster_col: "cluster"})
    selected = select_representative_rows(
        latents=mu,
        metadata=metadata,
        cluster_column="cluster",
        examples_per_cluster=int(args.examples_per_cluster),
        max_clusters=int(args.max_clusters),
    )
    selected = selected.reset_index(drop=True)

    device = select_device(args.device)
    pin_memory = device.type == "cuda"
    model = build_model(model_config).to(device)
    checkpoint = torch.load(experiment_dir / "best_checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = MelNormalizer.from_file(experiment_dir / "mel_normalization.json")

    dataset = AudioLyricsVAEDataset(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        input_embedding_column=input_embedding_column,
        mel_cache_chunks=int(data_config.get("mel_cache_chunks", 2)),
        indices=selected["row_index"].astype(int).tolist(),
    )
    loader = build_loader(dataset, len(selected), pin_memory)
    batch = next(iter(loader))
    prepared = prepare_batch(batch, device, normalizer)
    z = torch.from_numpy(mu[selected["latent_position"].to_numpy(dtype=np.int64)]).to(device)

    with torch.no_grad():
        decoded = model.decode(z)

    original_mel = batch["melspectrogram"].cpu().numpy()
    recon_mel = normalizer.denormalize(decoded["mel_recon_norm"]).cpu().numpy()
    lyrics_original = batch["lyrics_input"].cpu().numpy()
    lyrics_recon = decoded["lyrics_recon"].cpu().numpy()
    dataset.close()

    originals: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for index, (_, selected_row) in enumerate(selected.iterrows()):
        mel_orig = original_mel[index, 0]
        mel_hat = recon_mel[index, 0]
        mel_err = np.abs(mel_orig - mel_hat)
        lyrics_orig = lyrics_original[index]
        lyrics_hat = lyrics_recon[index]

        row_dict = selected_row.to_dict()
        row_dict["mel_mse"] = float(np.mean((mel_orig - mel_hat) ** 2))
        row_dict["lyrics_mse"] = float(np.mean((lyrics_orig - lyrics_hat) ** 2))
        row_dict["lyrics_cosine"] = cosine_similarity(lyrics_orig, lyrics_hat)
        rows.append(row_dict)

        originals.append(mel_orig)
        reconstructions.append(mel_hat)
        errors.append(mel_err)

    summary = pd.DataFrame(rows)
    write_dataframe(summary, output_dir / "selected_reconstruction_examples.csv")

    for index, row in summary.iterrows():
        safe_music = str(row["music_name"]).replace("/", "_")[:48]
        example_path = per_example_dir / (
            f"cluster_{int(row['cluster']):02d}_rank_{int(row['cluster_rank'])}_"
            f"row_{int(row['row_index'])}_{safe_music}.png"
        )
        plot_example_row(
            row=row,
            original_mel=originals[index],
            recon_mel=reconstructions[index],
            error_mel=errors[index],
            output_path=example_path,
        )

    plot_gallery(
        summary=summary,
        originals=originals,
        reconstructions=reconstructions,
        errors=errors,
        output_path=output_dir / "reconstruction_gallery.png",
    )

    manifest = {
        "scope": scope,
        "experiment_dir": str(experiment_dir),
        "latent_scope_dir": str(latent_scope_dir),
        "clustering_scope_dir": str(clustering_scope_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "cluster_column": cluster_col,
        "examples_per_cluster": int(args.examples_per_cluster),
        "max_clusters": int(args.max_clusters),
        "rendered_examples": int(len(summary)),
        "artifacts": [
            "selected_reconstruction_examples.csv",
            "reconstruction_gallery.png",
            "per_example/*.png",
        ],
    }
    save_json(manifest, output_dir / "reconstruction_manifest.json")
    log(f"Saved {scope} reconstruction outputs: {output_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    manifests = [run_scope(args, scope) for scope in scopes_from_arg(args.scope)]
    save_json(
        {
            "scope_count": len(manifests),
            "scopes": [manifest["scope"] for manifest in manifests],
            "output_root": str(project_path(args.output_root)),
        },
        project_path(args.output_root) / "reconstruction_manifest.json",
    )
    log(f"Reconstruction rendering complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
