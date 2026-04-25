from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_ROOT = PROJECT_ROOT / "01_3_mid_clustering" / "extra_helper_functions"
if str(HELPER_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPER_ROOT))

from mid_clustering_utils import (  # noqa: E402
    audio_lyrics_raw_features,
    autoencoder_embedding,
    fit_kmeans,
    load_latent_scope,
    observed_labels,
    pca_embedding,
    project_path,
    safe_calinski_harabasz,
    safe_davies_bouldin,
    save_json,
    standardized,
    supervised_metrics,
    write_dataframe,
)


def log(message: str) -> None:
    print(f"[mid_baseline_compare] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare medium audio+lyrics VAE latent K-Means against PCA, autoencoder, "
            "and direct spectral baselines."
        )
    )
    parser.add_argument(
        "--latent-root",
        default="01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001",
        help="Root containing validation/full latent exports from 01_2_inference.",
    )
    parser.add_argument(
        "--output-root",
        default="01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001",
        help="Root where comparison outputs will be written.",
    )
    parser.add_argument(
        "--model-input-path",
        default="0_data_pre_processing/processed_dataset/model_input/model_input_dataset.parquet",
        help="Processed model input parquet used for lyrics embeddings in the baselines.",
    )
    parser.add_argument(
        "--mel-tensor-dir",
        default="0_data_pre_processing/processed_dataset/mel_tensors",
        help="Mel tensor directory used for direct spectral feature clustering.",
    )
    parser.add_argument(
        "--lyrics-embedding-column",
        default="lyrics_e5_large_embedding",
        help="Original lyrics embedding column used by the audio+lyrics baselines.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which latent scope to compare.",
    )
    parser.add_argument(
        "--embedding",
        default="mu",
        choices=("mu", "logvar"),
        help="VAE latent array to compare against the baselines.",
    )
    parser.add_argument(
        "--label-column",
        default="music_lang",
        help="Target label column used for evaluation.",
    )
    parser.add_argument(
        "--label-id-column",
        default="music_lang_id",
        help="Integer id column aligned with --label-column. Falls back to factorization when missing.",
    )
    parser.add_argument(
        "--target-name",
        default="language",
        help="Display name for the evaluation target.",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=0,
        help="Cluster count. 0 means number of observed target labels.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
        help="PCA dimensions for the raw audio+lyrics baseline before K-Means.",
    )
    parser.add_argument(
        "--spectral-time-bins",
        type=int,
        default=32,
        help="Number of pooled time bins used in compact spectral features.",
    )
    parser.add_argument(
        "--autoencoder-latent-dim",
        type=int,
        default=64,
        help="Latent dimensions for the raw audio+lyrics autoencoder baseline.",
    )
    parser.add_argument(
        "--autoencoder-epochs",
        type=int,
        default=40,
        help="Training epochs for the raw audio+lyrics autoencoder baseline.",
    )
    parser.add_argument(
        "--autoencoder-batch-size",
        type=int,
        default=256,
        help="Batch size for the raw audio+lyrics autoencoder baseline.",
    )
    parser.add_argument(
        "--autoencoder-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the raw audio+lyrics autoencoder baseline.",
    )
    parser.add_argument(
        "--autoencoder-device",
        default="auto",
        help="Device for the autoencoder baseline: auto,cpu,cuda,cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed.")
    parser.add_argument("--n-init", type=int, default=50, help="KMeans initializations.")
    return parser.parse_args()


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def metric_row(
    *,
    scope: str,
    method: str,
    feature_source: str,
    values: np.ndarray,
    clusters: np.ndarray,
    target_ids: np.ndarray,
    k: int,
    target_name: str,
    label_column: str,
    explained_variance_ratio: float | None = None,
) -> dict[str, Any]:
    row = {
        "scope": scope,
        "method": method,
        "feature_source": feature_source,
        "target": target_name,
        "label_column": label_column,
        "k": int(k),
        "sample_count": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "cluster_count": int(len(np.unique(clusters))),
        "silhouette": float(supervised_metrics(values, clusters, target_ids)["silhouette"]),
        "calinski_harabasz": safe_calinski_harabasz(values, clusters),
        "davies_bouldin": safe_davies_bouldin(values, clusters),
        "pca_explained_variance_ratio_sum": explained_variance_ratio,
    }
    row.update(supervised_metrics(values, clusters, target_ids))
    return row


def assignment_base_columns(labels: pd.DataFrame, label_column: str, label_id_column: str | None) -> list[str]:
    desired = [
        "row_index",
        "music_id",
        "music_name",
        "art_id",
        "art_name",
        "music_lang",
        "music_lang_id",
        "main_genre",
        "main_genre_id",
    ]
    if label_column not in desired:
        desired.append(label_column)
    if label_id_column and label_id_column not in desired and label_id_column in labels.columns:
        desired.append(label_id_column)
    return [column for column in desired if column in labels.columns]


def run_scope(args: argparse.Namespace, scope: str) -> None:
    latent_scope_dir = project_path(args.latent_root) / scope
    output_dir = project_path(args.output_root) / scope
    model_input_path = project_path(args.model_input_path)
    mel_tensor_dir = project_path(args.mel_tensor_dir)

    vae_values, row_indices, labels = load_latent_scope(latent_scope_dir, args.embedding)
    target_ids, target_names = observed_labels(labels, args.label_column, args.label_id_column)
    k = int(args.cluster_k) if args.cluster_k else int(len(target_names))

    log(
        f"Comparing {scope}: rows={vae_values.shape[0]}, vae_dim={vae_values.shape[1]}, "
        f"target={args.label_column}, labels={len(target_names)}, k={k}"
    )

    raw_audio_lyrics, lyrics_values, spectral_values = audio_lyrics_raw_features(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        row_indices=row_indices,
        lyrics_embedding_column=args.lyrics_embedding_column,
        spectral_time_bins=args.spectral_time_bins,
    )
    log(
        f"Loaded baseline features: raw_audio_lyrics={raw_audio_lyrics.shape}, "
        f"lyrics={lyrics_values.shape}, spectral={spectral_values.shape}"
    )

    vae_features = standardized(vae_values)
    vae_clusters = fit_kmeans(vae_features, k, args.seed, args.n_init)

    pca_features, pca_model = pca_embedding(raw_audio_lyrics, args.pca_components, args.seed)
    pca_clusters = fit_kmeans(pca_features, k, args.seed, args.n_init)

    autoencoder_features = standardized(
        autoencoder_embedding(
            raw_audio_lyrics,
            latent_dim=args.autoencoder_latent_dim,
            epochs=args.autoencoder_epochs,
            batch_size=args.autoencoder_batch_size,
            learning_rate=args.autoencoder_learning_rate,
            seed=args.seed,
            device_name=args.autoencoder_device,
        )
    )
    autoencoder_clusters = fit_kmeans(autoencoder_features, k, args.seed, args.n_init)

    spectral_features = standardized(spectral_values)
    spectral_clusters = fit_kmeans(spectral_features, k, args.seed, args.n_init)

    comparison = pd.DataFrame(
        [
            metric_row(
                scope=scope,
                method=f"vae_{args.embedding}_kmeans",
                feature_source=f"standardized_{args.embedding}",
                values=vae_features,
                clusters=vae_clusters,
                target_ids=target_ids,
                k=k,
                target_name=args.target_name,
                label_column=args.label_column,
            ),
            metric_row(
                scope=scope,
                method="pca_audio_lyrics_kmeans",
                feature_source="standardized_audio_lyrics_pca",
                values=pca_features,
                clusters=pca_clusters,
                target_ids=target_ids,
                k=k,
                target_name=args.target_name,
                label_column=args.label_column,
                explained_variance_ratio=float(np.sum(pca_model.explained_variance_ratio_)),
            ),
            metric_row(
                scope=scope,
                method="autoencoder_audio_lyrics_kmeans",
                feature_source="standardized_audio_lyrics_autoencoder",
                values=autoencoder_features,
                clusters=autoencoder_clusters,
                target_ids=target_ids,
                k=k,
                target_name=args.target_name,
                label_column=args.label_column,
            ),
            metric_row(
                scope=scope,
                method="spectral_feature_kmeans",
                feature_source="standardized_compact_spectral_features",
                values=spectral_features,
                clusters=spectral_clusters,
                target_ids=target_ids,
                k=k,
                target_name=args.target_name,
                label_column=args.label_column,
            ),
        ]
    )

    assignments = labels[assignment_base_columns(labels, args.label_column, args.label_id_column)].copy()
    assignments[f"vae_{args.embedding}_kmeans_k{k}"] = vae_clusters
    assignments[f"pca_audio_lyrics_kmeans_k{k}"] = pca_clusters
    assignments[f"autoencoder_audio_lyrics_kmeans_k{k}"] = autoencoder_clusters
    assignments[f"spectral_feature_kmeans_k{k}"] = spectral_clusters

    write_dataframe(comparison, output_dir / "baseline_kmeans_comparison.csv")
    write_dataframe(assignments, output_dir / "baseline_kmeans_assignments.parquet")
    save_json(
        {
            "latent_scope_dir": str(latent_scope_dir),
            "output_dir": str(output_dir),
            "model_input_path": str(model_input_path),
            "mel_tensor_dir": str(mel_tensor_dir),
            "scope": scope,
            "vae_embedding": args.embedding,
            "label_column": args.label_column,
            "label_id_column": args.label_id_column,
            "target_name": args.target_name,
            "target_labels": list(target_names),
            "cluster_k": int(k),
            "lyrics_embedding_column": args.lyrics_embedding_column,
            "spectral_time_bins": int(args.spectral_time_bins),
            "pca_components_requested": int(args.pca_components),
            "pca_components_used": int(pca_features.shape[1]),
            "pca_explained_variance_ratio_sum": float(np.sum(pca_model.explained_variance_ratio_)),
            "autoencoder": {
                "latent_dim": int(args.autoencoder_latent_dim),
                "epochs": int(args.autoencoder_epochs),
                "batch_size": int(args.autoencoder_batch_size),
                "learning_rate": float(args.autoencoder_learning_rate),
                "device": args.autoencoder_device,
            },
            "seed": int(args.seed),
            "n_init": int(args.n_init),
            "outputs": [
                "baseline_kmeans_comparison.csv",
                "baseline_kmeans_assignments.parquet",
                "baseline_kmeans_metadata.json",
            ],
        },
        output_dir / "baseline_kmeans_metadata.json",
    )
    log(f"Saved {scope} baseline comparison: {output_dir}")


def main() -> None:
    args = parse_args()
    for scope in scopes_from_arg(args.scope):
        run_scope(args, scope)
    log(f"Baseline comparison complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
