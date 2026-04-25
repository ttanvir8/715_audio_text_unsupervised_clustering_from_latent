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
    fit_kmeans,
    load_latent_scope,
    named_cluster_summary,
    observed_labels,
    parse_int_csv_list,
    project_path,
    safe_davies_bouldin,
    save_json,
    standardized,
    supervised_metrics,
    write_dataframe,
)


SUMMARY_LABEL_SPECS = [
    ("music_lang", "language"),
    ("main_genre", "genre"),
]


def log(message: str) -> None:
    print(f"[mid_cluster_latents] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster medium audio+lyrics VAE encoder latents and evaluate against label targets."
    )
    parser.add_argument(
        "--latent-root",
        default="01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001",
        help="Root containing validation/full latent exports from 01_2_inference.",
    )
    parser.add_argument(
        "--output-root",
        default="01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001",
        help="Root where clustering assignments and metrics will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which latent scope to cluster.",
    )
    parser.add_argument(
        "--embedding",
        default="mu",
        choices=("mu", "logvar"),
        help="Latent array to cluster.",
    )
    parser.add_argument(
        "--label-column",
        default="music_lang",
        help="Target label column used for the main evaluation.",
    )
    parser.add_argument(
        "--label-id-column",
        default="music_lang_id",
        help="Integer id column aligned with --label-column. Falls back to factorization when missing.",
    )
    parser.add_argument(
        "--target-name",
        default="language",
        help="Display name for the main evaluation target.",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=0,
        help="Cluster count for the main run. 0 means the number of observed labels.",
    )
    parser.add_argument(
        "--sweep-k",
        default="2,3,4,5,6,8",
        help="Comma-separated k values for the silhouette and target-metric sweep.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed for clustering.")
    parser.add_argument("--n-init", type=int, default=50, help="KMeans initializations.")
    parser.add_argument(
        "--make-plot",
        action="store_true",
        help="Save a PCA scatter plot colored by cluster and marked by the main label.",
    )
    return parser.parse_args()


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def assignment_name(embedding: str, k: int, target_name: str) -> str:
    return f"kmeans__{embedding}__{target_name}_k{k}"


def make_pca_plot(
    values: np.ndarray,
    assignments: pd.DataFrame,
    assignment_column: str,
    label_column: str,
    output_path: str | Path,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    component_count = min(2, values.shape[0], values.shape[1])
    if component_count < 2:
        return

    points = PCA(n_components=2, random_state=seed).fit_transform(values)
    labels = assignments[label_column].astype(str).fillna("nan")
    unique_labels = sorted(labels.unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    show_legend = len(unique_labels) <= len(markers)

    plt.figure(figsize=(8, 6))
    for label_index, label_value in enumerate(unique_labels):
        mask = labels.to_numpy() == label_value
        marker = markers[label_index % len(markers)] if show_legend else "o"
        kwargs: dict[str, Any] = {
            "c": assignments.loc[mask, assignment_column].to_numpy(),
            "cmap": "tab20",
            "marker": marker,
            "s": 26,
            "alpha": 0.80,
            "edgecolors": "none",
        }
        if show_legend:
            kwargs["label"] = label_value
        plt.scatter(points[mask, 0], points[mask, 1], **kwargs)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"{assignment_column} on audio+lyrics VAE latents")
    if show_legend:
        plt.legend(title=label_column, loc="best")
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


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
    values, _, labels = load_latent_scope(latent_scope_dir, args.embedding)
    values_std = standardized(values)

    target_ids, target_names = observed_labels(labels, args.label_column, args.label_id_column)
    cluster_k = int(args.cluster_k) if args.cluster_k else int(len(target_names))

    log(
        f"Clustering {scope}: rows={values.shape[0]}, dim={values.shape[1]}, "
        f"target={args.label_column}, labels={len(target_names)}, k={cluster_k}"
    )
    main_assignment = assignment_name(args.embedding, cluster_k, args.target_name)
    main_clusters = fit_kmeans(values_std, cluster_k, args.seed, args.n_init)
    assignments = labels[assignment_base_columns(labels, args.label_column, args.label_id_column)].copy()
    assignments[main_assignment] = main_clusters

    metrics_rows: list[dict[str, Any]] = []
    main_metrics = {
        "scope": scope,
        "embedding": args.embedding,
        "algorithm": "kmeans",
        "assignment": main_assignment,
        "k": cluster_k,
        "sample_count": int(values.shape[0]),
        "cluster_count": int(len(np.unique(main_clusters))),
        "target": args.target_name,
        "label_column": args.label_column,
        "davies_bouldin": safe_davies_bouldin(values_std, main_clusters),
    }
    main_metrics.update(supervised_metrics(values_std, main_clusters, target_ids))
    metrics_rows.append(main_metrics)

    sweep_rows: list[dict[str, Any]] = []
    for k in parse_int_csv_list(args.sweep_k):
        if k < 2 or k > values_std.shape[0]:
            continue
        clusters = fit_kmeans(values_std, k, args.seed, args.n_init)
        row = {
            "scope": scope,
            "embedding": args.embedding,
            "algorithm": "kmeans",
            "k": int(k),
            "sample_count": int(values_std.shape[0]),
            "cluster_count": int(len(np.unique(clusters))),
            "target": args.target_name,
            "label_column": args.label_column,
            "davies_bouldin": safe_davies_bouldin(values_std, clusters),
        }
        row.update(supervised_metrics(values_std, clusters, target_ids))
        sweep_rows.append(row)

    write_dataframe(assignments, output_dir / "cluster_assignments.parquet")
    write_dataframe(pd.DataFrame(metrics_rows), output_dir / "metrics.csv")
    write_dataframe(pd.DataFrame(sweep_rows), output_dir / "silhouette_sweep.csv")

    for summary_column, summary_slug in SUMMARY_LABEL_SPECS:
        if summary_column not in assignments.columns:
            continue
        summary = named_cluster_summary(assignments, main_assignment, summary_column, summary_slug)
        write_dataframe(summary, output_dir / f"cluster_summary_by_{summary_slug}.csv")

    save_json(
        {
            "latent_scope_dir": str(latent_scope_dir),
            "output_dir": str(output_dir),
            "scope": scope,
            "embedding": args.embedding,
            "label_column": args.label_column,
            "label_id_column": args.label_id_column,
            "target_name": args.target_name,
            "target_labels": list(target_names),
            "cluster_k": int(cluster_k),
            "seed": int(args.seed),
            "n_init": int(args.n_init),
            "sweep_k": parse_int_csv_list(args.sweep_k),
            "value_shape": list(values.shape),
            "standardized": True,
        },
        output_dir / "run_metadata.json",
    )
    if args.make_plot:
        make_pca_plot(
            values_std,
            assignments,
            main_assignment,
            args.label_column,
            output_dir / "latent_pca_cluster_scatter.png",
            args.seed,
        )
    log(f"Saved {scope} clustering outputs: {output_dir}")


def main() -> None:
    args = parse_args()
    for scope in scopes_from_arg(args.scope):
        run_scope(args, scope)
    log(f"Clustering complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
