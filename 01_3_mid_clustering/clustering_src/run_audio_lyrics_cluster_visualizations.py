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
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_ROOT = PROJECT_ROOT / "01_3_mid_clustering" / "extra_helper_functions"
if str(HELPER_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPER_ROOT))

from mid_clustering_utils import project_path, save_json, write_dataframe  # noqa: E402


def log(message: str) -> None:
    print(f"[mid_cluster_visualize] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create latent-space plots, cluster distributions, and metric comparison "
            "figures for medium audio+lyrics clustering outputs."
        )
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
        help="Root where visualization artifacts will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which scope to visualize.",
    )
    parser.add_argument(
        "--embedding",
        choices=("mu", "logvar"),
        default="mu",
        help="Latent array to visualize.",
    )
    parser.add_argument(
        "--max-manifold-points",
        type=int,
        default=3000,
        help="Maximum rows used for t-SNE per scope. Use 0 or a negative value for all rows.",
    )
    parser.add_argument(
        "--genre-top-n",
        type=int,
        default=12,
        help="How many genre labels to keep separate in genre-colored plots and stacked bars.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Requested t-SNE perplexity.")
    parser.add_argument("--tsne-max-iter", type=int, default=1000, help="t-SNE optimization steps.")
    return parser.parse_args()


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def assignment_column(assignments: pd.DataFrame, metrics: pd.DataFrame | None = None) -> str:
    if metrics is not None and "assignment" in metrics.columns:
        candidates = [str(value) for value in metrics["assignment"].dropna().unique().tolist()]
        if candidates:
            return candidates[0]
    candidates = [column for column in assignments.columns if column.startswith("kmeans__")]
    if not candidates:
        raise ValueError("No K-Means assignment column found in cluster_assignments.parquet.")
    return candidates[0]


def balanced_sample_indices(metadata: pd.DataFrame, max_points: int, seed: int) -> np.ndarray:
    row_count = len(metadata)
    if max_points <= 0 or row_count <= max_points:
        return np.arange(row_count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    groups = [group.index.to_numpy(dtype=np.int64) for _, group in metadata.groupby("cluster", sort=True)]
    target_per_cluster = max(1, max_points // max(1, len(groups)))

    for indices in groups:
        chosen = rng.choice(indices, size=min(len(indices), target_per_cluster), replace=False)
        selected.update(int(index) for index in chosen.tolist())

    remaining = max_points - len(selected)
    if remaining > 0:
        available = np.array(
            [index for index in range(row_count) if index not in selected],
            dtype=np.int64,
        )
        if len(available):
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            selected.update(int(index) for index in extra.tolist())

    return np.asarray(sorted(selected)[:max_points], dtype=np.int64)


def effective_perplexity(sample_count: int, requested: float) -> float:
    if sample_count < 4:
        raise ValueError(f"Need at least 4 points for t-SNE, got {sample_count}.")
    upper = max(1.0, (sample_count - 1) / 3.0)
    return float(min(requested, upper))


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path)


def standardized(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values).astype(np.float32, copy=False)


def load_scope_inputs(
    latent_scope_dir: Path,
    clustering_scope_dir: Path,
    embedding: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, str]:
    values = np.load(latent_scope_dir / f"{embedding}.npy").astype(np.float32, copy=False)
    labels = pq.read_table(latent_scope_dir / "labels.parquet").to_pandas()
    assignments = pq.read_table(clustering_scope_dir / "cluster_assignments.parquet").to_pandas()
    metrics_path = clustering_scope_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    assign_col = assignment_column(assignments, metrics)

    if values.shape[0] != len(labels) or values.shape[0] != len(assignments):
        raise ValueError(
            "Scope inputs are not row-aligned: "
            f"values={values.shape[0]}, labels={len(labels)}, assignments={len(assignments)}"
        )
    if "row_index" in labels.columns and not labels["row_index"].equals(assignments["row_index"]):
        raise ValueError("labels.parquet and cluster_assignments.parquet row_index columns differ.")

    metadata = labels.copy()
    metadata["cluster"] = assignments[assign_col].to_numpy()
    return values, metadata, assignments, assign_col


def top_n_with_other(values: pd.Series, top_n: int) -> pd.Series:
    counts = values.astype(str).value_counts()
    keep = set(counts.head(max(1, int(top_n))).index.tolist())
    collapsed = values.astype(str).copy()
    return collapsed.where(collapsed.isin(keep), other="Other")


def scatter_by_category(
    points: np.ndarray,
    metadata: pd.DataFrame,
    category_column: str,
    title: str,
    output_path: str | Path,
    show_legend: bool = True,
) -> None:
    categories = metadata[category_column].astype(str)
    names = sorted(categories.unique().tolist())
    cmap_name = "tab10" if len(names) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    colors = {name: cmap(index % cmap.N) for index, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in names:
        mask = categories == name
        ax.scatter(
            points[mask.to_numpy(), 0],
            points[mask.to_numpy(), 1],
            s=16,
            alpha=0.76,
            color=colors[name],
            label=name,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    if show_legend:
        ax.legend(title=category_column, loc="best", markerscale=1.3, fontsize=8)
    ax.grid(alpha=0.18)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_point_table(
    metadata: pd.DataFrame,
    points: np.ndarray,
    x_name: str,
    y_name: str,
    output_path: str | Path,
) -> None:
    table = metadata.copy()
    table[x_name] = points[:, 0]
    table[y_name] = points[:, 1]
    write_parquet(table, output_path)


def pca_points(values: np.ndarray, seed: int) -> tuple[np.ndarray, PCA]:
    component_count = min(2, values.shape[0], values.shape[1])
    if component_count < 2:
        raise ValueError(f"Need at least two rows and two dims for PCA, got {values.shape}.")
    model = PCA(n_components=2, random_state=seed)
    points = model.fit_transform(values).astype(np.float32, copy=False)
    return points, model


def tsne_points(values: np.ndarray, seed: int, perplexity: float, max_iter: int) -> np.ndarray:
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity(values.shape[0], perplexity),
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=max_iter,
    )
    return reducer.fit_transform(values).astype(np.float32, copy=False)


def plot_distribution_heatmap(
    counts: pd.DataFrame,
    title: str,
    output_path: str | Path,
) -> None:
    fig_width = max(8.0, 0.45 * counts.shape[1] + 4.0)
    fig_height = max(4.5, 0.45 * counts.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    values = counts.to_numpy(dtype=np.float32)
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(counts.shape[1]))
    ax.set_yticks(np.arange(counts.shape[0]))
    ax.set_xticklabels(counts.columns.tolist(), rotation=45, ha="right")
    ax.set_yticklabels(counts.index.tolist())
    ax.set_xlabel("label")
    ax.set_ylabel("cluster")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_distribution_stacked_bar(
    counts: pd.DataFrame,
    title: str,
    output_path: str | Path,
) -> None:
    proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig_width = max(8.0, 0.55 * counts.shape[0] + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bottom = np.zeros(len(proportions), dtype=np.float32)
    cmap = plt.get_cmap("tab20")

    for index, column in enumerate(proportions.columns.tolist()):
        values = proportions[column].to_numpy(dtype=np.float32)
        ax.bar(
            proportions.index.astype(str),
            values,
            bottom=bottom,
            label=str(column),
            color=cmap(index % cmap.N),
            width=0.75,
        )
        bottom += values

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("cluster")
    ax.set_ylabel("proportion")
    ax.set_title(title)
    ax.legend(title="label", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def metric_chart_frame(scope_dir: Path) -> pd.DataFrame:
    baseline_path = scope_dir / "baseline_kmeans_comparison.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        desired_columns = ["method", "silhouette", "nmi", "ari", "purity"]
        return baseline[desired_columns].copy()

    metrics_path = scope_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return pd.DataFrame()
    metric_frame = metrics[["assignment", "silhouette", "nmi", "ari", "purity"]].copy()
    metric_frame = metric_frame.rename(columns={"assignment": "method"})
    return metric_frame


def plot_metric_comparison(metric_frame: pd.DataFrame, output_path: str | Path) -> None:
    if metric_frame.empty:
        return

    metric_columns = ["silhouette", "nmi", "ari", "purity"]
    methods = metric_frame["method"].astype(str).tolist()
    x = np.arange(len(metric_columns), dtype=np.float32)
    width = 0.8 / max(1, len(methods))
    fig_width = max(8.5, 1.6 * len(methods) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for index, (_, row) in enumerate(metric_frame.iterrows()):
        values = [float(row[column]) for column in metric_columns]
        offset = x + index * width - 0.4 + width / 2.0
        ax.bar(offset, values, width, label=str(row["method"]))

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_columns)
    ax.set_ylabel("score")
    ax.set_title("Clustering Metric Comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_distribution_frames(
    metadata: pd.DataFrame,
    label_column: str,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = metadata[label_column].astype(str)
    if top_n is not None:
        labels = top_n_with_other(labels, top_n)
    counts = pd.crosstab(metadata["cluster"].astype(str), labels)
    proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return counts, proportions


def run_scope(args: argparse.Namespace, scope: str) -> dict[str, Any]:
    latent_scope_dir = project_path(args.latent_root) / scope
    clustering_scope_dir = project_path(args.clustering_root) / scope
    output_dir = project_path(args.output_root) / scope / "visualizations"
    latent_dir = output_dir / "latent_space"
    distribution_dir = output_dir / "distributions"
    metrics_dir = output_dir / "metrics"

    values, metadata, assignments, assign_col = load_scope_inputs(
        latent_scope_dir=latent_scope_dir,
        clustering_scope_dir=clustering_scope_dir,
        embedding=args.embedding,
    )
    values_std = standardized(values)

    log(
        f"Visualizing {scope}: embedding={args.embedding}, rows={len(metadata)}, "
        f"assignment={assign_col}"
    )

    pca_vals, pca_model = pca_points(values_std, int(args.seed))
    save_point_table(metadata, pca_vals, "pca_x", "pca_y", latent_dir / f"pca_{args.embedding}_points.parquet")
    scatter_by_category(
        pca_vals,
        metadata,
        "cluster",
        f"{scope} {args.embedding} PCA colored by cluster",
        latent_dir / f"pca_{args.embedding}_colored_by_cluster.png",
    )
    scatter_by_category(
        pca_vals,
        metadata,
        "music_lang",
        f"{scope} {args.embedding} PCA colored by language",
        latent_dir / f"pca_{args.embedding}_colored_by_language.png",
    )
    genre_plot_metadata = metadata.copy()
    genre_plot_metadata["genre_top"] = top_n_with_other(genre_plot_metadata["main_genre"], int(args.genre_top_n))
    scatter_by_category(
        pca_vals,
        genre_plot_metadata,
        "genre_top",
        f"{scope} {args.embedding} PCA colored by top genres",
        latent_dir / f"pca_{args.embedding}_colored_by_genre_top{int(args.genre_top_n)}.png",
        show_legend=True,
    )

    sample_indices = balanced_sample_indices(metadata, int(args.max_manifold_points), int(args.seed))
    sampled_values = values_std[sample_indices]
    sampled_metadata = metadata.iloc[sample_indices].reset_index(drop=True)
    sampled_metadata["genre_top"] = top_n_with_other(sampled_metadata["main_genre"], int(args.genre_top_n))
    tsne_vals = tsne_points(
        sampled_values,
        int(args.seed),
        float(args.perplexity),
        int(args.tsne_max_iter),
    )
    save_point_table(
        sampled_metadata,
        tsne_vals,
        "tsne_x",
        "tsne_y",
        latent_dir / f"tsne_{args.embedding}_points.parquet",
    )
    scatter_by_category(
        tsne_vals,
        sampled_metadata,
        "cluster",
        f"{scope} {args.embedding} t-SNE colored by cluster",
        latent_dir / f"tsne_{args.embedding}_colored_by_cluster.png",
    )
    scatter_by_category(
        tsne_vals,
        sampled_metadata,
        "music_lang",
        f"{scope} {args.embedding} t-SNE colored by language",
        latent_dir / f"tsne_{args.embedding}_colored_by_language.png",
    )
    scatter_by_category(
        tsne_vals,
        sampled_metadata,
        "genre_top",
        f"{scope} {args.embedding} t-SNE colored by top genres",
        latent_dir / f"tsne_{args.embedding}_colored_by_genre_top{int(args.genre_top_n)}.png",
        show_legend=True,
    )

    language_counts, language_proportions = build_distribution_frames(metadata, "music_lang")
    write_dataframe(language_counts.reset_index(), distribution_dir / "cluster_language_counts.csv")
    write_dataframe(language_proportions.reset_index(), distribution_dir / "cluster_language_proportions.csv")
    plot_distribution_heatmap(
        language_proportions,
        f"{scope} cluster vs language proportions",
        distribution_dir / "cluster_language_heatmap.png",
    )
    plot_distribution_stacked_bar(
        language_counts,
        f"{scope} language mix per cluster",
        distribution_dir / "cluster_language_stacked_bar.png",
    )

    genre_counts, genre_proportions = build_distribution_frames(
        metadata,
        "main_genre",
        top_n=int(args.genre_top_n),
    )
    write_dataframe(genre_counts.reset_index(), distribution_dir / "cluster_genre_counts_top.csv")
    write_dataframe(genre_proportions.reset_index(), distribution_dir / "cluster_genre_proportions_top.csv")
    plot_distribution_heatmap(
        genre_proportions,
        f"{scope} cluster vs genre proportions (top genres + Other)",
        distribution_dir / "cluster_genre_heatmap_top.png",
    )
    plot_distribution_stacked_bar(
        genre_counts,
        f"{scope} genre mix per cluster (top genres + Other)",
        distribution_dir / "cluster_genre_stacked_bar_top.png",
    )

    metric_frame = metric_chart_frame(clustering_scope_dir)
    write_dataframe(metric_frame, metrics_dir / "metric_comparison_table.csv")
    plot_metric_comparison(metric_frame, metrics_dir / "metric_comparison.png")

    manifest = {
        "scope": scope,
        "embedding": args.embedding,
        "latent_scope_dir": str(latent_scope_dir),
        "clustering_scope_dir": str(clustering_scope_dir),
        "output_dir": str(output_dir),
        "assignment_column": assign_col,
        "row_count": int(len(metadata)),
        "cluster_count": int(metadata["cluster"].nunique()),
        "pca_explained_variance_ratio_sum": float(np.sum(pca_model.explained_variance_ratio_)),
        "tsne_plotted_rows": int(len(sampled_metadata)),
        "max_manifold_points": int(args.max_manifold_points),
        "genre_top_n": int(args.genre_top_n),
        "artifacts": [
            "latent_space/pca_*.png",
            "latent_space/tsne_*.png",
            "latent_space/*_points.parquet",
            "distributions/cluster_language_*.csv",
            "distributions/cluster_language_*.png",
            "distributions/cluster_genre_*.csv",
            "distributions/cluster_genre_*.png",
            "metrics/metric_comparison_table.csv",
            "metrics/metric_comparison.png",
        ],
    }
    save_json(manifest, output_dir / "visualization_manifest.json")
    log(f"Saved {scope} visualization outputs: {output_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    manifests = [run_scope(args, scope) for scope in scopes_from_arg(args.scope)]
    save_json(
        {
            "scope_count": len(manifests),
            "scopes": [manifest["scope"] for manifest in manifests],
            "output_root": str(project_path(args.output_root)),
            "embedding": args.embedding,
        },
        project_path(args.output_root) / "visualization_manifest.json",
    )
    log(f"Visualization complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
