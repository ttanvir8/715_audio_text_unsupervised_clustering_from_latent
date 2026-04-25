from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[mid_latent_zoom_names] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create zoomed medium latent-space plots with music_name labels."
    )
    parser.add_argument(
        "--points-root",
        default="01_4_mid_visualize/visualization_outputs/audio_lyrics_vae_run_001",
        help="Root containing scope/visualizations/latent_space point tables.",
    )
    parser.add_argument(
        "--output-root",
        default="01_4_mid_visualize/visualization_outputs/audio_lyrics_vae_run_001",
        help="Root where zoomed plots will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "val", "full", "both"),
        default="both",
        help="Which scope to visualize. val is an alias for validation.",
    )
    parser.add_argument("--method", choices=("pca", "tsne"), default="tsne")
    parser.add_argument("--embedding", choices=("mu", "logvar"), default="mu")
    parser.add_argument(
        "--focus-cluster",
        default="auto-smallest",
        help=(
            "Cluster to zoom into. Use auto-smallest, all, or an integer cluster id. "
            "Ignored when --music-name-contains or --row-index is used."
        ),
    )
    parser.add_argument(
        "--focus-language",
        default=None,
        help="Optional language code to zoom into, for example en, es, pt, or pt-br.",
    )
    parser.add_argument(
        "--music-name-contains",
        default=None,
        help="Zoom around rows whose music_name contains this case-insensitive text.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Zoom around the nearest latent neighbors of this row_index.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=60,
        help="Nearest neighbors to include when zooming around a row or music-name match.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=60,
        help="Maximum music_name labels drawn on one plot.",
    )
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.18,
        help="Extra axis padding around the focused latent region.",
    )
    parser.add_argument("--label-font-size", type=float, default=7.0)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    if scope == "val":
        return ["validation"]
    return [scope]


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path)


def load_points(points_root: Path, scope: str, method: str, embedding: str) -> pd.DataFrame:
    path = points_root / scope / "visualizations" / "latent_space" / f"{method}_{embedding}_points.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing point table: {path}. Run the mid visualization bundle first."
        )
    points = pq.read_table(path).to_pandas()
    x_col = f"{method}_x"
    y_col = f"{method}_y"
    required = {"row_index", "music_name", "music_lang", "cluster", x_col, y_col}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return points


def nearest_rows(points: pd.DataFrame, center_index: int, x_col: str, y_col: str, count: int) -> pd.DataFrame:
    center = points.loc[points["row_index"] == center_index]
    if center.empty:
        raise ValueError(f"row_index={center_index} is not present in the point table.")
    center_x = float(center.iloc[0][x_col])
    center_y = float(center.iloc[0][y_col])
    distances = (points[x_col] - center_x) ** 2 + (points[y_col] - center_y) ** 2
    return points.assign(_distance=distances).nsmallest(max(1, count), "_distance").drop(columns="_distance")


def slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "value"


def choose_focus_rows(points: pd.DataFrame, args: argparse.Namespace, x_col: str, y_col: str) -> tuple[pd.DataFrame, str]:
    focus = points
    name_parts: list[str] = []

    if args.row_index is not None:
        return nearest_rows(points, int(args.row_index), x_col, y_col, int(args.neighbors)), (
            f"row_index_{int(args.row_index)}_neighbors"
        )

    if args.music_name_contains:
        mask = points["music_name"].astype(str).str.contains(
            str(args.music_name_contains),
            case=False,
            na=False,
            regex=False,
        )
        matches = points.loc[mask]
        if matches.empty:
            raise ValueError(f"No music_name contains: {args.music_name_contains!r}")
        center = matches.iloc[0]
        neighbors = nearest_rows(points, int(center["row_index"]), x_col, y_col, int(args.neighbors))
        return neighbors, f"name_{slug(str(args.music_name_contains))}_neighbors"

    if args.focus_language:
        focus = focus.loc[focus["music_lang"].astype(str) == str(args.focus_language)]
        name_parts.append(f"language_{slug(str(args.focus_language))}")
        if focus.empty:
            raise ValueError(f"No rows found for language: {args.focus_language}")

    cluster_arg = str(args.focus_cluster)
    if cluster_arg == "auto-smallest":
        counts = focus["cluster"].value_counts().sort_values(kind="stable")
        cluster_id = counts.index[0]
        focus = focus.loc[focus["cluster"] == cluster_id]
        name_parts.append(f"cluster_{cluster_id}_smallest")
    elif cluster_arg != "all":
        cluster_id = int(cluster_arg)
        focus = focus.loc[focus["cluster"] == cluster_id]
        name_parts.append(f"cluster_{cluster_id}")
        if focus.empty:
            raise ValueError(f"No rows found for cluster: {cluster_id}")
    else:
        name_parts.append("all")

    return focus, "_".join(name_parts)


def sample_label_rows(focus: pd.DataFrame, max_labels: int, seed: int) -> pd.DataFrame:
    if max_labels <= 0 or len(focus) <= max_labels:
        return focus.copy()
    return focus.sample(n=max_labels, random_state=seed).sort_values(["cluster", "row_index"])


def axis_limits(
    focus: pd.DataFrame,
    x_col: str,
    y_col: str,
    padding_fraction: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(focus[x_col].min())
    x_max = float(focus[x_col].max())
    y_min = float(focus[y_col].min())
    y_max = float(focus[y_col].max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    x_pad = x_span * padding_fraction
    y_pad = y_span * padding_fraction
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def truncate_label(value: Any, max_chars: int = 34) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= max_chars else f"{text[: max_chars - 3]}..."


def category_colors(values: pd.Series) -> dict[str, Any]:
    names = sorted(values.astype(str).unique().tolist())
    cmap = plt.get_cmap("tab10")
    return {name: cmap(index % cmap.N) for index, name in enumerate(names)}


def plot_zoom(
    points: pd.DataFrame,
    focus: pd.DataFrame,
    labels: pd.DataFrame,
    scope: str,
    method: str,
    embedding: str,
    focus_name: str,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    x_col = f"{method}_x"
    y_col = f"{method}_y"
    colors = category_colors(points["cluster"])
    xlim, ylim = axis_limits(focus, x_col, y_col, float(args.padding_fraction))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        points[x_col],
        points[y_col],
        s=10,
        color="#c7cbd1",
        alpha=0.28,
        edgecolors="none",
        label="context",
    )

    for cluster, group in focus.groupby("cluster", sort=True):
        cluster_name = str(cluster)
        ax.scatter(
            group[x_col],
            group[y_col],
            s=42,
            color=colors.get(cluster_name, "#4c78a8"),
            alpha=0.92,
            edgecolors="white",
            linewidths=0.45,
            label=f"cluster {cluster_name}",
        )

    label_effects = [path_effects.withStroke(linewidth=2.5, foreground="white")]
    for _, row in labels.iterrows():
        label = truncate_label(row["music_name"])
        ax.scatter(
            [row[x_col]],
            [row[y_col]],
            s=58,
            color=colors.get(str(row["cluster"]), "#4c78a8"),
            edgecolors="#111111",
            linewidths=0.5,
            zorder=4,
        )
        ax.annotate(
            label,
            (float(row[x_col]), float(row[y_col])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=float(args.label_font_size),
            color="#111111",
            path_effects=label_effects,
            zorder=5,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"{scope} {method.upper()} {embedding} zoom: {focus_name.replace('_', ' ')}")
    ax.set_xlabel(f"{method} dimension 1")
    ax.set_ylabel(f"{method} dimension 2")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def run_scope(args: argparse.Namespace, scope: str) -> dict[str, Any]:
    points_root = project_path(args.points_root)
    output_root = project_path(args.output_root)
    points = load_points(points_root, scope, args.method, args.embedding)
    x_col = f"{args.method}_x"
    y_col = f"{args.method}_y"
    focus, focus_name = choose_focus_rows(points, args, x_col, y_col)
    label_rows = sample_label_rows(focus, int(args.max_labels), int(args.seed))

    output_dir = output_root / scope / "visualizations" / "latent_space" / "zoomed_music_names"
    plot_path = output_dir / f"{args.method}_{args.embedding}_{focus_name}_music_names.png"
    labels_path = output_dir / f"{args.method}_{args.embedding}_{focus_name}_labeled_points.parquet"
    manifest_path = output_dir / f"{args.method}_{args.embedding}_{focus_name}_manifest.json"

    plot_zoom(
        points=points,
        focus=focus,
        labels=label_rows,
        scope=scope,
        method=args.method,
        embedding=args.embedding,
        focus_name=focus_name,
        output_path=plot_path,
        args=args,
    )
    write_parquet(label_rows, labels_path)
    manifest = {
        "scope": scope,
        "method": args.method,
        "embedding": args.embedding,
        "focus_name": focus_name,
        "plotted_rows": int(len(focus)),
        "labeled_rows": int(len(label_rows)),
        "plot_path": str(plot_path),
        "labeled_points_path": str(labels_path),
    }
    save_json(manifest, manifest_path)
    log(f"Saved {scope} zoomed music-name plot: {plot_path}")
    return manifest


def main() -> None:
    args = parse_args()
    manifests = [run_scope(args, scope) for scope in scopes_from_arg(args.scope)]
    save_json(
        {
            "scope_count": len(manifests),
            "scopes": [manifest["scope"] for manifest in manifests],
            "output_root": str(project_path(args.output_root)),
            "method": args.method,
            "embedding": args.embedding,
        },
        project_path(args.output_root) / "zoomed_music_names_manifest.json",
    )
    log(f"Zoomed music-name visualization complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
