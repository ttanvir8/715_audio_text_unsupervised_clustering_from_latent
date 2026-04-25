from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[mid_visualize_bundle] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full medium visualization bundle into 01_4_mid_visualize."
    )
    parser.add_argument(
        "--experiment-dir",
        default="01_1_mid_training/experiments/audio_lyrics_vae_run_001",
        help="Training experiment directory containing the best checkpoint.",
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
        default="01_4_mid_visualize/visualization_outputs/audio_lyrics_vae_run_001",
        help="Root where medium visualization artifacts will be written.",
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
    parser.add_argument("--device", default="auto", help="Device for reconstruction inference.")
    parser.add_argument(
        "--max-manifold-points",
        type=int,
        default=3000,
        help="Maximum rows used for t-SNE per scope.",
    )
    parser.add_argument(
        "--genre-top-n",
        type=int,
        default=12,
        help="How many genre labels to keep separate in genre plots.",
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=1,
        help="How many representative reconstructions to render per cluster.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Maximum number of clusters to render in the reconstruction step.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed.")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_command(cmd: list[str]) -> None:
    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cluster_viz_script = project_path(
        "01_3_mid_clustering/clustering_src/run_audio_lyrics_cluster_visualizations.py"
    )
    recon_script = project_path(
        "01_3_mid_clustering/clustering_src/run_audio_lyrics_reconstruction_examples.py"
    )
    zoom_script = project_path(
        "01_4_mid_visualize/visualize_src/run_mid_latent_zoom_music_names.py"
    )

    run_command(
        [
            sys.executable,
            str(cluster_viz_script),
            "--latent-root",
            str(project_path(args.latent_root)),
            "--clustering-root",
            str(project_path(args.clustering_root)),
            "--output-root",
            str(output_root),
            "--scope",
            args.scope,
            "--embedding",
            args.embedding,
            "--max-manifold-points",
            str(int(args.max_manifold_points)),
            "--genre-top-n",
            str(int(args.genre_top_n)),
            "--seed",
            str(int(args.seed)),
        ]
    )
    run_command(
        [
            sys.executable,
            str(recon_script),
            "--experiment-dir",
            str(project_path(args.experiment_dir)),
            "--latent-root",
            str(project_path(args.latent_root)),
            "--clustering-root",
            str(project_path(args.clustering_root)),
            "--output-root",
            str(output_root),
            "--scope",
            args.scope,
            "--device",
            args.device,
            "--examples-per-cluster",
            str(int(args.examples_per_cluster)),
            "--max-clusters",
            str(int(args.max_clusters)),
            "--seed",
            str(int(args.seed)),
        ]
    )
    run_command(
        [
            sys.executable,
            str(zoom_script),
            "--points-root",
            str(output_root),
            "--output-root",
            str(output_root),
            "--scope",
            args.scope,
            "--method",
            "tsne",
            "--embedding",
            args.embedding,
            "--seed",
            str(int(args.seed)),
        ]
    )

    save_json(
        {
            "output_root": str(output_root),
            "scope": args.scope,
            "embedding": args.embedding,
            "steps": [
                "run_audio_lyrics_cluster_visualizations.py",
                "run_audio_lyrics_reconstruction_examples.py",
                "run_mid_latent_zoom_music_names.py",
            ],
        },
        output_root / "visualization_bundle_manifest.json",
    )
    log(f"Medium visualization bundle complete: {output_root}")


if __name__ == "__main__":
    main()
