# Medium Audio+Lyrics VAE Clustering

This stage clusters `01_2_inference` encoder latents and compares them against
audio+lyrics baselines on the same validation/full rows.

## Latent K-Means

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_3_mid_clustering/clustering_src/run_audio_lyrics_latent_clustering.py \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu \
  --make-plot
```

Outputs per scope:

```text
cluster_assignments.parquet
metrics.csv
silhouette_sweep.csv
cluster_summary_by_language.csv
cluster_summary_by_genre.csv
run_metadata.json
latent_pca_cluster_scatter.png
```

The default clustering target is `music_lang`, so the main run uses `k` equal to
the number of observed language labels unless `--cluster-k` is set.

## Baseline Comparison

Compare the VAE latent K-Means run against three audio+lyrics baselines on the
same rows:

- PCA over concatenated lyrics embeddings and compact spectral features
- Autoencoder over concatenated lyrics embeddings and compact spectral features
- Direct compact spectral feature clustering

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_3_mid_clustering/clustering_src/compare_audio_lyrics_kmeans_baselines.py \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu
```

Outputs per scope:

```text
baseline_kmeans_comparison.csv
baseline_kmeans_assignments.parquet
baseline_kmeans_metadata.json
```

The comparison table reports Silhouette Score, Calinski-Harabasz Index, NMI, ARI,
Davies-Bouldin Index, and Purity for the VAE view and the audio+lyrics baselines.

For the current `audio_lyrics_vae_run_001` results, the main pattern is:

- `vae_mu_kmeans` is best on Silhouette and Calinski-Harabasz, so its latent space is the most compact geometrically.
- `pca_audio_lyrics_kmeans` and `autoencoder_audio_lyrics_kmeans` are much better on NMI, ARI, and Purity, so they align better with the language labels.
- `spectral_feature_kmeans` stays weak on label agreement, which suggests mel-only compact features are not enough for language recovery here.

A dedicated comparison note is available at:

```text
01_3_mid_clustering/method_comparison_analysis.md
```

## Clustering Algorithm Comparison

Compare three clustering algorithms directly on the standardized VAE latent
representation:

- K-Means
- Agglomerative Clustering with Ward linkage
- DBSCAN with a small parameter sweep

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_3_mid_clustering/clustering_src/compare_audio_lyrics_clustering_algorithms.py \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu
```

Outputs per scope:

```text
clustering_algorithm_comparison.csv
clustering_algorithm_assignments.parquet
dbscan_parameter_sweep.csv
clustering_algorithm_metadata.json
```

The algorithm comparison reports Silhouette Score, Davies-Bouldin Index, NMI,
ARI, and Purity, along with the number of clusters and DBSCAN noise fraction.

## Detailed Visualizations

Create latent-space plots, cluster distribution plots, and a metric comparison
chart directly from the clustering outputs:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_3_mid_clustering/clustering_src/run_audio_lyrics_cluster_visualizations.py \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --clustering-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu
```

Per scope, the script writes visualization artifacts under:

```text
visualizations/
|-- latent_space/
|-- distributions/
|-- metrics/
`-- visualization_manifest.json
```

## Reconstruction Examples

Render VAE reconstruction examples from cluster-representative latent vectors:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_3_mid_clustering/clustering_src/run_audio_lyrics_reconstruction_examples.py \
  --experiment-dir 01_1_mid_training/experiments/audio_lyrics_vae_run_001 \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --clustering-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --device auto
```

Per scope, the script writes reconstruction artifacts under:

```text
reconstruction_examples/
|-- reconstruction_gallery.png
|-- selected_reconstruction_examples.csv
|-- per_example/
`-- reconstruction_manifest.json
```
