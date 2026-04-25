# Easy Lyrics VAE Clustering

This stage clusters `0_2_easy_inference` encoder latents and evaluates the clusters against `music_lang`.

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 0_3_easy_clustering/clustering_src/run_lyrics_latent_clustering.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --output-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
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
run_metadata.json
latent_pca_cluster_scatter.png
```

The main clustering run uses `k` equal to the number of language labels unless `--language-k` is set.

## PCA + K-Means Baseline

Compare the VAE latent K-Means run against PCA over the original lyrics embeddings followed by K-Means:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 0_3_easy_clustering/clustering_src/compare_pca_kmeans_baseline.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --output-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
  --scope both \
  --embedding mu \
  --pca-components 64
```

Outputs per scope:

```text
pca_kmeans_baseline_comparison.csv
pca_kmeans_baseline_assignments.parquet
pca_kmeans_baseline_metadata.json
```

The comparison table reports Silhouette Score and Calinski-Harabasz Index for `vae_mu_kmeans` and `pca_lyrics_kmeans` on the same rows.
