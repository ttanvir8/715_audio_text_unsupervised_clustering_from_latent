# Medium Audio+Lyrics VAE Visualize

This stage visualizes the medium audio+lyrics VAE clustering outputs in a
dedicated visualization directory, similar to `0_4_easy_visualize`.

## Run All

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_4_mid_visualize/visualize_src/run_mid_visualization_bundle.py \
  --experiment-dir 01_1_mid_training/experiments/audio_lyrics_vae_run_001 \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --clustering-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --output-root 01_4_mid_visualize/visualization_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu \
  --device auto
```

This runs:

- latent-space plots
- cluster distribution plots
- metric comparison charts
- VAE reconstruction examples
- zoomed music-name plots

## Outputs

Per scope, the bundle writes:

```text
visualizations/
|-- latent_space/
|-- distributions/
|-- metrics/
`-- visualization_manifest.json
reconstruction_examples/
|-- reconstruction_gallery.png
|-- selected_reconstruction_examples.csv
|-- per_example/
`-- reconstruction_manifest.json
```

The zoomed music-name outputs are written under:

```text
visualizations/latent_space/zoomed_music_names/
|-- tsne_mu_cluster_<id>_smallest_music_names.png
|-- tsne_mu_cluster_<id>_smallest_labeled_points.parquet
`-- tsne_mu_cluster_<id>_smallest_manifest.json
```

## Zoom Only

If the bundle has already created the latent-space point tables, you can rerun
only the zoomed music-name step:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_4_mid_visualize/visualize_src/run_mid_latent_zoom_music_names.py \
  --points-root 01_4_mid_visualize/visualization_outputs/audio_lyrics_vae_run_001 \
  --output-root 01_4_mid_visualize/visualization_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --method tsne \
  --embedding mu
```

Useful options:

```bash
--focus-cluster 0
--focus-language en
--music-name-contains "song title text"
--row-index 3304
--max-labels 80
```
