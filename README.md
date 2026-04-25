# VAE-Based Unsupervised Clustering of Multilingual and Multi-genre Music Using Multimodal Features
The codebase for the project. It's the complete preprocessing, VAE training, latent inference, clustering, and visualization pipeline for hybrid-language music data.

## Repository Map

```text
0_data_pre_processing/   Raw-to-model-input preprocessing and embedding generation
0_1_easy_training/       Easy lyrics-only VAE training
0_2_easy_inference/      Easy checkpoint latent export
0_3_easy_clustering/     Easy latent K-Means and PCA baseline
0_4_easy_visualize/      Easy latent-space and cluster visualizations
01_1_mid_training/       Medium audio+lyrics VAE training
01_2_inference/          Medium checkpoint latent export
01_3_mid_clustering/     Medium clustering, baselines, and reconstruction examples
01_4_mid_visualize/      Medium visualization bundle
1_training/              Full multimodal conditional/beta VAE training
2_inference/             Full multimodal latent export
3_clustering/            Full multimodal clustering comparisons
4_Visulizations/         Full multimodal visualization scripts
```

## Reproducible Setup

Use `uv` to create the project environment on a new Linux or WSL device. The setup script creates `.venv`, installs pinned requirements, and can optionally install GPU extras and download the raw dataset.

Install `uv` first if the device does not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then run the setup from the repository root:

```bash
cd /path/to/this/repo/in/local/dir
bash scripts/setup_env.sh
source .venv/bin/activate
```

For CUDA/RAPIDS clustering support:

```bash
bash scripts/setup_env.sh --with-gpu
source .venv/bin/activate
```

For optional CPU UMAP support:

```bash
bash scripts/setup_env.sh --with-optional
source .venv/bin/activate
```

For a full new-device setup including the raw dataset:

```bash
bash scripts/setup_env.sh --with-gpu --with-optional --download-raw
source .venv/bin/activate
```

Useful script options:

```text
--venv-dir PATH       Virtual environment path. Default: ./.venv
--python VERSION      Python version for uv venv. Default: 3.12
--with-gpu            Install requirements-gpu.txt
--with-optional       Install requirements-optional.txt
--download-raw        Download the raw Zenodo parquet file
--skip-verify         Skip post-install import checks
```

## Requirement Files

```text
requirements.txt            Core pinned Python stack
requirements-gpu.txt        Optional pinned RAPIDS/cuML CUDA 12 stack
requirements-optional.txt   Optional CPU-side extras, currently umap-learn
```

The pinned core environment was written from the working Python `3.12.3` setup used for this project. GPU extras require Linux/WSL2, a supported NVIDIA GPU, and a compatible CUDA 12 stack. RAPIDS wheels are sensitive to Python and CUDA versions; see the official RAPIDS install guide at https://docs.rapids.ai/install/ if the pinned GPU file does not match a new device.

## Raw Dataset

The raw dataset comes from the OpenAIRE-linked DOI `10.5281/zenodo.4636802`, which resolves to the Zenodo record:

```text
https://zenodo.org/records/4636802
```

Zenodo lists the file as:

```text
4mula_small.parquet
size: 12.8 GB
md5: 30210cf6f52449c8d0670fc0942410c4
```

The project expects the file here:

```text
0_data_pre_processing/raw_dataset/4mula_small.parquet
```

Download it through the setup script:

```bash
bash scripts/setup_env.sh --download-raw
```

Or download only the file manually:

```bash
mkdir -p 0_data_pre_processing/raw_dataset
curl -L --fail --continue-at - \
  --output 0_data_pre_processing/raw_dataset/4mula_small.parquet \
  "https://zenodo.org/records/4636802/files/4mula_small.parquet?download=1"
md5sum 0_data_pre_processing/raw_dataset/4mula_small.parquet
```

The checksum should be:

```text
30210cf6f52449c8d0670fc0942410c4
```

## Verify The Install

Run this from the repo root after activating the environment:

```bash
python - <<'PY'
import duckdb
import librosa
import matplotlib
import numpy
import pandas
import polars
import pyarrow
import sklearn
import torch
import transformers
import yaml

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("core imports ok")
PY
```

For optional cuML:

```bash
python - <<'PY'
import cuml
print("cuml:", cuml.__version__)
PY
```

## Hugging Face Model Setup

Most generated datasets and checkpoints are already present in this repo. If you need to regenerate E5 lyrics embeddings, first activate the uv environment, then download the model locally where the preprocessing script expects it:

```bash
cd 0_data_pre_processing
hf download intfloat/multilingual-e5-large --local-dir ./embedding_models/intfloat/multilingual-e5-large
cd ..
```

The lyrics preprocessing script loads this model with `local_files_only=True`.

## Common Commands

Run commands from the repository root unless a command explicitly changes directory.
Activate the AI environment first:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate
uv pip list | grep -E "torch|pyarrow|pandas|numpy|scikit-learn|PyYAML|tqdm|transformers"
```

### `0_data_pre_processing/`

Raw-to-model-input preprocessing and embedding generation.

Download the local E5 model before regenerating lyrics embeddings:

```bash
cd 0_data_pre_processing
hf download intfloat/multilingual-e5-large --local-dir ./embedding_models/intfloat/multilingual-e5-large
cd ..
```

Generate lyrics, genre/language, and mel tensor inputs:

```bash
python 0_data_pre_processing/src/preprocess_lyrics_e5_large.py --device auto
python 0_data_pre_processing/src/preprocess_genre_language.py
python 0_data_pre_processing/src/preprocess_mels_to_tensor_chunks.py --resume
```

Build the compact row-aligned model input parquet:

```bash
python 0_data_pre_processing/src/01_create_model_input_base_chunks.py
python 0_data_pre_processing/src/02_add_lyrics_to_model_input_chunks.py
python 0_data_pre_processing/src/03_add_genre_language_to_model_input_chunks.py
python 0_data_pre_processing/src/04_compact_model_input_chunks.py
```

Optional raw-mel parquet join:

```bash
python 0_data_pre_processing/src/add_mels_to_model_input_with_duckdb.py \
  --memory-limit 8GB \
  --threads 1
```

### `0_1_easy_training/`

Easy lyrics-only VAE training.

```bash
python 0_1_easy_training/sanity_check/check_lyrics_vae_pipeline.py \
  --experiment-config 0_1_easy_training/configs/experiments/run_001.yaml \
  --batch-size 4 \
  --device cpu

python 0_1_easy_training/training_scripts/train_vae.py \
  --experiment-config 0_1_easy_training/configs/experiments/run_001.yaml
```

### `0_2_easy_inference/`

Easy checkpoint latent export.

```bash
python 0_2_easy_inference/inference_src/export_best_checkpoint_latents.py \
  --experiment-dir 0_1_easy_training/experiments/lyrics_vae_run_001 \
  --scope both \
  --device auto \
  --overwrite
```

### `0_3_easy_clustering/`

Easy latent K-Means and PCA baseline.

```bash
python 0_3_easy_clustering/clustering_src/run_lyrics_latent_clustering.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --output-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
  --scope both \
  --embedding mu \
  --make-plot

python 0_3_easy_clustering/clustering_src/compare_pca_kmeans_baseline.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --output-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
  --scope both \
  --embedding mu \
  --pca-components 64
```

### `0_4_easy_visualize/`

Easy latent-space and cluster visualizations.

```bash
python 0_4_easy_visualize/visualize_src/run_easy_cluster_manifold.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --clustering-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
  --output-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --scope both \
  --method tsne \
  --embedding mu

python 0_4_easy_visualize/visualize_src/run_easy_latent_space.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --clustering-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
  --output-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --scope both

python 0_4_easy_visualize/visualize_src/run_easy_latent_zoom_music_names.py \
  --points-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --output-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --scope both \
  --method tsne \
  --embedding mu
```

### `01_1_mid_training/`

Medium audio+lyrics VAE training.

```bash
python 01_1_mid_training/sanity_check/check_audio_lyrics_vae_pipeline.py \
  --experiment-config 01_1_mid_training/configs/experiments/run_001.yaml \
  --batch-size 2 \
  --device cpu

python 01_1_mid_training/training_scripts/train_vae.py \
  --experiment-config 01_1_mid_training/configs/experiments/run_001.yaml
```

### `01_2_inference/`

Medium checkpoint latent export.

```bash
python 01_2_inference/inference_src/export_best_checkpoint_latents.py \
  --experiment-dir 01_1_mid_training/experiments/audio_lyrics_vae_run_001 \
  --scope both \
  --device auto \
  --overwrite
```

### `01_3_mid_clustering/`

Medium clustering, baselines, and reconstruction examples.

```bash
python 01_3_mid_clustering/clustering_src/run_audio_lyrics_latent_clustering.py \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu \
  --make-plot

python 01_3_mid_clustering/clustering_src/compare_audio_lyrics_kmeans_baselines.py \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu

python 01_3_mid_clustering/clustering_src/compare_audio_lyrics_clustering_algorithms.py \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu

python 01_3_mid_clustering/clustering_src/run_audio_lyrics_reconstruction_examples.py \
  --experiment-dir 01_1_mid_training/experiments/audio_lyrics_vae_run_001 \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --clustering-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --output-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --device auto
```

### `01_4_mid_visualize/`

Medium visualization bundle.

```bash
python 01_4_mid_visualize/visualize_src/run_mid_visualization_bundle.py \
  --experiment-dir 01_1_mid_training/experiments/audio_lyrics_vae_run_001 \
  --latent-root 01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001 \
  --clustering-root 01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001 \
  --output-root 01_4_mid_visualize/visualization_outputs/audio_lyrics_vae_run_001 \
  --scope both \
  --embedding mu \
  --device auto
```

### `1_training/`

Full multimodal conditional/beta VAE training.

```bash
python 1_training/sanity_check/check_vae_pipeline.py \
  --experiment-config 1_training/configs/experiments/run_001.yaml \
  --batch-size 2

python 1_training/training_scripts/train_vae.py \
  --experiment-config 1_training/configs/experiments/run_full_001.yaml
```

### `2_inference/`

Full multimodal latent export.

```bash
python 2_inference/inference_src/export_best_checkpoint_latents.py \
  --experiment-dir 1_training/experiments/mm_cbeta_vae_experiment_002 \
  --scope both \
  --device cuda \
  --batch-size 48
```

### `3_clustering/`

Full multimodal clustering comparisons.

```bash
python 3_clustering/clustering_src/run_multimodal_clustering.py \
  --latent-root 2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002 \
  --output-root 3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002 \
  --scope both \
  --comparison-methods pca_kmeans,autoencoder_kmeans,spectral_feature_kmeans,vae_mu_kmeans,vae_metadata_assisted_kmeans \
  --comparison-kmeans-algorithm cuml_kmeans
```

Use CPU/scikit-learn K-Means instead of cuML when RAPIDS is not installed:

```bash
python 3_clustering/clustering_src/run_multimodal_clustering.py \
  --latent-root 2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002 \
  --output-root 3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002 \
  --scope both \
  --comparison-methods pca_kmeans,autoencoder_kmeans,spectral_feature_kmeans,vae_mu_kmeans,vae_metadata_assisted_kmeans \
  --comparison-kmeans-algorithm kmeans
```

### `4_Visulizations/`

Full multimodal visualization scripts.

```bash
python 4_Visulizations/visualize_src/run_visualizations.py \
  --experiment-name mm_cbeta_vae_experiment_002 \
  --latent-root 2_inference/latents_by_best_checkpoints \
  --clustering-root 3_clustering/clustering_outputs \
  --output-root 4_Visulizations/cluster_visulization_outputs \
  --scope both
```

Shortcut shell runner:

```bash
bash 4_Visulizations/run_all_visualizations.sh
```

More detailed stage-specific notes live in the README files inside each pipeline directory.

## Notes

- `scripts/setup_env.sh` is the preferred setup path for a new device.
- `--download-raw` downloads a large 12.8 GB file, so run it only when the device has enough disk space and a stable connection.
- The training configs are in each stage's `configs/` directory. Experiment outputs are written under each stage's `experiments/`, `latents_by_best_checkpoints/`, `clustering_outputs/`, or `visualization_outputs/` directory.
