# VAE Training Pipeline

This directory contains the VAE-only training pipeline for the `MM-CBetaVAE` described in `../advanced_architecture.md`.

The full run trains the model, writes checkpoints, saves train-only mel normalization stats, and exports validation latent embeddings for later clustering.

## 1. Activate The AI Env

Always use the project AI environment:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate
```

Check the core packages before installing anything new:

```bash
uv pip list | grep -E "torch|pyarrow|pandas|numpy|scikit-learn|PyYAML|tqdm"
```

The current pipeline uses packages already available in the AI env:

```text
torch
pyarrow
pandas
numpy
scikit-learn
PyYAML
tqdm
```

## 2. Run Preflight Checks

From the repository root:

```bash
python 1_training/dataset.py
```

Expected tensor shapes:

```text
lyrics_e5_large_embedding: [B, 1024]
melspectrogram: [B, 1, 128, 1292]
main_genre_embedding: [B, 32]
music_lang_embedding: [B, 8]
condition_embedding: [B, 24]
row_index: [B]
```

Run the VAE one-batch forward/backward sanity check:

```bash
python 1_training/sanity_check/check_vae_pipeline.py \
  --experiment-config 1_training/configs/experiments/run_001.yaml \
  --batch-size 2
```

Expected output includes:

```text
VAE sanity check passed.
mel_recon_norm: (2, 1, 128, 1292)
lyrics_recon: (2, 1024)
metadata_recon: (2, 40)
mu: (2, 64)
logvar: (2, 64)
```

## 3. Create A Full-Run Experiment Config

Do not reuse the smoke-test output directory for a real run. Create a new experiment YAML, for example:

```text
1_training/configs/experiments/run_full_001.yaml
```

Use this content:

```yaml
run_name: mm_cbeta_vae_full_001
data_config: 1_training/configs/data_base.yaml
vae_config: 1_training/configs/vae/mm_cbeta_vae_default.yaml
clustering_config: 1_training/configs/clustering/clustering_default.yaml
output_dir: 1_training/experiments/mm_cbeta_vae_full_001
```

The default VAE training config is:

```text
1_training/configs/vae/mm_cbeta_vae_default.yaml
```

Important defaults:

```yaml
training:
  epochs: 100
  batch_size: 8
  device: cuda
  mixed_precision: true
  early_stopping_patience: 15
```

CUDA is the default and recommended device. If CUDA memory allows it, use `batch_size: 16`. For a CPU-only debug run, explicitly set `training.device: cpu` in a temporary VAE config and keep `batch_size: 2` or `batch_size: 4`.

## 4. Optional Smoke Run

Before a full run, do a tiny end-to-end training smoke test:

```bash
python 1_training/training_scripts/train_vae.py \
  --experiment-config 1_training/configs/experiments/run_full_001.yaml \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 2 \
  --max-val-batches 1
```

This should create:

```text
1_training/experiments/mm_cbeta_vae_full_001/
|-- resolved_config.yaml
|-- run_manifest.yaml
|-- split_indices.json
|-- mel_normalization.json
|-- metrics.csv
|-- checkpoints_of_each_epochs/epoch_0001.pt
|-- best_checkpoint/model.pt
|-- best_checkpoint/model_config.json
`-- latent_exports/
    |-- val_mu.npy
    |-- val_metadata_assisted.npy
    |-- val_labels.parquet
    `-- latent_export_metadata.json
```

## 5. Run Full Training

For the proper full run, remove all smoke-only limits:

```bash
python 1_training/training_scripts/train_vae.py \
  --experiment-config 1_training/configs/experiments/run_full_001.yaml
```

Do not pass these flags in a full run:

```text
--max-train-batches
--max-val-batches
--max-stat-batches
```

Those flags are only for smoke tests. A full run must compute mel stats from the full train split, train on all train batches, validate on all validation batches, and export all validation latents.

The first long step is mel normalization:

```text
[train_vae] Computing mel normalization stats from full train split with a non-shuffled sequential loader
```

This scans every training mel tensor before model training starts. It should show a `mel stats` progress bar. On the full split this is about `1027` batches with the default `batch_size: 8`.

## 6. Monitor Training

Watch the metrics file:

```bash
tail -f 1_training/experiments/mm_cbeta_vae_full_001/metrics.csv
```

Each epoch writes train and validation rows with:

```text
loss
mel_recon_loss
lyrics_recon_loss
metadata_recon_loss
kl_loss
beta
```

Expected behavior:

- `beta` warms up toward `4.0`.
- Validation loss should become more stable after early epochs.
- KL should remain finite and nonzero because free-bits are enabled.
- The best checkpoint is selected by lowest validation total loss.

## 7. Verify Full-Run Artifacts

After training completes, verify the latent exports:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root = Path("1_training/experiments/mm_cbeta_vae_full_001")
mu = np.load(root / "latent_exports/val_mu.npy")
assisted = np.load(root / "latent_exports/val_metadata_assisted.npy")
labels = pq.read_table(root / "latent_exports/val_labels.parquet")

print("mu:", mu.shape)
print("assisted:", assisted.shape)
print("labels:", labels.num_rows, labels.column_names)

assert mu.shape[0] == labels.num_rows
assert assisted.shape[0] == labels.num_rows
assert mu.shape[1] == 64
assert assisted.shape[1] == 80
assert "row_index" in labels.column_names
PY
```

Expected full validation row count is about 15 percent of 9661 rows:

```text
1450 validation rows
```

Expected embedding shapes:

```text
val_mu.npy: [1450, 64]
val_metadata_assisted.npy: [1450, 80]
val_labels.parquet: 1450 rows
```

## 8. Output Meaning

Important files:

| File | Purpose |
| --- | --- |
| `resolved_config.yaml` | Fully expanded config used for this run |
| `run_manifest.yaml` | Links data, VAE, clustering config, and output dir |
| `split_indices.json` | Exact train/validation row split |
| `mel_normalization.json` | Train-only log-mel mean/std |
| `metrics.csv` | Per-epoch train/validation metrics |
| `checkpoints_of_each_epochs/epoch_*.pt` | Epoch checkpoints |
| `best_checkpoint/model.pt` | Best validation-loss checkpoint |
| `latent_exports/val_mu.npy` | Fair clustering embedding |
| `latent_exports/val_metadata_assisted.npy` | Metadata-assisted clustering embedding |
| `latent_exports/val_labels.parquet` | Row-aligned validation labels |

For fair clustering, use only:

```text
latent_exports/val_mu.npy
```

Use `val_metadata_assisted.npy` only for the separate metadata-assisted report.

## 9. Common Issues

### CUDA Is Not Visible

The default config uses `device: cuda`. If CUDA is not visible, training fails with a clear error instead of silently falling back to CPU. CPU is fine for smoke tests only if you explicitly set `training.device: cpu` in a temporary config.

Check device visibility:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

### Existing Experiment Directory

The trainer overwrites `metrics.csv` for the selected output directory and writes new checkpoints/exports. For real experiments, use a new `run_name` and `output_dir` instead of mixing smoke and full artifacts.

### Interrupted Training

Resume support is not implemented yet. If a full run is interrupted, start a new run directory or rerun the same experiment from epoch 1.
