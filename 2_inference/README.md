# Encoder Latent Inference

This stage loads a trained experiment checkpoint and exports encoder latents for downstream clustering.

## Experiment 2

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 2_inference/inference_src/export_best_checkpoint_latents.py \
  --experiment-dir 1_training/experiments/mm_cbeta_vae_experiment_002 \
  --scope both \
  --device cuda \
  --batch-size 48
```

Outputs are written to:

```text
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/
|-- validation/
`-- full/
```

Each scope contains `mu.npy`, `metadata_assisted.npy`, `labels.parquet`, `row_indices.npy`, and `latent_export_metadata.json`.
