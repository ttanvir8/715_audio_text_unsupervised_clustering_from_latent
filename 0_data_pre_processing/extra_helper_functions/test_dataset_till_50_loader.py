import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "1_training"))

from dataset_till_50 import create_dataloader_till_50  # noqa: E402


def print_tensor_summary(name: str, tensor: torch.Tensor) -> None:
    tensor_float = tensor.float()
    print(f"{name}:")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  min: {tensor_float.min().item():.6f}")
    print(f"  max: {tensor_float.max().item():.6f}")
    print(f"  mean: {tensor_float.mean().item():.6f}")
    print(f"  std: {tensor_float.std().item():.6f}")


def main() -> None:
    loader = create_dataloader_till_50(batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print("First 4 instances:")
    for index in range(4):
        print(f"instance {index}:")
        print(f"  music_id: {batch['music_id'][index]}")
        print(f"  music_name: {batch['music_name'][index]}")
        print(f"  art_id: {batch['art_id'][index]}")
        print(f"  art_name: {batch['art_name'][index]}")

    print("\nBatch tensor summaries:")
    for key in [
        "lyrics_e5_large_embedding",
        "melspectrogram",
        "main_genre_embedding",
        "music_lang_embedding",
        "genre_condition_embedding",
        "language_condition_embedding",
        "condition_embedding",
    ]:
        print_tensor_summary(key, batch[key])


if __name__ == "__main__":
    main()
