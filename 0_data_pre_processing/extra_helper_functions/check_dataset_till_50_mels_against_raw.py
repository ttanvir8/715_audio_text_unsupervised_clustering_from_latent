import sys
import argparse
import random
from pathlib import Path

import duckdb
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "0_data_pre_processing"
RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
sys.path.append(str(PROJECT_ROOT / "1_training"))

from dataset_till_50 import MusicVAEDatasetTill50  # noqa: E402


def quote(value: str) -> str:
    return value.replace("'", "''")


def normalize_raw_mel(mel: object, target_shape: tuple[int, int, int]) -> torch.Tensor:
    raw = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
    _, target_bins, target_frames = target_shape
    _, bins, frames = raw.shape

    if bins < target_bins:
        raw = torch.nn.functional.pad(raw, (0, 0, 0, target_bins - bins))
    elif bins > target_bins:
        raw = raw[:, :target_bins, :]

    if frames < target_frames:
        raw = torch.nn.functional.pad(raw, (0, target_frames - frames))
    elif frames > target_frames:
        raw = raw[:, :, :target_frames]

    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random dataset_till_50 mels against raw parquet mels fetched with DuckDB."
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MusicVAEDatasetTill50()
    con = duckdb.connect()
    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(len(dataset)), k=min(args.count, len(dataset)))
    samples = [(index, dataset[index]) for index in sample_indices]

    print(f"random_seed: {args.seed}")
    print(f"dataset_indices: {sample_indices}")
    for index, sample in samples:
        music_id = sample["music_id"]
        mel_from_loader = sample["melspectrogram"].float()
        raw_music_id, raw_music_name, raw_mel = con.execute(
            f"""
            select music_id, music_name, melspectrogram
            from read_parquet('{RAW_DATASET}')
            limit 1
            offset {index}
            """
        ).fetchone()
        raw_tensor = normalize_raw_mel(raw_mel, tuple(mel_from_loader.shape))
        abs_diff = (mel_from_loader - raw_tensor).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        is_match = torch.allclose(mel_from_loader, raw_tensor, atol=1e-3, rtol=1e-3)

        print(f"dataset_index {index}:")
        print(f"  loader_music_id: {music_id}")
        print(f"  loader_music_name: {sample['music_name']}")
        print(f"  raw_music_id: {raw_music_id}")
        print(f"  raw_music_name: {raw_music_name}")
        print(f"  music_id_match: {music_id == raw_music_id}")
        print(f"  max_abs_diff: {max_diff:.6f}")
        print(f"  mean_abs_diff: {mean_diff:.6f}")
        print(f"  allclose_atol_rtol_1e-3: {is_match}")


if __name__ == "__main__":
    main()
