import argparse
import random
from pathlib import Path

import duckdb
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "0_data_pre_processing"
RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
MEL_TENSOR_DIR = DATA_ROOT / "processed_dataset" / "mel_tensors"
CHUNK_SIZE = 64
CHUNK_COUNT = 50


def quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def nested_shape(value: object) -> list[int]:
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def normalize_raw_list(raw_mel: list[list[float]]) -> list[list[float]]:
    target_bins = 128
    target_frames = 1292
    rows = [list(row[:target_frames]) for row in raw_mel[:target_bins]]
    for row in rows:
        if len(row) < target_frames:
            row.extend([0.0] * (target_frames - len(row)))
    while len(rows) < target_bins:
        rows.append([0.0] * target_frames)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw melspectrogram Python lists against .pt chunk Python lists."
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument(
        "--max-index",
        type=int,
        default=256,
        help="Randomly sample from [0, max-index). Keep this small because raw parquet mel decoding is slow.",
    )
    return parser.parse_args()


def load_raw_rows(indices: list[int]) -> dict[int, tuple[str, str, list[list[float]]]]:
    if not indices:
        return {}

    index_values = ", ".join(str(index) for index in sorted(indices))
    row_limit = max(indices) + 1
    query = f"""
        select __row_index, music_id, music_name, melspectrogram
        from (
            select
                row_number() over () - 1 as __row_index,
                music_id,
                music_name,
                melspectrogram
            from (
                select music_id, music_name, melspectrogram
                from read_parquet('{quote_path(RAW_DATASET)}')
                limit {row_limit}
            )
        )
        where __row_index in ({index_values})
        order by __row_index
    """
    rows = duckdb.sql(query).fetchall()
    return {
        row_index: (music_id, music_name, mel)
        for row_index, music_id, music_name, mel in rows
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    max_index = min(args.max_index, CHUNK_SIZE * CHUNK_COUNT)
    indices = rng.sample(range(max_index), k=min(args.count, max_index))
    raw_rows = load_raw_rows(indices)

    print(f"random_seed: {args.seed}")
    print(f"indices: {indices}")

    for index in indices:
        raw_music_id, raw_music_name, raw_mel_list = raw_rows[index]

        chunk_index = index // CHUNK_SIZE
        offset = index % CHUNK_SIZE
        chunk_path = MEL_TENSOR_DIR / f"chunk_{chunk_index:05d}.pt"
        pt_mel_list = torch.load(chunk_path, map_location="cpu", weights_only=True)[
            offset
        ].squeeze(0).tolist()

        normalized_raw_list = normalize_raw_list(raw_mel_list)
        raw_tensor = torch.tensor(normalized_raw_list, dtype=torch.float32)
        pt_tensor = torch.tensor(pt_mel_list, dtype=torch.float32)
        abs_diff = (raw_tensor - pt_tensor).abs()

        print(f"\nindex {index}:")
        print(f"  raw_music_id: {raw_music_id}")
        print(f"  raw_music_name: {raw_music_name}")
        print(f"  raw_python_type: {type(raw_mel_list).__name__}")
        print(f"  raw_nested_shape: {nested_shape(raw_mel_list)}")
        print(f"  pt_python_type_after_tolist: {type(pt_mel_list).__name__}")
        print(f"  pt_nested_shape_after_squeeze: {nested_shape(pt_mel_list)}")
        print(f"  raw_first_row_first_8: {normalized_raw_list[0][:8]}")
        print(f"  pt_first_row_first_8: {pt_mel_list[0][:8]}")
        print(f"  max_abs_diff_float16_expected: {abs_diff.max().item():.6f}")
        print(f"  mean_abs_diff_float16_expected: {abs_diff.mean().item():.6f}")
        print(f"  allclose_atol_rtol_1e-3: {torch.allclose(raw_tensor, pt_tensor, atol=1e-3, rtol=1e-3)}")


if __name__ == "__main__":
    main()
