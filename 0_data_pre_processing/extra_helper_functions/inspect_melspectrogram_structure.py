import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"


def nested_shape(value: Any) -> list[int | str]:
    shape: list[int | str] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        lengths = [len(item) for item in current if isinstance(item, list)]
        if lengths and len(set(lengths)) > 1:
            shape.append(f"ragged:{min(lengths)}..{max(lengths)}")
            break
        current = current[0]
    return shape


def numeric_summary(value: Any) -> dict[str, float | int | str]:
    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "dtype": str(array.dtype),
        "ndim": int(array.ndim),
        "size": int(array.size),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "nan_count": int(np.isnan(array).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw melspectrogram structure.")
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--rows", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Dataset: {args.raw_dataset}")
    query = f"""
        select music_id, music_name, melspectrogram
        from read_parquet('{args.raw_dataset}')
        limit {args.rows}
    """
    rows = duckdb.sql(query).fetchall()
    for index, row in enumerate(rows):
        music_id, music_name, mel = row
        tensor = torch.tensor(mel, dtype=torch.float32)
        report = {
            "row_index": index,
            "music_id": music_id,
            "music_name": music_name,
            "python_type": type(mel).__name__,
            "nested_shape": nested_shape(mel),
            "torch_tensor_shape": list(tensor.shape),
            "torch_tensor_dtype": str(tensor.dtype),
            "numeric_summary": numeric_summary(mel),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
