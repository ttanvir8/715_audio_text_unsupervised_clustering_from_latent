import argparse
import json
from pathlib import Path

import duckdb


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_dataset" / "toy_mel_extract"
DEFAULT_OUTPUT_FILE = "toy_mel_extract.parquet"
OUTPUT_COLUMNS = ["music_id", "music_name", "melspectrogram"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the first N raw melspectrogram rows into a toy dataset."
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / DEFAULT_OUTPUT_FILE
    query = f"""
        copy (
            select music_id, music_name, melspectrogram
            from read_parquet('{args.raw_dataset}')
            limit {args.limit}
        )
        to '{output_path}'
        (format parquet)
    """
    duckdb.sql(query)

    metadata = {
        "raw_dataset": str(args.raw_dataset),
        "output_file": DEFAULT_OUTPUT_FILE,
        "row_count": args.limit,
        "columns": OUTPUT_COLUMNS,
        "melspectrogram_structure": "list<list<double>>, expected tensor shape per row: [128, 1292]",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {args.limit} rows to {output_path}")


if __name__ == "__main__":
    main()
