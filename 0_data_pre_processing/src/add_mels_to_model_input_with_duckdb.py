import argparse
import json
from pathlib import Path

import duckdb


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
DEFAULT_MODEL_INPUT = DATA_ROOT / "processed_dataset" / "model_input" / "model_input_dataset.parquet"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_dataset" / "model_input_with_mels"
DEFAULT_OUTPUT_FILE = "model_input_with_mels.parquet"

OUTPUT_COLUMNS = [
    "music_id",
    "music_name",
    "lyrics_e5_large_embedding",
    "melspectrogram",
    "main_genre_embedding",
    "music_lang_embedding",
    "genre_condition_embedding",
    "language_condition_embedding",
    "condition_embedding",
    "art_id",
    "art_name",
]


def quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use DuckDB to add raw melspectrogram values to the model input parquet."
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--model-input", type=Path, default=DEFAULT_MODEL_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_file
    temp_dir = args.output_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"set memory_limit='{args.memory_limit}'")
    con.execute(f"set temp_directory='{quote_path(temp_dir)}'")
    con.execute(f"set threads={args.threads}")
    con.execute("set preserve_insertion_order=false")

    limit_sql = f"limit {args.limit}" if args.limit is not None else ""
    query = f"""
        copy (
            with
            base as (
                select row_number() over () - 1 as __row_index, *
                from read_parquet('{quote_path(args.model_input)}')
                {limit_sql}
            ),
            raw_mels as (
                select row_number() over () - 1 as __row_index, melspectrogram
                from read_parquet('{quote_path(args.raw_dataset)}')
                {limit_sql}
            )
            select
                base.music_id,
                base.music_name,
                base.lyrics_e5_large_embedding,
                raw_mels.melspectrogram,
                base.main_genre_embedding,
                base.music_lang_embedding,
                base.genre_condition_embedding,
                base.language_condition_embedding,
                base.condition_embedding,
                base.art_id,
                base.art_name
            from base
            join raw_mels using (__row_index)
            order by base.__row_index
        )
        to '{quote_path(output_path)}'
        (format parquet, compression zstd)
    """
    con.execute(query)

    row_count = con.execute(
        f"select count(*) from read_parquet('{quote_path(output_path)}')"
    ).fetchone()[0]
    metadata = {
        "raw_dataset": str(args.raw_dataset),
        "model_input": str(args.model_input),
        "output_file": args.output_file,
        "row_count": row_count,
        "limit": args.limit,
        "memory_limit": args.memory_limit,
        "threads": args.threads,
        "preserve_insertion_order": False,
        "columns": OUTPUT_COLUMNS,
        "join_key": "row position via DuckDB row_number(), not music_id",
        "melspectrogram_structure": "list<list<double>>, torch tensor shape per row expected [128, 1292]",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()
