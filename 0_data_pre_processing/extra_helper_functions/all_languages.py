import os

import polars as pl


PARQUET_PATH = "raw_dataset/4mula_small.parquet"
TARGET_COLUMN = "music_lang"


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"Error: {PARQUET_PATH} not found.")
        return

    df = pl.scan_parquet(PARQUET_PATH).select(TARGET_COLUMN).collect()

    lang_counts = (
        df.group_by(TARGET_COLUMN)
        .len()
        .sort("len", descending=True)
    )

    print(f"Unique languages in {TARGET_COLUMN}: {lang_counts.height}")
    for row in lang_counts.iter_rows(named=True):
        print(f"{row[TARGET_COLUMN]}: {row['len']}")


if __name__ == "__main__":
    main()
