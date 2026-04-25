import ast
import json
import os
import re

import polars as pl


PARQUET_PATH = "raw_dataset/4mula_small.parquet"
TARGET_COLUMN = "musicnn_tags"


def normalize_tags(value):
    if value is None:
        return []

    if isinstance(value, list):
        tags = value
    elif isinstance(value, tuple):
        tags = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        parsed = None
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
                break
            except Exception:
                pass

        if isinstance(parsed, (list, tuple, set)):
            tags = list(parsed)
        elif isinstance(parsed, str):
            tags = [parsed]
        else:
            tags = re.split(r"[,\|;/]+", text)
    else:
        tags = [value]

    cleaned = []
    for tag in tags:
        if tag is None:
            continue
        tag_text = str(tag).strip().strip("'\"")
        if tag_text:
            cleaned.append(tag_text)
    return cleaned


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"Error: {PARQUET_PATH} not found.")
        return

    df = pl.scan_parquet(PARQUET_PATH).select(TARGET_COLUMN).collect()

    unique_tags = set()
    for value in df[TARGET_COLUMN].to_list():
        unique_tags.update(normalize_tags(value))

    sorted_tags = sorted(unique_tags)

    print(f"Unique tags in {TARGET_COLUMN}: {len(sorted_tags)}")
    for tag in sorted_tags:
        print(tag)


if __name__ == "__main__":
    main()
