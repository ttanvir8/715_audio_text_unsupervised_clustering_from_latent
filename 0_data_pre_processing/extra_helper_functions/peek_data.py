import polars as pl
try:
    df = pl.read_parquet("raw_dataset/4mula_small.parquet", n_rows=5)
    print("Columns:", df.columns)
    print("Schema:", df.schema)
    first_row = df.head(1)
    print("First row:")
    columns = [
        "music_id",
        "music_name",
        "music_lang",
        "music_lyrics",
        "art_id",
        "art_name",
        "art_rank",
        "main_genre",
        "related_genre",
        "related_art",
        "related_music",
        "musicnn_tags",
        "melspectrogram",
    ]
    for col in columns:
        value = first_row.get_column(col)[0] if col in first_row.columns else None
        print(f"{col}: {value}")
    
    # Let's get total row count without loading memory
    print("Total rows (approx/exact via lazy):", pl.scan_parquet("raw_dataset/4mula_small.parquet").select(pl.len()).collect().item())
except Exception as e:
    print("Error:", e)
