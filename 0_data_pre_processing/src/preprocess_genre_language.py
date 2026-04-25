import argparse
import json
import logging
from pathlib import Path

import polars as pl
import torch
from torch import nn


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_dataset" / "genre_language"
GENRE_METADATA_DIM = 32
LANGUAGE_METADATA_DIM = 8
GENRE_CONDITION_DIM = 16
LANGUAGE_CONDITION_DIM = 8
LOGGER = logging.getLogger(__name__)


def normalize_category(value: object) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def build_mapping(values: list[str]) -> dict[str, int]:
    unique_values = sorted(set(values))
    if "Unknown" in unique_values:
        unique_values.remove("Unknown")
        unique_values.append("Unknown")
    return {value: index for index, value in enumerate(unique_values)}


def make_embedding(num_embeddings: int, embedding_dim: int, seed: int) -> nn.Embedding:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    with torch.no_grad():
        weights = torch.empty((num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(weights, generator=generator)
        embedding.weight.copy_(weights)
    return embedding.eval()


def load_categories(raw_dataset: Path) -> pl.DataFrame:
    columns = ["music_id", "main_genre", "music_lang"]
    return pl.scan_parquet(raw_dataset).select(columns).collect().with_columns(
        pl.col("main_genre")
        .map_elements(normalize_category, return_dtype=pl.String)
        .alias("main_genre_clean"),
        pl.col("music_lang")
        .map_elements(normalize_category, return_dtype=pl.String)
        .alias("music_lang_clean"),
    )


def ids_from_mapping(values: list[str], mapping: dict[str, int]) -> list[int]:
    if "Unknown" in mapping:
        unknown_id = mapping["Unknown"]
        return [mapping.get(value, unknown_id) for value in values]
    return [mapping[value] for value in values]


def lookup_embeddings(ids: list[int], embedding: nn.Embedding) -> list[list[float]]:
    id_tensor = torch.tensor(ids, dtype=torch.long)
    with torch.inference_mode():
        vectors = embedding(id_tensor).float()
    return vectors.tolist()


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    row_count: int,
    genre_to_id: dict[str, int],
    language_to_id: dict[str, int],
) -> None:
    metadata = {
        "raw_dataset": str(args.raw_dataset),
        "row_count": row_count,
        "genre_count": len(genre_to_id),
        "language_count": len(language_to_id),
        "genre_metadata_dim": GENRE_METADATA_DIM,
        "language_metadata_dim": LANGUAGE_METADATA_DIM,
        "genre_condition_dim": GENRE_CONDITION_DIM,
        "language_condition_dim": LANGUAGE_CONDITION_DIM,
        "condition_dim": GENRE_CONDITION_DIM + LANGUAGE_CONDITION_DIM,
        "embedding_seed": args.embedding_seed,
        "unknown_value": "Unknown",
        "outputs": {
            "processed_rows": "genre_language_embeddings.parquet",
            "mappings": "mappings.json",
            "embedding_weights": "embedding_weights.pt",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess main_genre and music_lang into IDs and embedding vectors."
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-seed", type=int, default=751)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading genre/language columns from: %s", args.raw_dataset)
    df = load_categories(args.raw_dataset)
    genre_values = df["main_genre_clean"].to_list()
    language_values = df["music_lang_clean"].to_list()

    genre_to_id = build_mapping(genre_values)
    language_to_id = build_mapping(language_values)
    genre_ids = ids_from_mapping(genre_values, genre_to_id)
    language_ids = ids_from_mapping(language_values, language_to_id)
    LOGGER.info("Mapped %s genres and %s languages", len(genre_to_id), len(language_to_id))

    genre_metadata_embedding = make_embedding(
        len(genre_to_id), GENRE_METADATA_DIM, args.embedding_seed
    )
    language_metadata_embedding = make_embedding(
        len(language_to_id), LANGUAGE_METADATA_DIM, args.embedding_seed + 1
    )
    genre_condition_embedding = make_embedding(
        len(genre_to_id), GENRE_CONDITION_DIM, args.embedding_seed + 2
    )
    language_condition_embedding = make_embedding(
        len(language_to_id), LANGUAGE_CONDITION_DIM, args.embedding_seed + 3
    )

    genre_metadata_vectors = lookup_embeddings(genre_ids, genre_metadata_embedding)
    language_metadata_vectors = lookup_embeddings(language_ids, language_metadata_embedding)
    genre_condition_vectors = lookup_embeddings(genre_ids, genre_condition_embedding)
    language_condition_vectors = lookup_embeddings(language_ids, language_condition_embedding)
    condition_vectors = [
        genre_vector + language_vector
        for genre_vector, language_vector in zip(genre_condition_vectors, language_condition_vectors)
    ]

    output = df.select(["music_id", "main_genre", "music_lang"]).with_columns(
        pl.Series("main_genre_clean", genre_values),
        pl.Series("music_lang_clean", language_values),
        pl.Series("main_genre_id", genre_ids),
        pl.Series("music_lang_id", language_ids),
        pl.Series("main_genre_embedding", genre_metadata_vectors),
        pl.Series("music_lang_embedding", language_metadata_vectors),
        pl.Series("genre_condition_embedding", genre_condition_vectors),
        pl.Series("language_condition_embedding", language_condition_vectors),
        pl.Series("condition_embedding", condition_vectors),
    )

    output_path = args.output_dir / "genre_language_embeddings.parquet"
    mappings_path = args.output_dir / "mappings.json"
    weights_path = args.output_dir / "embedding_weights.pt"

    output.write_parquet(output_path)
    mappings = {
        "genre_to_id": genre_to_id,
        "language_to_id": language_to_id,
        "id_to_genre": {str(index): value for value, index in genre_to_id.items()},
        "id_to_language": {str(index): value for value, index in language_to_id.items()},
    }
    mappings_path.write_text(json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "genre_metadata_embedding": genre_metadata_embedding.state_dict(),
            "language_metadata_embedding": language_metadata_embedding.state_dict(),
            "genre_condition_embedding": genre_condition_embedding.state_dict(),
            "language_condition_embedding": language_condition_embedding.state_dict(),
        },
        weights_path,
    )
    write_metadata(args.output_dir, args, output.height, genre_to_id, language_to_id)

    LOGGER.info("Wrote %s rows to %s", output.height, output_path)
    LOGGER.info("Wrote mappings to %s", mappings_path)
    LOGGER.info("Wrote embedding weights to %s", weights_path)


if __name__ == "__main__":
    main()
