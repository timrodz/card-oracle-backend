import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:  # pragma: no cover - torch is a dependency of sentence-transformers
    torch = None


@dataclass
class Config:
    dataset_path: Path
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    model_name: str
    chunk_tokens: int
    chunk_overlap: int
    embed_batch_size: int
    mongo_batch_size: int
    normalize_embeddings: bool


def load_config() -> Config:
    load_dotenv(dotenv_path=".env")
    dataset_path = Path(os.getenv("SCRYFALL_DATASET_PATH", "datasets/scryfall"))
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI is required in the environment")

    mongodb_db = os.getenv("MONGODB_DB", "mtg")
    mongodb_collection = os.getenv("MONGODB_COLLECTION", "card_embeddings")
    model_name = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
    chunk_tokens = int(os.getenv("CHUNK_TOKENS", "320"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
    embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    mongo_batch_size = int(os.getenv("MONGO_BATCH_SIZE", "500"))
    normalize_embeddings = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

    if chunk_overlap >= chunk_tokens:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_TOKENS")

    return Config(
        dataset_path=dataset_path,
        mongodb_uri=mongodb_uri,
        mongodb_db=mongodb_db,
        mongodb_collection=mongodb_collection,
        model_name=model_name,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        embed_batch_size=embed_batch_size,
        mongo_batch_size=mongo_batch_size,
        normalize_embeddings=normalize_embeddings,
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def iter_dataset_files(dataset_path: Path) -> List[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    return sorted(p for p in dataset_path.iterdir() if p.suffix == ".json")


def load_raw_cards(
    dataset_path: Path, limit: Optional[int]
) -> Iterator[Dict[str, Any]]:
    yielded = 0
    for path in iter_dataset_files(dataset_path):
        logging.info("Loading dataset file: %s", path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, got {type(data)}")
        for card in data:
            if isinstance(card, dict):
                yield card
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def select_and_validate_fields(raw_card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    oracle_text = raw_card.get("oracle_text")
    if not oracle_text or not isinstance(oracle_text, str) or not oracle_text.strip():
        oracle_text = None

    prices = raw_card.get("prices") or {}
    price_usd = prices.get("usd") if isinstance(prices, dict) else None
    return {
        "id": raw_card.get("id"),
        "name": raw_card.get("name"),
        "collector_number": raw_card.get("collector_number"),
        "type_line": raw_card.get("type_line"),
        "rarity": raw_card.get("rarity"),
        "oracle_text": oracle_text,
        "price_usd": price_usd,
        "cmc": raw_card.get("cmc"),
        "mana_cost": raw_card.get("mana_cost"),
        "set_name": raw_card.get("set_name"),
    }


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip()


def chunk_oracle_text(
    oracle_text: Optional[str],
    tokenizer,
    max_tokens: int,
    overlap: int,
) -> List[str]:
    if not oracle_text:
        return []
    encoding = tokenizer(
        oracle_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding.get("offset_mapping", [])
    if not offsets:
        return []

    chunks: List[str] = []
    start = 0
    total_tokens = len(offsets)
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk = oracle_text[char_start:char_end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= total_tokens:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunk_records(
    card: Dict[str, Any], chunks: List[str]
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        records.append(
            {
                "_id": f"{card['id']}:{index}",
                "source_id": card["id"],
                "chunk_index": index,
                "chunk_count": len(chunks),
                "chunk_text": chunk,
                "oracle_text": card["oracle_text"],
                "has_oracle_text": True,
                "price_usd": card["price_usd"],
                "cmc": card["cmc"],
                "mana_cost": card["mana_cost"],
                "set_name": card["set_name"],
                "name": card["name"],
                "type_line": card["type_line"],
                "rarity": card["rarity"],
            }
        )
    return records


def build_empty_record(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": f"{card['id']}:-1",
        "source_id": card["id"],
        "chunk_index": -1,
        "chunk_count": 0,
        "chunk_text": None,
        "oracle_text": card["oracle_text"],
        "has_oracle_text": False,
        "price_usd": card["price_usd"],
        "cmc": card["cmc"],
        "mana_cost": card["mana_cost"],
        "set_name": card["set_name"],
        "name": card["name"],
        "type_line": card["type_line"],
        "rarity": card["rarity"],
        "embeddings": None,
        "embedding_model": None,
        "embedding_dim": 0,
        "embedded_at": None,
    }


def embed_chunks(
    model: SentenceTransformer,
    chunk_records: List[Dict[str, Any]],
    batch_size: int,
    normalize_embeddings: bool,
    model_name: str,
) -> List[Dict[str, Any]]:
    texts = [record["chunk_text"] for record in chunk_records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    for record, embedding in zip(chunk_records, embeddings):
        record["embeddings"] = embedding.tolist()
        record["embedding_model"] = model_name
        record["embedding_dim"] = len(record["embeddings"])
        record["embedded_at"] = datetime.now(timezone.utc)
    return chunk_records


def upsert_embeddings(collection, records: List[Dict[str, Any]]) -> None:
    operations = [ReplaceOne({"_id": rec["_id"]}, rec, upsert=True) for rec in records]
    if not operations:
        return
    collection.bulk_write(operations, ordered=False)


def run_pipeline(config: Config, limit: Optional[int]) -> None:
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"

    logging.info("Loading embedding model: %s (device=%s)", config.model_name, device)
    model = SentenceTransformer(config.model_name, device=device)
    tokenizer = model.tokenizer

    client = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    total_cards = 0
    missing_oracle_text = 0
    total_chunks = 0
    buffer: List[Dict[str, Any]] = []
    empty_buffer: List[Dict[str, Any]] = []

    for raw_card in load_raw_cards(config.dataset_path, limit):
        total_cards += 1
        card = select_and_validate_fields(raw_card)
        if card is None:
            continue

        card["oracle_text"] = normalize_text(card["oracle_text"])
        if not card["oracle_text"]:
            missing_oracle_text += 1
            empty_buffer.append(build_empty_record(card))
            if len(empty_buffer) >= config.mongo_batch_size:
                upsert_embeddings(collection, empty_buffer)
                logging.info("Upserted %d empty records", len(empty_buffer))
                empty_buffer.clear()
            continue

        chunks = chunk_oracle_text(
            card["oracle_text"],
            tokenizer,
            config.chunk_tokens,
            config.chunk_overlap,
        )
        if not chunks:
            empty_buffer.append(build_empty_record(card))
            if len(empty_buffer) >= config.mongo_batch_size:
                upsert_embeddings(collection, empty_buffer)
                logging.info("Upserted %d empty records", len(empty_buffer))
                empty_buffer.clear()
            continue

        chunk_records = build_chunk_records(card, chunks)
        buffer.extend(chunk_records)
        total_chunks += len(chunk_records)

        if len(buffer) >= config.mongo_batch_size:
            embedded = embed_chunks(
                model,
                buffer,
                config.embed_batch_size,
                config.normalize_embeddings,
                config.model_name,
            )
            upsert_embeddings(collection, embedded)
            logging.info("Upserted %d chunks", len(buffer))
            buffer.clear()

    if buffer:
        embedded = embed_chunks(
            model,
            buffer,
            config.embed_batch_size,
            config.normalize_embeddings,
            config.model_name,
        )
        upsert_embeddings(collection, embedded)
        logging.info("Upserted %d chunks", len(buffer))

    if empty_buffer:
        upsert_embeddings(collection, empty_buffer)
        logging.info("Upserted %d empty records", len(empty_buffer))

    logging.info("Total cards read: %d", total_cards)
    logging.info("Total cards missing oracle_text: %d", missing_oracle_text)
    logging.info("Total chunks created: %d", total_chunks)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Create MTG card embeddings.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to Scryfall JSON file or directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cards processed (useful for sampling).",
    )
    args = parser.parse_args()

    config = load_config()
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path

    run_pipeline(config, args.limit)


if __name__ == "__main__":
    main()
