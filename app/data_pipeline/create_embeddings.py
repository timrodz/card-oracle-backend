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

from app.data_pipeline.embeddings import embed_text, load_embedder


@dataclass
class Config:
    dataset_path: Path
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    model_name: str
    model_path: str
    mongo_batch_size: int
    normalize_embeddings: bool


def load_config() -> Config:
    load_dotenv(dotenv_path=".env")
    dataset_path = Path(os.getenv("SCRYFALL_DATASET_PATH", "datasets/scryfall"))
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI is required in the environment")

    mongodb_db = os.getenv("MONGODB_DB", "mtg")
    mongodb_collection = os.getenv("MONGODB_COLLECTION_EMBEDDINGS", "card_embeddings")
    model_name = os.getenv("EMBED_MODEL_NAME", "mixedbread-ai/mxbai-embed-xsmall-v1")
    model_path = os.getenv(
        "EMBED_MODEL_PATH", "models/mixedbread-ai/mxbai-embed-xsmall-v1"
    )
    mongo_batch_size = int(os.getenv("MONGO_BATCH_SIZE", "500"))
    normalize_embeddings = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

    return Config(
        dataset_path=dataset_path,
        mongodb_uri=mongodb_uri,
        mongodb_db=mongodb_db,
        mongodb_collection=mongodb_collection,
        model_name=model_name,
        model_path=model_path,
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
        "flavor_text": raw_card.get("flavor_text"),
        "price_usd": price_usd,
        "cmc": raw_card.get("cmc"),
        "mana_cost": raw_card.get("mana_cost"),
        "set_name": raw_card.get("set_name"),
    }


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip()


def normalize_mana_symbols(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def build_searchable_representation(card: Dict[str, Any]) -> str:
    name = card.get("name") or "Unknown"
    type_line = card.get("type_line") or "Unknown"
    mana_cost = normalize_mana_symbols(card.get("mana_cost"))
    cmc = card.get("cmc")
    oracle_text = normalize_mana_symbols(card.get("oracle_text"))
    flavor_text = card.get("flavor_text")
    price_usd = card.get("price_usd")

    cost_parts: List[str] = []
    if mana_cost:
        cost_parts.append(f"Cost: {mana_cost}")
    if cmc is not None:
        cost_parts.append(f"(CMC {cmc})")
        cost_parts.append(f"(Mana Value {cmc})")
    cost_section = " ".join(cost_parts) if cost_parts else "Cost: None"

    abilities_section = (
        f"Abilities: {oracle_text}" if oracle_text else "Abilities: None"
    )
    flavor_section = f"Flavor: {flavor_text}" if flavor_text else "Flavor: None"

    if price_usd is None:
        price_section = "Current Price: None"
    else:
        price_section = f"Current Price: ${price_usd} USD"

    return (
        f"Card Name: {name}. "
        f"Type: {type_line}. "
        f"{cost_section}. "
        f"{abilities_section}. "
        f"{flavor_section}. "
        f"{price_section}."
    )


def build_card_record(card: Dict[str, Any]) -> Dict[str, Any]:
    searchable_representation = build_searchable_representation(card)
    return {
        "_id": card["id"],
        "source_id": card["id"],
        "summary": searchable_representation,
    }


def embed_chunks(
    model,
    chunk_records: List[Dict[str, Any]],
    normalize_embeddings: bool,
    model_name: str,
) -> List[Dict[str, Any]]:
    for record in chunk_records:
        record["embeddings"] = embed_text(
            model,
            record["summary"],
            normalize_embeddings,
        )
        record["created_at"] = datetime.now(timezone.utc)
    return chunk_records


def upsert_embeddings(collection, records: List[Dict[str, Any]]) -> None:
    operations = [ReplaceOne({"_id": rec["_id"]}, rec, upsert=True) for rec in records]
    if not operations:
        return
    collection.bulk_write(operations, ordered=False)


def run_pipeline(config: Config, limit: Optional[int]) -> None:
    model = load_embedder(config.model_name, config.model_path)

    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    total_cards = 0
    missing_oracle_text = 0
    total_chunks = 0
    buffer: List[Dict[str, Any]] = []

    for raw_card in load_raw_cards(config.dataset_path, limit):
        total_cards += 1
        card = select_and_validate_fields(raw_card)
        if card is None:
            continue

        card["oracle_text"] = normalize_text(card["oracle_text"])
        if not card["oracle_text"]:
            missing_oracle_text += 1

        record = build_card_record(card)
        buffer.append(record)
        total_chunks += 1

        if len(buffer) >= config.mongo_batch_size:
            embedded = embed_chunks(
                model,
                buffer,
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
            config.normalize_embeddings,
            config.model_name,
        )
        upsert_embeddings(collection, embedded)
        logging.info("Upserted %d chunks", len(buffer))

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
