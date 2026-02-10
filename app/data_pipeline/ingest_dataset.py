import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import ValidationError
from pymongo import MongoClient, ReplaceOne

from app.settings import get_settings


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_config() -> Dict[str, Any]:
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "dataset_path": Path(settings.scryfall_dataset_path),
        "mongodb_uri": settings.mongodb_uri,
        "mongodb_db": settings.mongodb_db,
        "mongodb_collection": settings.mongodb_collection,
        "mongo_batch_size": settings.mongo_batch_size,
    }


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


def upsert_cards(collection, cards: List[Dict[str, Any]]) -> None:
    operations = [ReplaceOne({"_id": card["_id"]}, card, upsert=True) for card in cards]
    if not operations:
        return
    collection.bulk_write(operations, ordered=False)


def run_pipeline(config: Dict[str, Any], limit: Optional[int]) -> None:
    client: MongoClient = MongoClient(config["mongodb_uri"])
    collection = client[config["mongodb_db"]][config["mongodb_collection"]]

    total_cards = 0
    skipped_cards = 0
    buffer: List[Dict[str, Any]] = []

    for raw_card in load_raw_cards(config["dataset_path"], limit):
        total_cards += 1
        card_id = raw_card.get("id")
        if not card_id:
            skipped_cards += 1
            continue
        record = dict(raw_card)
        record["_id"] = card_id
        record.pop("id", None)
        buffer.append(record)

        if len(buffer) >= config["mongo_batch_size"]:
            upsert_cards(collection, buffer)
            logging.info("Upserted %d cards", len(buffer))
            buffer.clear()

    if buffer:
        upsert_cards(collection, buffer)
        logging.info("Upserted %d cards", len(buffer))

    logging.info("Total cards read: %d", total_cards)
    logging.info("Total cards skipped (missing id): %d", skipped_cards)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Load Scryfall cards into MongoDB.")
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
        config["dataset_path"] = args.dataset_path

    run_pipeline(config, args.limit)


if __name__ == "__main__":
    main()
