import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import ValidationError
from pymongo import MongoClient, ReplaceOne

from app.models.dataset import DatasetFileInput
from app.models.scryfall import ScryfallCard
from app.settings import get_settings


@dataclass
class Config:
    dataset_file: Path
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    mongo_batch_size: int


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_config() -> Config:
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    return Config(
        dataset_file=settings.scryfall_dataset_file,
        mongodb_uri=settings.mongodb_uri,
        mongodb_db=settings.mongodb_db,
        mongodb_collection=settings.mongodb_collection,
        mongo_batch_size=settings.mongo_batch_size,
    )


def load_raw_cards(
    *, dataset_file: Path, limit: Optional[int]
) -> Iterator[ScryfallCard]:
    validated_input = DatasetFileInput(dataset_file=dataset_file)
    yielded = 0
    path = validated_input.dataset_file
    logging.info("Loading dataset file: %s", path)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")

    for index, card in enumerate(data):
        try:
            validated_card = ScryfallCard.model_validate(card)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid Scryfall card in {path}[{index}]: {exc}"
            ) from exc
        yield validated_card
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def upsert_cards(*, collection, cards: List[Dict[str, Any]]) -> None:
    operations = [ReplaceOne({"_id": card["_id"]}, card, upsert=True) for card in cards]
    if not operations:
        return
    collection.bulk_write(operations, ordered=False)


def run_pipeline(*, config: Config, limit: Optional[int]) -> None:
    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    total_cards = 0
    skipped_cards = 0
    buffer: List[Dict[str, Any]] = []

    for raw_card in load_raw_cards(dataset_file=config.dataset_file, limit=limit):
        total_cards += 1
        card_id = raw_card.id
        record = raw_card.model_dump(by_alias=True, exclude_none=True)
        record["_id"] = card_id
        record.pop("id", None)
        buffer.append(record)

        if len(buffer) >= config.mongo_batch_size:
            upsert_cards(collection=collection, cards=buffer)
            logging.info("Upserted %d cards", len(buffer))
            buffer.clear()

    if buffer:
        upsert_cards(collection=collection, cards=buffer)
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
        help="Path to a Scryfall JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cards processed (useful for sampling).",
    )
    args = parser.parse_args()

    config = load_config()
    run_pipeline(config=config, limit=args.limit)


if __name__ == "__main__":
    main()
