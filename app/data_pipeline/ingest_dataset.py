import argparse
import json
from pathlib import Path
from typing import Iterator, List, Optional

from loguru import logger
from pydantic import ValidationError
from pymongo import ReplaceOne

from app.core.config import db_settings, path_settings
from app.core.db import database
from app.models.scryfall import ScryfallCard


class IngestDataset:
    def __load_scryfall_cards(self, limit: Optional[int]) -> Iterator[ScryfallCard]:
        path = path_settings.scryfall_dataset_file
        yielded = 0
        logger.info("Loading dataset file: %s", path)

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

    def __upsert_cards(self, cards: List[ScryfallCard]) -> None:
        operations = [
            ReplaceOne(
                {"_id": card.id},
                card.model_dump(exclude={"id"}),
                upsert=True,
            )
            for card in cards
        ]
        if not operations:
            return
        database.cards_collection.bulk_write(operations, ordered=False)

    def run_pipeline(self, limit: Optional[int]) -> None:

        total_cards = 0
        skipped_cards = 0
        buffer: List[ScryfallCard] = []

        for scryfall_card in self.__load_scryfall_cards(limit):
            total_cards += 1
            buffer.append(scryfall_card)

            if len(buffer) >= db_settings.batch_size:
                self.__upsert_cards(buffer)
                logger.info("Upserted %d cards", len(buffer))
                buffer.clear()

        if buffer:
            self.__upsert_cards(buffer)
            logger.info("Upserted %d cards", len(buffer))

        logger.info("Total cards read: %d", total_cards)
        logger.info("Total cards skipped (missing id): %d", skipped_cards)


def main() -> None:
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

    ingestion = IngestDataset()
    ingestion.run_pipeline(args.limit)


if __name__ == "__main__":
    main()
