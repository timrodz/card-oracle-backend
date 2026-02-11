import argparse
import re
from typing import Iterator, List, Optional

from loguru import logger
from pymongo import ReplaceOne
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer

from app.core.config import db_settings, transformer_settings
from app.core.db import database
from app.data_pipeline.sentence_transformers import (
    embed_text,
    load_transformer,
)
from app.models.db import CardEmbeddingRecord, ScryfallCardRecord
from app.models.scryfall import ScryfallCardFace


class Embeddings:
    def __load_db_cards(self, limit: Optional[int]) -> Iterator[ScryfallCardRecord]:
        cursor = database.cards_collection.find()

        if limit is not None:
            cursor = cursor.limit(limit)
        for card in cursor:
            yield ScryfallCardRecord.model_validate(card)

    def __normalize_mana_symbols(self, text: str | None) -> str | None:
        if text is None:
            return None
        normalized = text.replace("{", " ").replace("}", " ")
        return re.sub(r"\s+", " ", normalized).strip()

    def __is_empty_face(self, face: ScryfallCardFace) -> bool:
        is_empty_mana_cost = face.mana_cost == ""
        if isinstance(face.colors, list):
            is_empty_colors = len(face.colors) == 0
        else:
            is_empty_colors = face.colors is None
        return is_empty_mana_cost and is_empty_colors

    def __is_empty_card(self, card: ScryfallCardRecord) -> bool:
        is_empty_cmc = card.cmc == 0
        is_empty_colors = len(card.colors) == 0
        is_empty_color_identity = len(card.color_identity) == 0
        is_empty_keywords = len(card.keywords) == 0
        is_empty_mana_cost = card.mana_cost in {"", None}
        return (
            is_empty_cmc
            and is_empty_colors
            and is_empty_color_identity
            and is_empty_keywords
            and is_empty_mana_cost
        )

    def __should_filter_out_empty_card(self, card: ScryfallCardRecord) -> bool:
        type_line = card.type_line
        if type_line not in {"Card", "Card // Card"}:
            return False

        faces = card.card_faces
        if faces:
            return any(self.__is_empty_face(face) for face in faces)

        return self.__is_empty_card(card)

    def __build_chunks_as_searchable_representation(
        self, card: ScryfallCardRecord
    ) -> str:
        name = card.name
        type_line = card.type_line
        mana_cost = self.__normalize_mana_symbols(card.mana_cost)
        cmc = card.cmc
        oracle_text = self.__normalize_mana_symbols(card.oracle_text)
        flavor_text = card.flavor_text
        price_usd = card.prices.usd
        set_name = card.set_name

        cost_parts: List[str] = []
        if mana_cost:
            cost_parts.append(f"Cost: {mana_cost}")
        if cmc is not None:
            cost_parts.append(f"(CMC or mana value {cmc})")
        cost_section = " ".join(cost_parts) if cost_parts else "Cost: None"

        abilities_section = (
            f"Abilities: {oracle_text}" if oracle_text else "Abilities: None"
        )

        # Unused
        if price_usd is None:
            _price_section = "Current Price: None"
        else:
            _price_section = f"Current Price: ${price_usd} USD"

        _flavor_section = f"Flavor: {flavor_text}" if flavor_text else "Flavor: None"

        return f"Card Name: {name}. Type: {type_line}. Set: {set_name}. {cost_section}. {abilities_section}. "

    def __build_card_record(
        self, card: ScryfallCardRecord
    ) -> CardEmbeddingRecord | None:
        if card.mongo_id is None:
            return None
        searchable_representation = self.__build_chunks_as_searchable_representation(
            card
        )
        return CardEmbeddingRecord(
            _id=card.mongo_id,
            source_id=card.mongo_id,
            summary=searchable_representation,
        )

    def __embed_records(
        self,
        *,
        model,
        records: List[CardEmbeddingRecord],
        normalize_embeddings: bool,
    ) -> List[CardEmbeddingRecord]:
        for record in records:
            record.embeddings = embed_text(
                model=model,
                text=record.summary,
                normalize=normalize_embeddings,
            )
        return records

    def create_search_index(self) -> None:
        # Create your index model, then create the search index
        search_index_model = SearchIndexModel(
            name="vector_index",
            type="vectorSearch",
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": transformer_settings.embedding_dimensions,
                        "path": "embeddings",
                        "similarity": "cosine",
                    }
                ]
            },
        )
        logger.info(f"Creating search index for {database.embeddings_collection.name}")
        database.embeddings_collection.create_search_index(model=search_index_model)
        logger.info("Search index created")

    def __upsert_embeddings(self, records: List[CardEmbeddingRecord]) -> None:
        operations = [
            ReplaceOne(
                {"_id": rec.mongo_id},
                rec.model_dump(by_alias=True, exclude_none=True),
                upsert=True,
            )
            for rec in records
        ]
        if not operations:
            return
        database.embeddings_collection.bulk_write(operations, ordered=False)

    def __upsert_embedding_batch(
        self, *, model: SentenceTransformer, records: List[CardEmbeddingRecord]
    ):
        embedded_records = self.__embed_records(
            model=model,
            records=records,
            normalize_embeddings=transformer_settings.normalize_embeddings,
        )
        self.__upsert_embeddings(embedded_records)
        logger.info("Upserted %d chunks", len(records))
        records.clear()

    def run_pipeline(self, limit: Optional[int]) -> None:
        model = load_transformer()

        total_cards = 0
        total_empty_cards = 0
        total_chunks = 0
        record_batch: List[CardEmbeddingRecord] = []

        cards = self.__load_db_cards(limit)
        for db_card in cards:
            total_cards += 1
            if self.__should_filter_out_empty_card(db_card):
                logger.debug("Empty card %s: %s", db_card.mongo_id, db_card.name)
                total_empty_cards += 1
                continue

            record = self.__build_card_record(db_card)
            if record is None:
                continue
            record_batch.append(record)
            total_chunks += 1

            # Once the amount of records reaches the batch size, upsert those in one go
            if len(record_batch) >= db_settings.batch_size:
                self.__upsert_embedding_batch(model=model, records=record_batch)

        # Last batch
        if record_batch:
            self.__upsert_embedding_batch(model=model, records=record_batch)

        logger.info("Total cards read: %d", total_cards)
        logger.info("Total empty cards: %d", total_empty_cards)
        logger.info("Total chunks created: %d", total_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MTG card embeddings.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cards processed (useful for sampling).",
    )
    args = parser.parse_args()

    embeddings = Embeddings()
    embeddings.run_pipeline(args.limit)
    embeddings.create_search_index()


if __name__ == "__main__":
    main()
