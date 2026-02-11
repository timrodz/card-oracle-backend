import argparse
import logging
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional

from pydantic import ValidationError
from pymongo import MongoClient, ReplaceOne

from app.data_pipeline.sentence_transformers import (
    embed_text,
    load_transformer,
)
from app.models.embeddings import CardEmbeddingRecord
from app.models.scryfall import ScryfallCard, ScryfallCardFace
from app.settings import get_settings


@dataclass
class Config:
    mongodb_uri: str
    mongodb_db: str
    mongodb_source_collection: str
    mongodb_embeddings_collection: str
    embed_model_name: str
    embed_model_path: str
    mongo_batch_size: int
    normalize_embeddings: bool


def load_config() -> Config:
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    return Config(
        mongodb_uri=settings.mongodb_uri,
        mongodb_db=settings.mongodb_db,
        mongodb_source_collection=settings.mongodb_collection,
        mongodb_embeddings_collection=settings.mongodb_collection_embeddings,
        embed_model_name=settings.embed_model_name,
        embed_model_path=settings.embed_model_path,
        mongo_batch_size=settings.mongo_batch_size,
        normalize_embeddings=settings.normalize_embeddings,
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_db_cards(*, source_collection, limit: Optional[int]) -> Iterator[ScryfallCard]:
    cursor = source_collection.find({})
    if limit is not None:
        cursor = cursor.limit(limit)
    for index, card in enumerate(cursor):
        try:
            yield ScryfallCard.model_validate(card)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid Scryfall card at cursor index {index}: {exc}"
            ) from exc


def _is_empty_face(face: ScryfallCardFace) -> bool:
    is_empty_mana_cost = face.mana_cost == ""
    if isinstance(face.colors, list):
        is_empty_colors = len(face.colors) == 0
    else:
        is_empty_colors = face.colors is None
    return is_empty_mana_cost and is_empty_colors


def _is_empty_card(card: ScryfallCard) -> bool:
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


def _should_filter_out_empty_card(card: ScryfallCard) -> bool:
    type_line = card.type_line
    if type_line not in {"Card", "Card // Card"}:
        return False

    faces = card.card_faces
    if faces:
        return any(_is_empty_face(face) for face in faces)

    return _is_empty_card(card)


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip()


def normalize_mana_symbols(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def build_searchable_representation(card: ScryfallCard) -> str:
    name = card.name
    type_line = card.type_line
    mana_cost = normalize_mana_symbols(card.mana_cost)
    cmc = card.cmc
    oracle_text = normalize_mana_symbols(card.oracle_text)
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


def build_card_record(card: ScryfallCard) -> CardEmbeddingRecord | None:
    if card.mongo_id is None:
        return None
    searchable_representation = build_searchable_representation(card)
    return CardEmbeddingRecord(
        _id=card.mongo_id,
        source_id=card.mongo_id,
        summary=searchable_representation,
    )


def embed_records(
    *,
    model,
    chunk_records: List[CardEmbeddingRecord],
    normalize_embeddings: bool,
) -> List[CardEmbeddingRecord]:
    for record in chunk_records:
        record.embeddings = embed_text(
            model,
            record.summary,
            normalize_embeddings,
        )
    return chunk_records


def upsert_embeddings(*, collection, records: List[CardEmbeddingRecord]) -> None:
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
    collection.bulk_write(operations, ordered=False)


def run_pipeline(*, config: Config, limit: Optional[int]) -> None:
    model = load_transformer(config.embed_model_name, config.embed_model_path)

    client: MongoClient = MongoClient(config.mongodb_uri)
    source_collection = client[config.mongodb_db][config.mongodb_source_collection]
    embeddings_collection = client[config.mongodb_db][
        config.mongodb_embeddings_collection
    ]

    total_cards = 0
    total_empty_cards = 0
    total_chunks = 0
    record_batch: List[CardEmbeddingRecord] = []

    for db_card in load_db_cards(source_collection=source_collection, limit=limit):
        total_cards += 1
        if _should_filter_out_empty_card(db_card):
            logging.debug("Empty card %s: %s", db_card.id, db_card.name)
            total_empty_cards += 1
            continue

        record = build_card_record(db_card)
        if record is None:
            continue
        record_batch.append(record)
        total_chunks += 1

        # Once the amount of records reaches the batch size, upsert those in one go
        if len(record_batch) >= config.mongo_batch_size:
            embedded_records = embed_records(
                model=model,
                chunk_records=record_batch,
                normalize_embeddings=config.normalize_embeddings,
            )
            upsert_embeddings(
                collection=embeddings_collection, records=embedded_records
            )
            logging.info("Upserted %d chunks", len(record_batch))
            record_batch.clear()

    # Last batch that isn't in mongo_batch_size
    if record_batch:
        embedded_records = embed_records(
            model=model,
            chunk_records=record_batch,
            normalize_embeddings=config.normalize_embeddings,
        )
        upsert_embeddings(collection=embeddings_collection, records=embedded_records)
        logging.info("Upserted %d chunks", len(record_batch))

    logging.info("Total cards read: %d", total_cards)
    logging.info("Total empty cards: %d", total_empty_cards)
    logging.info("Total chunks created: %d", total_chunks)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Create MTG card embeddings.")
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
