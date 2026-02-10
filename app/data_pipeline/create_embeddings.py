import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import ValidationError
from pymongo import MongoClient, ReplaceOne

from app.settings import get_settings
from app.data_pipeline.sentence_transformers import (
    embed_text,
    load_transformer,
)


@dataclass
class Config:
    dataset_path: Path
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
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
        dataset_path=Path(settings.scryfall_dataset_path),
        mongodb_uri=settings.mongodb_uri,
        mongodb_db=settings.mongodb_db,
        mongodb_collection=settings.mongodb_collection_embeddings,
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
    oracle_text = raw_card.get("oracle_text", "")
    prices = raw_card.get("prices", {})
    price_usd = prices.get("usd") if isinstance(prices, dict) else None
    return {
        "id": raw_card.get("id"),
        "name": raw_card.get("name"),
        "collector_number": raw_card.get("collector_number"),
        "type_line": raw_card.get("type_line"),
        "rarity": raw_card.get("rarity"),
        "oracle_text": normalize_text(oracle_text),
        "flavor_text": raw_card.get("flavor_text"),
        "price_usd": price_usd,
        "cmc": raw_card.get("cmc"),
        "mana_cost": raw_card.get("mana_cost"),
        "set_name": raw_card.get("set_name"),
    }


def _is_empty_face(face: Dict[str, Any]) -> bool:
    cmc = face.get("cmc")
    colors = face.get("colors")
    color_identity = face.get("color_identity")
    keywords = face.get("keywords")
    mana_cost = face.get("mana_cost")

    is_empty_cmc = cmc == 0 if cmc is not None else True
    is_empty_colors = isinstance(colors, list) and len(colors) == 0
    is_empty_color_identity = (
        isinstance(color_identity, list) and len(color_identity) == 0
        if color_identity is not None
        else True
    )
    is_empty_keywords = (
        isinstance(keywords, list) and len(keywords) == 0
        if keywords is not None
        else True
    )
    is_empty_mana_cost = mana_cost == "" if mana_cost is not None else True
    is_empty = (
        is_empty_cmc
        and is_empty_colors
        and is_empty_color_identity
        and is_empty_keywords
        and is_empty_mana_cost
    )
    # logging.info(
    #     f"Is empty {face.get('name')}: {is_empty}. CMC: {is_empty_cmc}. Mana_cost: {is_empty_mana_cost}. C:{is_empty_colors} CI:{is_empty_color_identity} K:{is_empty_keywords}\n\n{face}\n\n"
    # )
    return is_empty


def _should_filter_out_empty_card(raw_card: Dict[str, Any]) -> bool:
    type_line = raw_card.get("type_line")
    if type_line not in {"Card", "Card // Card"}:
        return False

    faces = raw_card.get("card_faces")
    if isinstance(faces, list) and faces:
        return any(isinstance(face, dict) and _is_empty_face(face) for face in faces)

    return _is_empty_face(raw_card)


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
    set_name = card.get("set_name")

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
    model = load_transformer(config.embed_model_name, config.embed_model_path)

    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    total_cards = 0
    total_empty_cards = 0
    total_chunks = 0
    buffer: List[Dict[str, Any]] = []

    for raw_card in load_raw_cards(config.dataset_path, limit):
        total_cards += 1
        if _should_filter_out_empty_card(raw_card):
            logging.debug(f"Empty card {raw_card.get('id')}: {raw_card.get('name')}")
            total_empty_cards += 1
            continue
        card = select_and_validate_fields(raw_card)
        if card is None:
            continue

        record = build_card_record(card)
        buffer.append(record)
        total_chunks += 1

        if len(buffer) >= config.mongo_batch_size:
            embedded = embed_chunks(
                model,
                buffer,
                config.normalize_embeddings,
            )
            upsert_embeddings(collection, embedded)
            logging.info("Upserted %d chunks", len(buffer))
            buffer.clear()

    if buffer:
        embedded = embed_chunks(
            model,
            buffer,
            config.normalize_embeddings,
        )
        upsert_embeddings(collection, embedded)
        logging.info("Upserted %d chunks", len(buffer))

    logging.info("Total cards read: %d", total_cards)
    logging.info("Total empty cards: %d", total_empty_cards)
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
