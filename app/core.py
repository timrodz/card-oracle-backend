import asyncio
import json
import re
from typing import Any, Dict, Iterator

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from pymongo import MongoClient

from app.data_pipeline import query_rag
from app.models.api import CardResponse, SearchResponse, StreamErrorEvent, StreamEvent
from app.settings import get_settings

stream_event_adapter: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


def get_cards_collection():
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    client: MongoClient = MongoClient(settings.mongodb_uri)
    return client[settings.mongodb_db][settings.mongodb_collection]


async def search_rag(query: str) -> Dict[str, Any]:
    try:
        config = query_rag.load_config()
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = await asyncio.to_thread(
            query_rag.search, question=query, config=config
        )
    except Exception as exc:  # pragma: no cover - depends on external services
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return SearchResponse.model_validate(result).model_dump(exclude_none=True)


def _encode_sse(event: Dict[str, Any]) -> str:
    try:
        validated_event = stream_event_adapter.validate_python(event)
        payload = json.dumps(validated_event.model_dump(exclude_none=True))
    except ValidationError as exc:
        fallback = StreamErrorEvent(
            type="error", message=f"Invalid stream event payload: {exc}"
        )
        payload = json.dumps(fallback.model_dump(exclude_none=True))
    return f"data: {payload}\n\n"


def search_rag_stream(query: str) -> Iterator[str]:
    try:
        config = query_rag.load_config()
    except (ValidationError, ValueError) as exc:
        yield _encode_sse({"type": "error", "message": str(exc), "query": query})
        yield _encode_sse({"type": "done"})
        return

    try:
        for event in query_rag.search_stream(question=query, config=config):
            yield _encode_sse(event)
    except Exception as exc:  # pragma: no cover - depends on external services
        yield _encode_sse(
            {"type": "error", "message": f"Search failed: {exc}", "query": query}
        )
        yield _encode_sse({"type": "done"})


async def fetch_card(id: str) -> Dict[str, Any]:
    collection = get_cards_collection()
    normalized_id = _normalize_card_id(id)
    query = {"$or": [{"_id": normalized_id}, {"source_id": normalized_id}]}
    card = await asyncio.to_thread(collection.find_one, query)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return CardResponse.model_validate(card).model_dump(
        by_alias=True, exclude_none=True
    )


def _normalize_card_id(raw_id: str) -> str:
    value = raw_id.strip()

    # Supports path values like `{source_id:uuid}` or `{source_id: uuid}`.
    match = re.fullmatch(r"\{\s*source_id\s*:\s*([^}]+)\s*\}", value)
    if match:
        return match.group(1).strip().strip('"').strip("'")

    # Supports JSON-like values such as `{"source_id":"uuid"}`.
    if value.startswith("{") and value.endswith("}"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, dict):
            source_id = parsed.get("source_id")
            if isinstance(source_id, str) and source_id.strip():
                return source_id.strip()
    return value
