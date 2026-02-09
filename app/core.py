import asyncio
import json
import os
from typing import Any, Dict, Iterator

from dotenv import load_dotenv
from fastapi import HTTPException
from pymongo import MongoClient

from app.data_pipeline import query_rag

load_dotenv(dotenv_path=".env")


def get_cards_collection():
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise HTTPException(status_code=500, detail="MONGODB_URI is required")
    mongodb_db = os.getenv("MONGODB_DB", "mtg")
    client: MongoClient = MongoClient(mongodb_uri)
    return client[mongodb_db]["cards"]


async def search_rag(query: str) -> Dict[str, Any]:
    try:
        config = query_rag.load_config()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = await asyncio.to_thread(query_rag.search, query, config)
    except Exception as exc:  # pragma: no cover - depends on external services
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return result


def _encode_sse(event: Dict[str, Any]) -> str:
    payload = json.dumps(event)
    return f"data: {payload}\n\n"


def search_rag_stream(query: str) -> Iterator[str]:
    try:
        config = query_rag.load_config()
    except ValueError as exc:
        yield _encode_sse({"type": "error", "message": str(exc), "query": query})
        yield _encode_sse({"type": "done"})
        return

    try:
        for event in query_rag.search_stream(query, config):
            if event["type"] == "meta":
                print(event)
            yield _encode_sse(event)
    except Exception as exc:  # pragma: no cover - depends on external services
        yield _encode_sse(
            {"type": "error", "message": f"Search failed: {exc}", "query": query}
        )
        yield _encode_sse({"type": "done"})


async def fetch_card(id: str) -> Dict[str, Any]:
    collection = get_cards_collection()
    query = {"_id": id}
    card = await asyncio.to_thread(collection.find_one, query)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return card
