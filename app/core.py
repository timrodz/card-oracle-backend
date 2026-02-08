import asyncio
import os
import urllib.parse
from typing import Any, Dict

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
    encoded_query = urllib.parse.urlencode(
        {"query": query},
        quote_via=urllib.parse.quote,
    )
    try:
        config = query_rag.load_config()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = await asyncio.to_thread(query_rag.query_rag, query, config)
    except Exception as exc:  # pragma: no cover - depends on external services
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return {
        "query": query,
        "encoded_query": encoded_query,
        **result,
    }


async def fetch_card(id: str) -> Dict[str, Any]:
    collection = get_cards_collection()
    query = {"id": id}
    card = await asyncio.to_thread(collection.find_one, query, {"_id": 0})
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return card
