import asyncio
import json
import re
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.core.db import database
from app.models.db import ScryfallCardRecord

router = APIRouter(prefix="/cards", tags=["items"])


def _normalize_card_id(raw_id: str) -> str:
    value = raw_id.strip()

    match = re.fullmatch(r"\{\s*source_id\s*:\s*([^}]+)\s*\}", value)
    if match:
        return match.group(1).strip().strip('"').strip("'")

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


@router.get("/{id}", response_model=ScryfallCardRecord)
async def fetch_card(id: str) -> Dict[str, Any]:

    normalized_id = _normalize_card_id(id)
    query = {"_id": normalized_id}
    card = await asyncio.to_thread(database.cards_collection.find_one, query)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return ScryfallCardRecord.model_validate(card).model_dump(
        by_alias=True, exclude_none=True
    )
