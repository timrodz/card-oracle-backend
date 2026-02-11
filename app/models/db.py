from pydantic import BaseModel, Field

from app.models.scryfall import ScryfallCardBase


class CardEmbeddingRecord(BaseModel):
    mongo_id: str = Field(alias="_id")
    source_id: str
    summary: str
    embeddings: list[float] = []


class ScryfallCardRecord(ScryfallCardBase):
    mongo_id: str = Field(alias="_id")
