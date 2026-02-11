from pydantic import BaseModel, Field


class CardEmbeddingRecord(BaseModel):
    mongo_id: str = Field(alias="_id")
    source_id: str
    summary: str
    embeddings: list[float] | None = []
