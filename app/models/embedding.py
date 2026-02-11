from pydantic import BaseModel


class CardEmbeddingVectorSearchResult(BaseModel):
    source_id: str
    summary: str
    score: float
