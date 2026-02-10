from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cors_origins: str = "http://localhost:3000"

    mongodb_uri: str
    mongodb_db: str = "mtg"
    mongodb_collection: str = "cards"
    mongodb_collection_embeddings: str = "card_embeddings"

    scryfall_dataset_path: Path = Path("datasets/scryfall")
    mongo_batch_size: int = 500

    embed_model_name: str = "mixedbread-ai/mxbai-embed-xsmall-v1"
    embed_model_path: str = "models/mixedbread-ai/mxbai-embed-xsmall-v1"
    normalize_embeddings: bool = True

    vector_index_name: str = "vector_index"
    vector_embed_path: str = "embeddings"
    vector_num_candidates: int = 100
    vector_limit: int = 5

    rag_max_context_chars: int = 4000
    llm_provider: str = "ollama"
    llm_model: str = "mistral"
    llm_endpoint: str | None = None
    llm_timeout: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
