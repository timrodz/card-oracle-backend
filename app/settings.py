from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_json_file_type(value: Path) -> Path:
    if value.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file path, got: {value}")
    return value


JsonFilePath = Annotated[Path, AfterValidator(_validate_json_file_type)]


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

    scryfall_dataset_file: JsonFilePath = Path("datasets/scryfall/bulk.json")
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
