from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, MongoDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_json_file_type(value: Path) -> Path:
    if value.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file path, got: {value}")
    return value


JsonFilePath = Annotated[Path, AfterValidator(_validate_json_file_type)]


class DatasetFileInput(BaseModel):
    dataset_file: JsonFilePath


class DatabaseSettings(BaseModel):
    uri: MongoDsn
    name: str
    cards_collection: str
    card_embeddings_collection: str
    batch_size: int


class PathSettings(BaseModel):
    scryfall_dataset_file: JsonFilePath


class TransformerSettings(BaseModel):
    embedding_model_name: str
    embedding_model_path: Path
    embedding_dimensions: int
    normalize_embeddings: bool
    vector_limit: int


class LlmSettings(BaseModel):
    rag_max_context_chars: int
    llm_provider: str
    llm_model: str
    llm_endpoint: str | None
    llm_timeout: int


class AppSettings(BaseModel):
    cors_origins: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_cors_origins: str = "http://localhost:3000"

    mongodb_uri: MongoDsn = MongoDsn("mongodb://localhost:27017/?directConnection=true")
    mongodb_db: str = "mtg"
    mongodb_cards_collection: str = "cards"
    mongodb_card_embeddings_collection: str = "card_embeddings"
    mongodb_batch_size: int = 500

    path_scryfall_dataset_file: JsonFilePath = Path("datasets/scryfall/bulk.json")

    transformers_embedding_model_name: str = "mixedbread-ai/mxbai-embed-xsmall-v1"
    transformers_embedding_model_path: Path = Path(
        "models/mixedbread-ai/mxbai-embed-xsmall-v1"
    )
    transformers_embedding_model_dimensions: int = 384
    transformers_normalize_embeddings: bool = True
    transformers_vector_limit: int = 5

    llm_rag_max_context_chars: int = 4000
    llm_provider: str = "ollama"
    llm_model: str = "mistral"
    llm_endpoint: str | None = None
    llm_timeout: int = 120

    @property
    def database_settings(self) -> DatabaseSettings:
        return DatabaseSettings(
            uri=self.mongodb_uri,
            name=self.mongodb_db,
            cards_collection=self.mongodb_cards_collection,
            card_embeddings_collection=self.mongodb_card_embeddings_collection,
            batch_size=self.mongodb_batch_size,
        )

    @property
    def path_settings(self) -> PathSettings:
        return PathSettings(
            scryfall_dataset_file=self.path_scryfall_dataset_file,
        )

    @property
    def transformer_settings(self) -> TransformerSettings:
        return TransformerSettings(
            embedding_model_name=self.transformers_embedding_model_name,
            embedding_model_path=self.transformers_embedding_model_path,
            embedding_dimensions=self.transformers_embedding_model_dimensions,
            normalize_embeddings=self.transformers_normalize_embeddings,
            vector_limit=self.transformers_vector_limit,
        )

    @property
    def llm_settings(self) -> LlmSettings:
        return LlmSettings(
            rag_max_context_chars=self.llm_rag_max_context_chars,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            llm_endpoint=self.llm_endpoint,
            llm_timeout=self.llm_timeout,
        )

    @property
    def app_settings(self) -> AppSettings:
        return AppSettings(cors_origins=self.app_cors_origins)


@lru_cache
def _get_settings() -> Settings:
    return Settings()


_settings = _get_settings()
app_settings = _settings.app_settings
db_settings = _settings.database_settings
path_settings = _settings.path_settings
transformer_settings = _settings.transformer_settings
llm_settings = _settings.llm_settings
