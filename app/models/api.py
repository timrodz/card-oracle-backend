from typing import Literal

from pydantic import BaseModel


class SearchResult(BaseModel):
    source_id: str
    summary: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    context: str
    answer: str | None = None
    source_id: str | None = None
    answer_raw: str | None = None


class StreamMetaEvent(BaseModel):
    type: Literal["meta"]
    results: list[SearchResult]
    context: str
    answer: str | None = None


class StreamChunkEvent(BaseModel):
    type: Literal["chunk"]
    content: str


class StreamErrorEvent(BaseModel):
    type: Literal["error"]
    message: str
    query: str | None = None


class StreamSeekingCardEvent(BaseModel):
    type: Literal["seeking_card"]


class StreamFoundCardEvent(BaseModel):
    type: Literal["found_card"]
    id: str


class StreamDoneEvent(BaseModel):
    type: Literal["done"]


StreamEvent = (
    StreamMetaEvent
    | StreamChunkEvent
    | StreamErrorEvent
    | StreamSeekingCardEvent
    | StreamFoundCardEvent
    | StreamDoneEvent
)
