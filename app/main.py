from typing import Any, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.core import fetch_card, search_rag, search_rag_stream
from app.models.api import CardResponse, SearchResponse
from app.settings import get_settings

app = FastAPI()

settings = get_settings()
cors_origins = settings.cors_origins
origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/search", response_model=SearchResponse)
async def search(query: str = Query(..., min_length=1)) -> Dict[str, Any]:
    return await search_rag(query)


@app.get("/stream/search")
async def stream_search(query: str = Query(..., min_length=1)) -> StreamingResponse:
    stream = search_rag_stream(query)
    return StreamingResponse(
        iterate_in_threadpool(stream), media_type="text/event-stream"
    )


@app.get("/cards/{id}", response_model=CardResponse)
async def get_card(id: str) -> Dict[str, Any]:
    return await fetch_card(id)
