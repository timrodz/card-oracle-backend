import os

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.core import fetch_card, search_rag, search_rag_stream

load_dotenv(dotenv_path=".env")

app = FastAPI()

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
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


@app.get("/search")
async def search(query: str = Query(..., min_length=1)) -> dict:
    return await search_rag(query)


@app.get("/stream/search")
async def stream_search(query: str = Query(..., min_length=1)) -> StreamingResponse:
    stream = search_rag_stream(query)
    return StreamingResponse(
        iterate_in_threadpool(stream), media_type="text/event-stream"
    )


@app.get("/cards/{id}")
async def get_card(id: str) -> dict:
    return await fetch_card(id)
