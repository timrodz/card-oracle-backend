from fastapi import FastAPI, Query  # noqa: E402

from app.core import fetch_card, search_rag  # noqa: E402

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/search")
async def search(query: str = Query(..., min_length=1)) -> dict:
    return await search_rag(query)


@app.get("/cards/{id}")
async def get_card(id: str) -> dict:
    return await fetch_card(id)
