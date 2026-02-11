from fastapi import APIRouter

from app.api.routes import cards, search

api_router = APIRouter()
api_router.include_router(cards.router)
api_router.include_router(search.router)
