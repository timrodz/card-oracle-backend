from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import app_settings

app = FastAPI(title="Card Oracle")

cors_origins = app_settings.cors_origins
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
def health_check():
    return {"Hello": "World"}


app.include_router(api_router)
