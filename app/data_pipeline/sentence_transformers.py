from pathlib import Path
from typing import List

import torch as torch_module
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import transformer_settings


def load_transformer() -> SentenceTransformer:
    device = "cpu"
    if torch_module.cuda.is_available():
        device = "cuda"
    model_name = transformer_settings.embedding_model_name
    model_path = transformer_settings.embedding_model_path
    resolved_path = Path(model_path)
    if resolved_path.exists():
        logger.info(
            "Loading embedding model from path: %s (device=%s)",
            resolved_path,
            device,
        )
        model = SentenceTransformer(str(resolved_path), device=device)
    else:
        logger.info(
            "Loading embedding model: %s (device=%s, path=%s)",
            model_name,
            device,
            model_path,
        )
        model = SentenceTransformer(model_name, device=device)
        resolved_path.mkdir(parents=True, exist_ok=True)
        model.save(str(resolved_path))
    return model


def embed_text(
    *,
    model: SentenceTransformer,
    text: str,
    normalize: bool,
) -> List[float]:
    embeddings = model.encode(
        text,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.tolist()
