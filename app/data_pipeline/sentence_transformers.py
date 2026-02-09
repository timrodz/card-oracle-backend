import logging
from pathlib import Path
from types import ModuleType
from typing import List, Optional

from sentence_transformers import SentenceTransformer

torch_module: Optional[ModuleType]
try:
    import torch as torch_module
except ImportError:  # pragma: no cover - torch is a dependency of sentence-transformers
    torch_module = None


def load_transformer(model_name: str, model_path: str) -> SentenceTransformer:
    device = "cpu"
    if torch_module is not None and torch_module.cuda.is_available():
        device = "cuda"
    resolved_path = Path(model_path)
    if resolved_path.exists():
        logging.info(
            "Loading embedding model from path: %s (device=%s)",
            resolved_path,
            device,
        )
        model = SentenceTransformer(str(resolved_path), device=device)
    else:
        logging.info(
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
