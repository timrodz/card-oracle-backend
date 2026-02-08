import logging
from types import ModuleType
from typing import List, Optional

from sentence_transformers import SentenceTransformer

torch_module: Optional[ModuleType]
try:
    import torch as torch_module
except ImportError:  # pragma: no cover - torch is a dependency of sentence-transformers
    torch_module = None


def load_embedder(model_name: str, model_path: str) -> SentenceTransformer:
    device = "cpu"
    if torch_module is not None and torch_module.cuda.is_available():
        device = "cuda"
    logging.info(
        "Loading embedding model: %s (device=%s, path=%s)",
        model_name,
        device,
        model_path,
    )
    model = SentenceTransformer(model_name, device=device)
    model.save(model_path)
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
