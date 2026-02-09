import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

from app.data_pipeline.embeddings import embed_text, load_embedder
from app.data_pipeline.providers.ollama import OllamaProvider
from app.data_pipeline.providers.provider import LLMProvider  # noqa: E402


@dataclass
class Config:
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    embed_model: str
    embed_model_path: str
    normalize_embeddings: bool
    vector_index: str
    vector_path: str
    num_candidates: int
    limit: int
    llm_provider: str
    llm_model: str
    llm_endpoint: str | None
    max_context_chars: int
    llm_timeout: int


def load_config() -> Config:
    load_dotenv(dotenv_path=".env")
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI is required in the environment")

    return Config(
        mongodb_uri=mongodb_uri,
        mongodb_db=os.getenv("MONGODB_DB", "mtg"),
        mongodb_collection=os.getenv(
            "MONGODB_COLLECTION_EMBEDDINGS", "card_embeddings"
        ),
        embed_model=os.getenv(
            "EMBED_MODEL_NAME", "mixedbread-ai/mxbai-embed-xsmall-v1"
        ),
        embed_model_path=os.getenv(
            "EMBED_MODEL_PATH", "models/mixedbread-ai/mxbai-embed-xsmall-v1"
        ),
        normalize_embeddings=os.getenv("NORMALIZE_EMBEDDINGS", "true").lower()
        == "true",
        vector_index=os.getenv("VECTOR_INDEX_NAME", "vector_index"),
        vector_path=os.getenv("VECTOR_EMBED_PATH", "embeddings"),
        num_candidates=int(os.getenv("VECTOR_NUM_CANDIDATES", "100")),
        limit=int(os.getenv("VECTOR_LIMIT", "5")),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "mistral"),
        llm_endpoint=os.getenv("LLM_ENDPOINT"),
        max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000")),
        llm_timeout=int(os.getenv("LLM_TIMEOUT", "120")),
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def embed_query(model, question: str, normalize: bool) -> List[float]:
    return embed_text(model, question, normalize)


def vector_search(
    collection,
    query_vector: List[float],
    vector_index: str,
    vector_path: str,
    num_candidates: int,
    limit: int,
) -> List[Dict[str, Any]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "queryVector": query_vector,
                "path": vector_path,
                # Either numCandidates or exact must be present, not both
                # "numCandidates": num_candidates,
                "exact": True,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "source_id": 1,
                "summary": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def format_result(result: Dict[str, Any]) -> str:
    return result["summary"]


def build_context(results: List[Dict[str, Any]], max_chars: int) -> str:
    sections: List[str] = []
    total = 0
    for result in results:
        section = result["summary"]
        if total + len(section) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                sections.append(section[:remaining])
            break
        sections.append(section)
        total += len(section)
    return "\n".join(sections).strip()


def build_prompt(question: str, context: str) -> str:
    payload = {
        "role": "You help users find cards for Magic: The Gathering.",
        "instructions": (
            "Answer the question using the provided context. "
            "If the context is insufficient, say so and suggest what to ask next."
        ),
        "context": context,
        "question": question,
    }
    logging.debug(payload)
    return json.dumps(payload)


def build_provider(config: Config) -> LLMProvider:
    provider = config.llm_provider.lower()
    if provider == "ollama":
        return OllamaProvider(config.llm_model, config.llm_timeout, config.llm_endpoint)
    raise ValueError(f"Unsupported LLM_PROVIDER: {config.llm_provider}")


def cleanup_response(response: str) -> str:
    text = response.replace('\\"', '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search(question: str, config: Config) -> Dict[str, Any]:
    embedder = load_embedder(config.embed_model, config.embed_model_path)
    query_embeddings = embed_query(embedder, question, config.normalize_embeddings)

    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    results = vector_search(
        collection,
        query_embeddings,
        config.vector_index,
        config.vector_path,
        config.num_candidates,
        config.limit,
    )

    if not results:
        return {"results": [], "context": "", "answer": None}

    context = build_context(results, config.max_context_chars)
    if not context:
        return {"results": results, "context": "", "answer": None}

    prompt = build_prompt(question, context)
    provider = build_provider(config)
    response = provider.generate(prompt)
    clean_response = cleanup_response(response)
    return {
        "results": results,
        "context": context,
        "answer": clean_response,
        "answer_raw": response,
    }


def search_stream(question: str, config: Config) -> Iterator[Dict[str, Any]]:
    embedder = load_embedder(config.embed_model, config.embed_model_path)
    query_embeddings = embed_query(embedder, question, config.normalize_embeddings)

    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    results = vector_search(
        collection,
        query_embeddings,
        config.vector_index,
        config.vector_path,
        config.num_candidates,
        config.limit,
    )

    if not results:
        yield {"type": "meta", "results": [], "context": "", "answer": None}
        yield {"type": "done"}
        return

    context = build_context(results, config.max_context_chars)
    if not context:
        yield {"type": "meta", "results": results, "context": "", "answer": None}
        yield {"type": "done"}
        return

    prompt = build_prompt(question, context)
    provider = build_provider(config)
    yield {"type": "meta", "results": results, "context": context, "answer": None}

    try:
        for chunk in provider.stream(prompt):
            if chunk:
                yield {"type": "chunk", "content": chunk}
    except Exception as exc:  # pragma: no cover - depends on runtime service state
        yield {"type": "error", "message": str(exc)}
    finally:
        yield {"type": "done"}


def log_results(results: List[Dict[str, Any]]) -> None:
    logging.info("Top %d results:", len(results))
    for result in results:
        logging.info("\n%s", format_result(result))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run a vector search query and answer with an LLM provider."
    )
    parser.add_argument("question", nargs="?", help="User question")
    parser.add_argument(
        "--question",
        dest="question_flag",
        help="User question (explicit flag form)",
    )
    parser.add_argument(
        "--debug-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    args = parser.parse_args()

    question = args.question_flag or args.question
    if args.debug_config:
        config = load_config()
        print(json.dumps(config.__dict__, indent=2))
        return
    if not question:
        raise SystemExit("Question is required. Provide it as an argument.")

    config = load_config()
    result = search(question, config)
    results = result["results"]
    if not results:
        logging.info("No results found for query.")
        return
    log_results(results)
    if not result["context"]:
        logging.info("No context built from results; skipping LLM response.")
        return
    response: Optional[str] = result["answer"]
    if response:
        print("\n---\nRAG Response\n---")
        print(response)


if __name__ == "__main__":
    main()
