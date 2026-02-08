import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import ollama
from dotenv import load_dotenv
from embeddings import embed_text, load_embedder
from pymongo import MongoClient


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
    ollama_model: str
    max_context_chars: int
    ollama_timeout: int


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
        embed_model=os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-xsmall-v1"),
        embed_model_path=os.getenv(
            "EMBED_MODEL", "models/mixedbread-ai/mxbai-embed-xsmall-v1"
        ),
        normalize_embeddings=os.getenv("NORMALIZE_EMBEDDINGS", "true").lower()
        == "true",
        vector_index=os.getenv("VECTOR_INDEX_NAME", "vector_index"),
        vector_path=os.getenv("VECTOR_EMBED_PATH", "embeddings"),
        num_candidates=int(os.getenv("VECTOR_NUM_CANDIDATES", "100")),
        limit=int(os.getenv("VECTOR_LIMIT", "5")),
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
        max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000")),
        ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
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
                "name": 1,
                "type_line": 1,
                "oracle_text": 1,
                "chunk_text": 1,
                "chunk_index": 1,
                "set_name": 1,
                "mana_cost": 1,
                "cmc": 1,
                "rarity": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def format_result(result: Dict[str, Any]) -> str:
    name = result.get("name") or "Unknown"
    type_line = result.get("type_line") or ""
    set_name = result.get("set_name") or ""
    mana_cost = result.get("mana_cost") or ""
    score = result.get("score")
    chunk_text = result.get("chunk_text") or result.get("oracle_text") or ""
    return (
        f"- {name} {mana_cost} ({set_name})\n"
        f"  {type_line}\n"
        f"  score: {score:.4f}\n"
        f"  {chunk_text}"
    )


def build_context(results: List[Dict[str, Any]], max_chars: int) -> str:
    sections: List[str] = []
    total = 0
    for result in results:
        name = result.get("name") or "Unknown"
        type_line = result.get("type_line") or ""
        oracle_text = result.get("oracle_text") or ""
        chunk_text = result.get("chunk_text") or ""
        set_name = result.get("set_name") or ""
        mana_cost = result.get("mana_cost") or ""

        section = (
            f"Card: {name}\n"
            f"Set: {set_name}\n"
            f"Type: {type_line}\n"
            f"Mana Cost: {mana_cost}\n"
            f"Oracle: {oracle_text}\n"
            f"Chunk: {chunk_text}\n"
        )
        if total + len(section) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                sections.append(section[:remaining])
            break
        sections.append(section)
        total += len(section)
    return "\n".join(sections).strip()


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a helpful Magic: The Gathering rules assistant. Answer the question using the context below. If the context is insufficient, say so and suggest what to ask next.
Context: {context}
Question: {question}
"""


def run_ollama(model: str, prompt: str, timeout: int) -> str:
    host = os.getenv("OLLAMA_HOST")
    try:
        if host:
            client = ollama.Client(host=host, timeout=timeout)
        else:
            client = ollama.Client(timeout=timeout)
    except TypeError:
        client = ollama.Client(host=host) if host else ollama.Client()

    try:
        result = client.generate(model=model, prompt=prompt)
    except Exception as exc:  # pragma: no cover - depends on runtime service state
        raise RuntimeError(f"ollama generate failed: {exc}") from exc

    response = result["response"]
    if not response:
        raise RuntimeError("ollama generate returned no response")
    return response.strip()


def run_query(question: str, config: Config) -> None:
    embedder = load_embedder(config.embed_model, config.embed_model_path)
    query_vector = embed_query(embedder, question, config.normalize_embeddings)

    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    results = vector_search(
        collection,
        query_vector,
        config.vector_index,
        config.vector_path,
        config.num_candidates,
        config.limit,
    )

    if not results:
        logging.info("No results found for query.")
        return

    logging.info("Top %d results:", len(results))
    for result in results:
        logging.info("\n%s", format_result(result))

    context = build_context(results, config.max_context_chars)
    if not context:
        logging.info("No context built from results; skipping LLM response.")
        return

    prompt = build_prompt(question, context)
    response = run_ollama(config.ollama_model, prompt, config.ollama_timeout)
    print("\n---\nRAG Response\n---")
    print(response)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run a vector search query and answer with Ollama."
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
    run_query(question, config)


if __name__ == "__main__":
    main()
