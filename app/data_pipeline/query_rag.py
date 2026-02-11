import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from pydantic import ValidationError
from pymongo import MongoClient

from app.data_pipeline.providers.ollama import OllamaProvider
from app.data_pipeline.providers.provider import LLMProvider
from app.data_pipeline.sentence_transformers import (
    embed_text,
    load_transformer,
)
from app.models.api import SearchResponse, SearchResult
from app.settings import get_settings


@dataclass
class Config:
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    embed_model_name: str
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
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    return Config(
        mongodb_uri=settings.mongodb_uri,
        mongodb_db=settings.mongodb_db,
        mongodb_collection=settings.mongodb_collection_embeddings,
        embed_model_name=settings.embed_model_name,
        embed_model_path=settings.embed_model_path,
        normalize_embeddings=settings.normalize_embeddings,
        vector_index=settings.vector_index_name,
        vector_path=settings.vector_embed_path,
        num_candidates=settings.vector_num_candidates,
        limit=settings.vector_limit,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_endpoint=settings.llm_endpoint,
        max_context_chars=settings.rag_max_context_chars,
        llm_timeout=settings.llm_timeout,
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def embed_query(*, model, question: str, normalize: bool) -> List[float]:
    return embed_text(model, question, normalize)


def vector_search(
    *,
    collection,
    query_vector: List[float],
    vector_index: str,
    vector_path: str,
    num_candidates: int,
    limit: int,
) -> List[SearchResult]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "queryVector": query_vector,
                "path": vector_path,
                "numCandidates": num_candidates,
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
    raw_results = list(collection.aggregate(pipeline))
    results: List[SearchResult] = []
    for raw_result in raw_results:
        try:
            results.append(SearchResult.model_validate(raw_result))
        except ValidationError:
            logging.warning("Skipping invalid search result: %s", raw_result)
    logging.info("results: %s", [result.model_dump() for result in results])
    return results


def format_result(result: SearchResult) -> str:
    return result.summary


def build_context(
    *, results: List[SearchResult], max_chars: int, include_source_ids: bool
) -> str:
    sections: List[str] = []
    total = 0
    for result in results:
        header = ""
        if include_source_ids:
            source_id = result.source_id
            header = f"source_id: {source_id}\n" if source_id else ""
        section = f"{header}{result.summary}"
        if total + len(section) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                sections.append(section[:remaining])
            break
        sections.append(section)
        total += len(section)
    return "\n".join(sections).strip()


def build_prompt(*, question: str, context: str, require_json: bool) -> str:
    instructions = (
        "Answer the question using the provided context. "
        "If the context is insufficient, say so and suggest what to ask next."
    )
    if require_json:
        instructions = f"""{instructions}
            If you can confidently pinpoint a single specific card from the context,
            include its source_id in the response. Only use source_id values that
            appear in the context. If not confident, set source_id to null.
            Return only JSON with keys: answer (string), source_id (string|null)
            """
    payload = {
        "role": "You help users find cards for Magic: The Gathering.",
        "instructions": instructions,
        "context": context,
        "question": question,
    }
    logging.debug(payload)
    return json.dumps(payload)


def build_source_id_prompt(*, question: str, context: str, answer: str) -> str:
    payload = {
        "role": "You identify the best matching card from provided context.",
        "instructions": (
            """
            Given the question, context, and answer, choose the single best
            source_id if the answer clearly refers to one card.
            Only use source_id values that appear in the context.
            If not confident, set source_id to null.
            Return only JSON with key: source_id (string|null).
            Example response: { source_id: "77c6fa74-5543-42ac-9ead-0e890b188e99" }
            """
        ),
        "context": context,
        "question": question,
        "answer": answer,
    }
    logging.debug(payload)
    return json.dumps(payload)


def build_provider(config: Config) -> LLMProvider:
    provider = config.llm_provider.lower()
    if provider == "ollama":
        return OllamaProvider(config.llm_model, config.llm_timeout, config.llm_endpoint)
    raise ValueError(f"Unsupported LLM_PROVIDER: {config.llm_provider}")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_text(response: str) -> Optional[str]:
    stripped = _strip_code_fence(response)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return None


def parse_llm_response(response: str) -> tuple[str, Optional[str]]:
    candidates = [response]
    json_text = _extract_json_text(response)
    if json_text:
        candidates.append(json_text)
    candidates.append(response.replace('\\"', '"'))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            answer = payload.get("answer")
            source_id = payload.get("source_id")
            if isinstance(answer, str):
                return answer.strip(), source_id if isinstance(source_id, str) else None
    text = re.sub(r"\s+", " ", response).strip()
    return text, None


def parse_source_id_response(response: str) -> Optional[str]:
    candidates = [response]
    json_text = _extract_json_text(response)
    if json_text:
        candidates.append(json_text)
    candidates.append(response.replace('\\"', '"'))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            source_id = payload.get("source_id")
            if isinstance(source_id, str):
                return source_id
            if source_id is None:
                return None
    return None


def _dump_results(results: List[SearchResult]) -> List[Dict[str, Any]]:
    return [result.model_dump() for result in results]


def search(*, question: str, config: Config) -> Dict[str, Any]:
    embedder = load_transformer(config.embed_model_name, config.embed_model_path)
    query_embeddings = embed_query(
        model=embedder,
        question=question,
        normalize=config.normalize_embeddings,
    )

    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    results = vector_search(
        collection=collection,
        query_vector=query_embeddings,
        vector_index=config.vector_index,
        vector_path=config.vector_path,
        num_candidates=config.num_candidates,
        limit=config.limit,
    )

    if not results:
        return SearchResponse(
            results=[],
            context="",
            answer=None,
            source_id=None,
        ).model_dump()

    context = build_context(
        results=results,
        max_chars=config.max_context_chars,
        include_source_ids=True,
    )
    if not context:
        return SearchResponse(
            results=results,
            context="",
            answer=None,
            source_id=None,
        ).model_dump()

    prompt = build_prompt(question=question, context=context, require_json=True)
    provider = build_provider(config)
    response = provider.generate(prompt)
    clean_response, source_id = parse_llm_response(response)
    return SearchResponse(
        results=results,
        context=context,
        answer=clean_response,
        source_id=source_id,
        answer_raw=response,
    ).model_dump()


def search_stream(*, question: str, config: Config) -> Iterator[Dict[str, Any]]:
    embedder = load_transformer(config.embed_model_name, config.embed_model_path)
    query_embeddings = embed_query(
        model=embedder,
        question=question,
        normalize=config.normalize_embeddings,
    )

    client: MongoClient = MongoClient(config.mongodb_uri)
    collection = client[config.mongodb_db][config.mongodb_collection]

    results = vector_search(
        collection=collection,
        query_vector=query_embeddings,
        vector_index=config.vector_index,
        vector_path=config.vector_path,
        num_candidates=config.num_candidates,
        limit=config.limit,
    )
    serialized_results = _dump_results(results)

    if not results:
        yield {"type": "meta", "results": [], "context": ""}
        yield {"type": "done"}
        return

    context = build_context(
        results=results,
        max_chars=config.max_context_chars,
        include_source_ids=False,
    )
    if not context:
        yield {"type": "meta", "results": serialized_results, "context": ""}
        yield {"type": "done"}
        return

    prompt = build_prompt(question=question, context=context, require_json=False)
    provider = build_provider(config)
    yield {"type": "meta", "results": serialized_results, "context": context}

    streamed_parts: List[str] = []
    had_error = False
    try:
        for chunk in provider.stream(prompt):
            streamed_parts.append(chunk)
            yield {"type": "chunk", "content": chunk}
    except Exception as exc:  # pragma: no cover - depends on runtime service state
        had_error = True
        yield {"type": "error", "message": str(exc)}
    finally:
        if streamed_parts and not had_error:
            full_response = "".join(streamed_parts)
            source_context = build_context(
                results=results,
                max_chars=config.max_context_chars,
                include_source_ids=True,
            )
            source_prompt = build_source_id_prompt(
                question=question,
                context=source_context,
                answer=full_response,
            )
            try:
                yield {"type": "seeking_card"}
                source_response = provider.generate(source_prompt)
                source_id = parse_source_id_response(source_response)
            except Exception as exc:  # pragma: no cover - depends on runtime service
                yield {"type": "error", "message": str(exc)}
                yield {"type": "done"}
                return
            if source_id:
                yield {"type": "found_card", "id": source_id}
        yield {"type": "done"}


def log_results(results: List[SearchResult]) -> None:
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
    result = search(question=question, config=config)
    results = [SearchResult.model_validate(item) for item in result["results"]]
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
