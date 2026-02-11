import argparse
import json
from typing import Any, Iterator, List

from loguru import logger

from app.core.config import llm_settings, transformer_settings
from app.core.db import database
from app.core.llms.utils import (
    build_provider,
    parse_llm_response,
    parse_source_id_response,
)
from app.data_pipeline.sentence_transformers import (
    embed_text,
    load_transformer,
)
from app.models.api import (
    SearchResponse,
    SearchResult,
    StreamChunkEvent,
    StreamDoneEvent,
    StreamFoundCardEvent,
    StreamMetaEvent,
    StreamSeekingCardEvent,
)
from app.models.embedding import CardEmbeddingVectorSearchResult


class RagSearch:
    def __vector_search(
        self,
        query_vector: List[float],
    ) -> List[SearchResult]:
        vector_limit = transformer_settings.vector_limit

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_vector,
                    "path": "embeddings",
                    "exact": True,
                    "limit": vector_limit,
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

        raw_results = list(database.embeddings_collection.aggregate(pipeline))
        results: List[SearchResult] = []
        for raw_result in raw_results:
            embedding_record = CardEmbeddingVectorSearchResult.model_validate(
                raw_result
            )
            results.append(
                SearchResult(
                    source_id=embedding_record.source_id,
                    summary=embedding_record.summary,
                    score=embedding_record.score,
                )
            )
        logger.info("results: %s", [result.model_dump() for result in results])
        return results

    def __build_context(
        self, *, results: List[SearchResult], max_chars: int, include_source_ids: bool
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

    def __build_prompt(self, *, question: str, context: str, require_json: bool) -> str:
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
        logger.debug(payload)
        return json.dumps(payload)

    def __build_source_id_prompt(
        self, *, question: str, context: str, answer: str
    ) -> str:
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
        logger.debug(payload)
        return json.dumps(payload)

    def search(self, question: str) -> SearchResponse:

        embedder = load_transformer()
        query_embeddings = embed_text(
            model=embedder,
            text=question,
            normalize=transformer_settings.normalize_embeddings,
        )
        results = self.__vector_search(
            query_vector=query_embeddings,
        )

        if not results:
            return SearchResponse(
                results=[],
                context="",
                answer=None,
                source_id=None,
            )

        context = self.__build_context(
            results=results,
            max_chars=llm_settings.rag_max_context_chars,
            include_source_ids=True,
        )
        if not context:
            return SearchResponse(
                results=results,
                context="",
                answer=None,
                source_id=None,
            )

        prompt = self.__build_prompt(
            question=question, context=context, require_json=True
        )
        provider = build_provider()
        response = provider.generate(prompt)
        clean_response, source_id = parse_llm_response(response)
        return SearchResponse(
            results=results,
            context=context,
            answer=clean_response,
            source_id=source_id,
            answer_raw=response,
        )

    def search_stream(self, question: str) -> Iterator[dict[str, Any]]:
        embedder = load_transformer()
        query_embeddings = embed_text(
            model=embedder,
            text=question,
            normalize=transformer_settings.normalize_embeddings,
        )

        results = self.__vector_search(
            query_vector=query_embeddings,
        )
        if not results:
            yield StreamMetaEvent(type="meta", results=[], context="").model_dump(
                exclude_none=True
            )
            yield StreamDoneEvent(type="done").model_dump(exclude_none=True)
            return

        context = self.__build_context(
            results=results,
            max_chars=llm_settings.rag_max_context_chars,
            include_source_ids=False,
        )
        if not context:
            yield StreamMetaEvent(type="meta", results=results, context="").model_dump(
                exclude_none=True
            )
            yield StreamDoneEvent(type="done").model_dump(exclude_none=True)
            return

        prompt = self.__build_prompt(
            question=question, context=context, require_json=False
        )
        provider = build_provider()
        yield StreamMetaEvent(type="meta", results=results, context=context).model_dump(
            exclude_none=True
        )

        streamed_parts: List[str] = []
        for chunk in provider.stream(prompt):
            streamed_parts.append(chunk)
            yield StreamChunkEvent(type="chunk", content=chunk).model_dump(
                exclude_none=True
            )

        if streamed_parts:
            full_response = "".join(streamed_parts)
            source_context = self.__build_context(
                results=results,
                max_chars=llm_settings.rag_max_context_chars,
                include_source_ids=True,
            )
            source_prompt = self.__build_source_id_prompt(
                question=question,
                context=source_context,
                answer=full_response,
            )
            yield StreamSeekingCardEvent(type="seeking_card").model_dump(
                exclude_none=True
            )
            source_response = provider.generate(source_prompt)
            source_id = parse_source_id_response(source_response)
            if source_id:
                yield StreamFoundCardEvent(type="found_card", id=source_id).model_dump(
                    exclude_none=True
                )
        yield StreamDoneEvent(type="done").model_dump(exclude_none=True)


def main() -> None:
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
    if not question:
        raise SystemExit("Question is required. Provide it as an argument.")

    rag_search = RagSearch()
    result = rag_search.search(question)
    logger.info(f"\n---\nRAG Response\n---\n{result.answer}")


if __name__ == "__main__":
    main()
