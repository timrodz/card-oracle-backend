import re
from typing import Optional

from pydantic import ValidationError

from app.core.config import llm_settings
from app.core.llms.ollama import OllamaProvider
from app.core.llms.provider import LLMProvider
from app.models.llm import LLMAnswerPayload, LLMSourceIdPayload


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_text(response: str) -> str | None:
    stripped = strip_code_fence(response)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return None


def parse_llm_response(response: str) -> tuple[str, Optional[str]]:
    json_text = extract_json_text(response)
    if not json_text:
        raise ValueError("LLM response must contain a JSON object")
    payload = LLMAnswerPayload.model_validate_json(json_text)
    return payload.answer.strip(), payload.source_id


def parse_source_id_response(response: str) -> Optional[str]:
    json_text = extract_json_text(response)
    if not json_text:
        raise ValueError("LLM source-id response must contain a JSON object")
    try:
        payload = LLMSourceIdPayload.model_validate_json(json_text)
    except ValidationError:
        # Some models return JSON-like objects with unquoted keys, e.g. { source_id: "..." }.
        normalized_json_text = re.sub(
            r"([{\s,])([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', json_text
        )
        payload = LLMSourceIdPayload.model_validate_json(normalized_json_text)
    return payload.source_id


def build_provider() -> LLMProvider:
    provider = llm_settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaProvider(
            llm_settings.llm_model,
            llm_settings.llm_timeout,
            llm_settings.llm_endpoint,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {llm_settings.llm_provider}")
