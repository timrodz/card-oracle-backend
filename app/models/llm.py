from pydantic import BaseModel


class LLMAnswerPayload(BaseModel):
    answer: str
    source_id: str | None = None


class LLMSourceIdPayload(BaseModel):
    source_id: str | None = None
