from typing import Iterator


class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError
