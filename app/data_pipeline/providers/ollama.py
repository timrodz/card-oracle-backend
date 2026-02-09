import logging

from app.data_pipeline.providers.provider import LLMProvider


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, timeout: int, endpoint: str | None) -> None:
        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "ollama-python is required for LLM_PROVIDER=ollama"
            ) from exc

        try:
            if endpoint:
                self._client = ollama.Client(host=endpoint, timeout=timeout)
            else:
                self._client = ollama.Client(timeout=timeout)
        except TypeError:
            self._client = ollama.Client(host=endpoint) if endpoint else ollama.Client()
        self._model = model

    def generate(self, prompt: str) -> str:
        try:
            logging.debug(f"Generate:{prompt}")
            result = self._client.generate(model=self._model, prompt=prompt)
        except Exception as exc:  # pragma: no cover - depends on runtime service state
            raise RuntimeError(f"ollama generate failed: {exc}") from exc

        return result.response

    def stream(self, prompt: str):
        try:
            logging.debug(f"Stream:{prompt}")
            stream = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime service state
            raise RuntimeError(f"ollama chat stream failed: {exc}") from exc

        for chunk in stream:
            message = chunk.message
            text = message.content
            if text:
                yield text
