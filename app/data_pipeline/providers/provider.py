class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
