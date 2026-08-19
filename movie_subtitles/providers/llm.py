import logging
import os

import anthropic

logger = logging.getLogger("llm")

_MODEL = "claude-sonnet-5"

_SYSTEM_PROMPT = (
    "You are a subtitle translator. Translate the given text into the requested "
    "language. Respond with only the translation, no preamble, no explanation, "
    "no quotation marks."
)


def _build_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it to your Anthropic API key to use the 'elevenlabs' engine."
        )

    return anthropic.Anthropic(api_key=api_key)


class LLMTranslate:
    def __init__(self, model: str = _MODEL) -> None:
        self.model = model
        self.client = _build_client()

    def __call__(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        return self.translate(text, output_lang, budget_chars)

    def translate(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        prompt = f"Translate the following text into '{output_lang}':\n\n{text}"
        if budget_chars is not None:
            prompt += (
                f"\n\nThe translation must fit within roughly {budget_chars} characters. "
                "Stay as faithful as possible to the meaning while shortening phrasing, "
                "dropping filler, or rewording as needed to hit that budget."
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        translation = next((b.text for b in response.content if b.type == "text"), "")
        return translation.strip()
