import logging
import os

import anthropic

from movie_subtitles.providers.errors import vendor_errors
from movie_subtitles.providers.prompt import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger("llm")

_MODEL = "claude-sonnet-5"


def _build_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it to your Anthropic API key to use the 'anthropic' translation engine."
        )

    return anthropic.Anthropic(api_key=api_key)


class LLMTranslate:
    def __init__(self, model: str = _MODEL) -> None:
        self.model = model
        self.client = _build_client()

    def __call__(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        return self.translate(text, output_lang, budget_chars)

    def translate(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        prompt = build_prompt(text, output_lang, budget_chars)

        with vendor_errors(anthropic.AnthropicError, "Anthropic translation request"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

        translation = next((b.text for b in response.content if b.type == "text"), "")
        return translation.strip()
