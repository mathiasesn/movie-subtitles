"""Prompt text shared by the LLM-backed translation providers.

`LLMTranslate` (Anthropic) and `OpenAITranslate` must ask for the same thing in the
same words, otherwise `--translation-engine anthropic` and `--translation-engine
openai` quietly diverge in output style and the two are no longer comparable — which
is the whole point of having both. Kept free of vendor SDK imports so importing it
never pulls in a client library.
"""

SYSTEM_PROMPT = (
    "You are a subtitle translator. Translate the given text into the requested "
    "language. Respond with only the translation, no preamble, no explanation, "
    "no quotation marks."
)


def build_prompt(text: str, output_lang: str, budget_chars: int | None = None) -> str:
    """Build the user prompt, appending the length budget when one is given.

    The budget clause is what makes timing-drift fitting possible at all (step 1 of the
    drift strategy), so it is shared rather than copied per provider.
    """
    prompt = f"Translate the following text into '{output_lang}':\n\n{text}"
    if budget_chars is not None:
        prompt += (
            f"\n\nThe translation must fit within roughly {budget_chars} characters. "
            "Stay as faithful as possible to the meaning while shortening phrasing, "
            "dropping filler, or rewording as needed to hit that budget."
        )

    return prompt
