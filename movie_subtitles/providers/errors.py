"""Shared vendor-error-to-RuntimeError translation.

Every provider wraps its vendor SDK calls in a try/except that turns the vendor's own
exception type into a `RuntimeError` carrying a fixed "<what> failed: <exc>" message, so a
provider failure always reaches `cli.py`'s error-handling boundary as one of the exception
types it already catches. `vendor_errors()` is that pattern, factored out once so every
call site is greppable and the message shape cannot drift between providers.

Deliberately does not import any vendor SDK: each provider still names its own SDK's
exception type at its own call site (`ApiError`, `OpenAIError`, `anthropic.AnthropicError`,
...), passed in as `exc_type`. This module knows nothing about ElevenLabs, OpenAI or
Anthropic, matching the precedent set by `providers/base.py` and `providers/prompt.py` for
shared provider infrastructure that must not couple providers to each other.

For a lazily-streamed vendor response (e.g. `elevenlabs.py:Speak.speak`'s
`text_to_speech.convert`, `dubbing.py:ManagedDub._download`'s `dubbing.audio.get`), the
vendor exception may not actually be raised until the response is iterated -- the `with
vendor_errors(...):` block must span that iteration/write loop, not just the call that
returns the (still lazy) response.
"""

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def vendor_errors(exc_type: type[BaseException], what: str) -> Iterator[None]:
    """Re-raise any `exc_type` raised inside the block as `RuntimeError(f"{what} failed: ...")`.

    `what` should read as the start of a sentence naming the request/operation, e.g.
    "ElevenLabs TTS request" or "OpenAI translation request" -- the message becomes
    "{what} failed: {exc}". Only `exc_type` is caught; any other exception (e.g. a local
    `OSError` writing a response to disk) propagates unchanged, never relabelled as a
    vendor failure.
    """
    try:
        yield
    except exc_type as exc:
        raise RuntimeError(f"{what} failed: {exc}") from exc
