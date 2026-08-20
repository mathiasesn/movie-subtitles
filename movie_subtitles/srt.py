_CUE_PAD = 0.5


def pad_cue_end(end: float, next_start: float | None) -> float:
    """Pad a cue's end time for readability, capped so it never overlaps the next cue."""
    padded = end + _CUE_PAD
    return padded if next_start is None else min(padded, next_start)


def format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_block(index: int, start: float, end: float, text: str) -> str:
    start_time = format_timestamp(start)
    end_time = format_timestamp(end)
    text = text[1:] if text and text[0] == " " else text

    return f"{index}\n{start_time} --> {end_time}\n{text}\n\n"
