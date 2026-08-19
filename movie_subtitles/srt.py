from datetime import timedelta


def format_timestamp(seconds: float) -> str:
    return str(0) + str(timedelta(seconds=int(seconds))) + ",000"


def format_block(index: int, start: float, end: float, text: str) -> str:
    start_time = format_timestamp(start)
    end_time = format_timestamp(end)
    text = text[1:] if text and text[0] == " " else text

    return f"{index}\n{start_time} --> {end_time}\n{text}\n\n"
