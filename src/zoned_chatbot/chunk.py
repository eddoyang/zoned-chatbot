from dataclasses import dataclass

import tiktoken

from .config import CHUNK_OVERLAP, CHUNK_TOKENS
from .parse import ParsedDoc, page_at

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    index: int
    content: str
    char_start: int
    char_end: int
    page: int | None


def chunk(parsed: ParsedDoc) -> list[Chunk]:
    tokens = _enc.encode(parsed.text)
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    out: list[Chunk] = []

    for i, start_tok in enumerate(range(0, len(tokens), step)):
        window = tokens[start_tok : start_tok + CHUNK_TOKENS]

        if not window:
            break
        content = _enc.decode(window)
        char_start = len(_enc.decode(tokens[:start_tok]))
        char_end = char_start + len(content)
        out.append(Chunk(i, content, char_start, char_end, page_at(parsed, char_start)))

        if start_tok + CHUNK_TOKENS >= len(tokens):
            break

    return out
