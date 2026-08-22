from openai import OpenAI

from .config import EMBED_MODEL

_client = OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []

    for i in range(0, len(texts), 100):
        resp = _client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + 100])
        out.extend(d.embedding for d in resp.data)
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
