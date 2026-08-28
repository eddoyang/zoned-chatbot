import sys
from dataclasses import dataclass

from anthropic import Anthropic

from .config import TOP_K
from .retrieve import retrieve

client = Anthropic()

SYSTEM = (
    "Answer the question using only the provided document excerpts. "
    "Cite the document title and page for each claim. "
    "If the excerpts do not contain the answer, say so plainly. "
    "When excerpts from different documents give different answers, "
    "report both with their sources rather than choosing between them."
)


@dataclass
class Answer:
    text: str
    hits: list[dict]


def ask(question: str, hits: list[dict] | None = None) -> Answer:
    if hits is None:
        hits = retrieve(question, k=TOP_K)

    context = "\n\n---\n\n".join(
        f"[{h['title']} | {h['filename']} | p.{h['page']}]\n{h['content']}"
        for h in hits
    )

    msg = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<excerpts>\n{context}\n</excerpts>\n\nQuestion: {question}",
            }
        ],
    )

    return Answer(msg.content[0].text, hits)


def main() -> None:
    answer = ask(" ".join(sys.argv[1:]))
    print(answer.text)


if __name__ == "__main__":
    main()
