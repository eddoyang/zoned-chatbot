import sys

from anthropic import Anthropic

from .retrieve import retrieve

client = Anthropic()

SYSTEM = (
    "Answer the question using only the provided document excerpts. "
    "Cite the document title and page for each claim. "
    "If the excerpts do not contain the answer, say so plainly."
)

def ask(question: str) -> str:
    hits = retrieve(question, k=5)

    context = "\n\n---\n\n".join(
        f"[{h['title']}, p.{h['page']}]\n{h['content']}" for h in hits
    )

    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"<excerpts>\n{context}\n</excerpts>\n\nQuestion: {question}",
        }],
    )

    return msg.content[0].text


def main() -> None:
    print(ask(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()