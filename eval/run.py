import json
from pathlib import Path

from zoned_chatbot.ask import ask

ROOT = Path(__file__).resolve().parents[1]

rows = [
    json.loads(line)
    for line in (ROOT / "eval/golden_set.jsonl").read_text().splitlines()
    if line.strip()
]

out = []
for r in rows:
    if r["type"] == "expected_fail":
        continue

    answer = ask(r["question"])
    hits = answer.hits
    out.append(
        {
            "id": r["id"],
            "type": r["type"],
            "question": r["question"],
            "expected": r.get("expected_answer"),
            "expected_docs": r.get("expected_docs"),
            "answer": answer.text,
            "retrieved_pages": [h["page"] for h in hits],
            "retrieved_docs": sorted({h["filename"] for h in hits}),
            "distances": [round(h["distance"], 4) for h in hits],
        }
    )

(ROOT / "eval/baseline_phase2.json").write_text(json.dumps(out, indent=2))
print(f"wrote {len(out)} results")
