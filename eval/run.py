import json
from pathlib import Path

from zoned_chatbot.ask import ask
from zoned_chatbot.retrieve import retrieve
from zoned_chatbot.config import TOP_K

ROOT = Path(__file__).resolve().parents[1]
KEEP = {"real-F09", "real-F10", "real-F11"}

rows = [
    json.loads(line)
    for line in (ROOT / "eval/golden_set.jsonl").read_text().splitlines()
    if line.strip()
]

out = []
for r in rows:
    if r["id"] not in KEEP and r["type"] != "refusal":
        continue
    hits = retrieve(r["question"], k=TOP_K)
    out.append({
        "id": r["id"],
        "type": r["type"],
        "question": r["question"],
        "expected": r.get("expected_answer"),
        "answer": ask(r["question"], hits),
        "retrieved_pages": [h["page"] for h in hits],
        "retrieved_docs": sorted({h["filename"] for h in hits}),
        "distances": [round(h["distance"], 4) for h in hits],
    })

(ROOT / "eval/baseline_phase1.json").write_text(json.dumps(out, indent=2))
print(f"wrote {len(out)} results")