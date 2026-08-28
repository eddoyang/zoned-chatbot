import json
from pathlib import Path

from zoned_chatbot.ask import ask
from zoned_chatbot.config import (
    CHUNK_TOKENS,
    EMBED_MODEL,
    PER_DOC_CAP,
    POOL_SIZE,
    TOP_K,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/baseline_phase2_v3.json"


CONFIG = {
    "embed_model": EMBED_MODEL,
    "chunk_tokens": CHUNK_TOKENS,
    "top_k": TOP_K,
    "per_doc_cap": PER_DOC_CAP,
    "pool_size": POOL_SIZE,
}

rows = [
    json.loads(line)
    for line in (ROOT / "eval/golden_set.jsonl").read_text().splitlines()
    if line.strip()
]

out = []
for r in rows:
    if r["type"] == "expected_fail":
        continue


    rec = {
        "id": r["id"],
        "type": r["type"],
        "question": r["question"],
        "expected": r.get("expected_answer"),
        "expected_docs": r.get("expected_docs"),
        "expected_page": r.get("expected_page"),
    }

    try:
        answer = ask(r["question"])
        rec["answer"] = answer.text
        rec["hits"] = [
            {"doc": h["filename"], "page": h["page"],
            "distance": round(h["distance"], 4), "chunk_id": h["id"]}
            for h in answer.hits
        ]
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        print(f"failed {r['id']}: {rec['error']}")

    out.append(rec)
    OUT.write_text(json.dumps({"config": CONFIG, "results": out}, indent=2))


errors = sum(1 for r in out if "error" in r)
if errors > len(out) // 2:
    raise SystemExit(f"{errors}/{len(out)} rows failed — refusing to write {OUT.name}")

OUT.write_text(json.dumps({"config": CONFIG, "results": out}, indent=2))
print(f"wrote {len(out)} results, {errors} errors — cap={PER_DOC_CAP}, k={TOP_K}")
