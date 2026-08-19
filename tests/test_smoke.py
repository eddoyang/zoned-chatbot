from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_corpus_exists():
    pdfs = list((ROOT /"corpus").glob("*.pdf"))
    assert len(pdfs) >= 6, f"expected 6+ PDFs, found {len(pdfs)}"



def test_golden_set_is_valid_jsonl():
    import json
    path = ROOT / ("eval/golden_set.jsonl")
    
    if path.stat().st_size == 0:
        return                      # set not written
    
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f"line {i} is not valid JSON: {e}") from e
            