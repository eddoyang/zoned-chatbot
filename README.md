# Zoned

A Retrieval-augmented generation (RAG) that answers questions grounded solely in a set of user uploaded documents, with an explicit refusal when the answer isn't there.

--- 
## Vision
The primary goal is reducing hallucination: a system that answers only from the documents it's given, and says "not in the documents" instead of outputting something plausible when the answer isn't even there. Handing an LLM a question and letting it answer from memory allows it to confidently make things up; LLM's are trained to be helpful, and "I don't know" fights that instinct. The goal here is to close that gap: upload documents, ask questions, get answers grounded solely in those documents, with a real and reliable refusal. 

Retrieval is implemented in raw SQL over PostgreSQL + pgvector, not hidden behind a framework or an ORM. Implementing the retrieval layer directly in SQL instead of reaching for a library allows me to specifically build everything into one Postgres database, allowing me to debug retrieval and grounding problems directly on my own terms. Layer by layer, this will eventually allow me to build Zoned into a hybrid-search RAG system with citation verification and cross-encoder reranking. Raw SQL is also a deliberate architectural choice for this project, allowing me to build real SQL fluency.

## Tech stack

This is the *current* tech stack as of project state

**Core** — Python 3.13 · PostgreSQL 16 + pgvector · Docker Compose

**Retrieval** — raw SQL via psycopg 3 (no ORM), HNSW index with cosine
distance, `tsvector`/GIN full-text column for eventual hybrid search

**Ingestion** — Docling (PDF → text with page provenance) · tiktoken ·
OpenAI `text-embedding-3-small` (1536d)

**Generation** — Anthropic Messages API (Claude)

**Schema** — Alembic migrations, raw DDL, reversible

**Tooling** — uv · ruff · pytest

## Phases - TBF

| Phase | Goal | Status |
| --- | --- | --- |
| 0 | Corpus, golden sets, environment | Done |
| 1 | Skeleton: one PDF in, one right answer out, with a real schema | Done |
| 2 | Multi-document retrieval, idempotent ingestion, attribution | In Progress |
| 3 | Hybrid search, reranking, contextual retrieval | Planned |
| 4 | Grounding and reliable refusal (retrieval floor, citations, span verification) | Planned |


## Phase 1 Baseline
```
2026-08-20, git 145eecd
- config: text-embedding-3-small/1536, 800-token chunks, 100 overlap, dense top-5, no rerank
- corpus: rag_lewis.pdf only
- factual (3):   2/3 correct
- refusal (10):  10/10 correctly refused
```
#### Diagnosis: 
- The missed factual question was a precision problem; Correct chunks were retrieved, but all answers were ranked outside the top 5 window.
- All cosine distances were relatively high, with every answer greater than 0.40. 
- Every refusal answer started with a variation of "I cannot answer" or "The excerpt does not contain...", and then a short explanation of what the excerpt actually contains.
- Refusal answers gives us page numbers, however generally being random and within the document size.
- Refusal score is not yet a meaningful number. Expected to drop once given more documents.


#### Factual Results
```
F09: FAIL at k=5. Answer chunks rank 6, 8, 12, 15 of 25.
Top-5 distances 0.42-0.49, no separation — chunk dilution at 800 tokens.
Recall is fine, ranking is the problem.
Passes at k=10.

F10: PASS at k=5. Answer chunks rank 3, 4, 10, 15 of 25.
Top-5 distances 0.47-0.55, no separation — chunk dilution at 800 tokens.
Precision is okay, however distances are still far and relatively inconclusive.

F11: PASS at k=5. Answer chunks rank 3, 4, 5, 17 of 25.
Top-5 distances 0.42-0.49, no separation — chunk dilution at 800 tokens.
Precision is great, distances are still far — generally inconclusive.
```

## Getting started
```bash
# 1. Start the database
docker compose up -d
docker compose ps          # should read "healthy"

# 2. Sync the Python environment
uv sync

# 3. Apply the schema
uv run alembic upgrade head

# 4. Ingest a document
uv run python -m zoned_chatbot.ingest corpus/rag_lewis.pdf

# 5. Ask a question
uv run zoned-chatbot "What generator model does RAG use?"

# 6. Run the eval baseline
uv run python eval/run.py
```






