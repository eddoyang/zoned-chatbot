# Zoned
*A document Q&A chatbot that stays inside the lines.*

Upload your PDFs and ask questions. Zoned retrieves the relevant passages, answers from those passages only, and cites the document and page behind every claim. Ask it something your documents don't cover and it returns nothing rather than something plausible.

Most RAG systems treat "I don't know" as a failure state. 
Zoned treats it as a feature. 
Refusal accuracy is measured in the eval suite alongside answer quality.


## Tech stack

**Core** — Python 3.13 · PostgreSQL 16 + pgvector · Docker Compose

**Retrieval** — raw SQL via psycopg 3 (no ORM), HNSW index with cosine
distance, `tsvector`/GIN full-text column for hybrid search

**Ingestion** — Docling (PDF → text with page provenance) · tiktoken ·
OpenAI `text-embedding-3-small` (1536d)

**Generation** — Anthropic Messages API (Claude)

**Schema** — Alembic migrations, raw DDL, reversible

**Tooling** — uv · ruff · pytest









## CORPORA
I have 2 folders for both my corpora and evals. The ones ending with "_gen" are fictional, AI generated documents that are used solely for testing. The other corpus/eval contain real documents, which are also used for testing, but provide more realistic documents to use. I have both of these to help with testing and debugging the model.

Current corpus release: corpus-v1





# Phase 1
Phase 1 baseline — 2026-08-20, git <145eecd>
- config: text-embedding-3-small/1536, 800-token chunks, 100 overlap, dense top-5, no rerank
- corpus: rag_lewis.pdf only
- factual (3):   2/3 correct
- refusal (10):  10/10 correctly refused


### Notes: 
The correct factual questions had at 2-3 of the correct answers in the top 5.
All cosine distances were relatively high, with every answer greater than 0.40.

Every refusal answer started with a variation of "I cannot answer" or "The excerpt does not contain...", and then a short explanation of what the excerpt actually contains. It gives us page numbers, however generally being randomly chosen within the document. Expected to drop once given full corpus. 


Factual Results:

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
