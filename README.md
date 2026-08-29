# Zoned

A Retrieval-augmented generation (RAG) that answers questions grounded solely in a set of user uploaded documents, with an explicit refusal when the answer isn't there.

--- 
## Vision
The primary goal of Zoned is reducing hallucination: A system that answers only from the documents it's given, and says "not in the documents" instead of outputting something plausible when the answer isn't even there. Handing an LLM a question and letting it answer from memory allows it to confidently make things up; LLM's are trained to be helpful, and "I don't know" fights that instinct. In order to stop this instinct, the retrieval must be grounded and verified before outputting any answers.

Retrieval is implemented in raw SQL over PostgreSQL + pgvector, not hidden behind a framework or an ORM. Implementing the retrieval layer directly in SQL instead of reaching for a library enables me to specifically build everything into one Postgres database, allowing me to debug retrieval and grounding problems directly how I want to. Layer by layer, this will eventually let me to build Zoned into a hybrid-search RAG system with citation verification and cross-encoder reranking. Raw SQL is also a deliberate architectural choice for this project, allowing me to build SQL fluency.

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
| 1 | Skeleton: input one PDF, output one answer, a real schema | Done |
| 2 | Multi-document retrieval, idempotent ingestion, attribution | Done |
| 3 | Hybrid search, reranking, contextual retrieval | In-progress |
| 4 | Grounding and reliable refusal (retrieval floor, citations, span verification) | Planned |
| 5 |  | TBD |


## Golden Set Evaluation
Golden set contains 4 different question types
- **factual**: Questions that can be answered from sources in the corpora.
- **which_document**: Questions that test the retrievals ability to select the correct source document before the correct passage
- **refusal**: Questions that currently can _not_ be answered from sources in the corpora, and should be refused.
- **expected_fail**: Questions that the RAG system is currently unable to answer due to current limited parsing capabilities; Currently excluded in evaluations.

## Phase 2

### Legend
**attribution**:
- @1 represents the top hit containing the correct document
- @k represents the top k hits containing the correct document
- Only evaluated on answerable questions
  
**which_document**: 
- top hit contains the correct document for which_document questions

**factual**:
- correct - questions were answered correctly
- partial - questions were answered partly, but stated that the remainder was not in the excerpts
- wrong refusal - questions that are answerable but were refused

**prose-sourced**: 
- golden-set answers appear in running text.

**table/chart sourced**: 
- golden-set answers appear in tables/charts.


### 
  **Mechanical Evaluation**
| Metric                          | Result    |
| ------------------------------- | --------- |
| attribution@1 (answerable)      | **19/23** |
| attribution@k (answerable)      | **23/23** |
| which_document: correct top hit | **6/6**   |
| expected_page in hits           | **8/23**  |
| factual: correct                | **11/17** |
| factual: partial                | **4/17**  |
| factual: wrongly refused        | **2/17**  |
| correct refusals                | **10/10** |

**Source Type Evaluation**
|                              | attribution@1 | expected_page | answered correctly |
| -----------------------------| ------------- | ------------- | ------------------ |
|**prose-sourced (13)**        |    11/13      |     7/13      |       9/13         |
| **table/chart-sourced (10)** |    8/10       |     1/10      |       8/10         |

**Top-1 distance distribution**
|                | min   | median | max   |
| -------------- | ----- | ------ | ----- |
| **answerable** | 0.219 | 0.331  | 0.486 |
| **refusal**    | 0.270 | 0.414  | 0.540 |


### Diagnosis:
- expected_page in hits is an unreliable metric. Page records where a chunk starts, so any fact in the back half of a chunk is filed under the previous page.
- attribution@k is perfect and attribution@1 is 83%; Retrieval is able to find the right documents.
- attribution@1 misses are all within an overlapping pair of documents. Lexically near-identical documents are harder to distinguish.
- expected_pages drops heavily for tables/charts/figures. Given the metric's unreliability, gap is not evidence that parsing is blocking answers.
- There are zero fabricated answers across the 23 answerable questions. There was one hallucination that inferred a relationship between two real figures, which would pass with span verification.
- There is no document bias. NVDA FY2026 contributed 36 retrieved chunks against NVDA FY2025's 34. Ranked first six times against eight, respectively.
- For all hit sets, there were one distinct document 10 times and two distinct documents 23 times, never more. Retrieval was able to find the relevant document(s) for this corpus.
- PER_DOC_CAP = 3 caused false refusals by neglecting depth in single documents. Increasing the cap to 5 and TOP_K = 5 turned off the cap, fixing the problem for a corpus of 6 documents due to the lower diversity. Likely need to lower the cap when more documents are added to the corpus.
- Top 1 distance distributions for answerable and refusal questions overlap badly. At this stage, a retrieval-floor threshold won't work.
- Corpus has lots of boilerplate. About 56% - 96% of chunks in every long document, which caused chunk dilution.
- Correct refusals stayed at 10/10, and did not drop as predicted after multi-document ingestion was introduced.
- At least one two-column page was extracted interleaved, so chunks contain broken sentences interleaved with an unrelated one. 

#### Question Diagnosis
- F03: FY2024 graphics revenue also appears in FY2026 p.77. Golden set was updated to include the document.
- F04: Correct chunk was not in hit set, but recovered at k=10. Chunk sat below both copies of the wrong year's rankings.
- F08: Fact was buried in a generally unrelated chunk, and chunk was never retrieved at any k.
- F16: Answered correctly from a chunk spanning pages 160-161; scored as an expected_page miss because page records the chunk's start.
- R06: Correctly declined to give a 2025 frontier-model training cost, but then offered the 2024 Llama figure with the year clearly stated. Graded as correct refusal.
- W02: Gave the right answer, and then made an incorrect relationship between two figures from different years in the same document.
- F05, F06, F07, F14: Partials; Each answered the retrieved part and explicitly stated the rest wasn't in the excerpts.

## Phase 1
### Baseline
```
2026-08-20, git 145eecd
- config: text-embedding-3-small/1536, 800-token chunks, 100 overlap, dense top-5, no rerank
- corpus: rag_lewis.pdf only
- factual (3):   2/3 correct
- refusal (10):  10/10 correctly refused
```
### Diagnosis: 
- The missed factual question was a precision problem; Correct chunks were retrieved, but all answers were ranked outside the top 5 window.
- All cosine distances were relatively high, with every answer greater than 0.40.
- The spread of the top 5 hits across all three factual questions were 0.42-0.55.
- Every refusal answer started with a variation of "I cannot answer" or "The excerpt does not contain...", and then a short explanation of what the excerpt actually contains.
- Refusal answers gives us page numbers, however generally being random and within the document size.
- Refusal score is not yet a meaningful number. Expected to drop once given more documents.


### Factual Results
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






