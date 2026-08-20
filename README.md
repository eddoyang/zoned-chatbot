# Zoned
*A document Q&A chatbot that stays inside the lines.*

Upload your PDFs and ask questions. Zoned retrieves the relevant passages, answers from those passages only, and cites the document and page behind every claim. Ask it something your documents don't cover and it returns nothing rather than something plausible.

Most RAG systems treat "I don't know" as a failure state. 
Zoned treats it as a feature. 
Refusal accuracy is measured in the eval suite alongside answer quality.



## CORPUSES
I have 2 folders for both my corpuses and evals. The ones ending with "_gen" are fictional, AI generated documents that are used solely for testing. The other corpuses/evals are real documents, which are also used for testing, but provide more realistic documents to use. I have both of these to help with testing and debugging the model.

Current corpus release: corpus-v1





# Phase 1
Phase 1 baseline — 2026-08-20, git <SHA>
config: text-embedding-3-small/1536, 800-token chunks, 100 overlap, dense top-5, no rerank
corpus: rag_lewis.pdf only
factual (3):   2/3 correct
refusal (10):  10/10 correctly refused
notes: 

The correct factual questions had at 2-3 of the correct answers in the top 5.
All euclidean distances were relatively high, with every answer greater than 0.40.

Every refusal answer started with a variation of "I can't not answer" or "The excerpt does not contai...", and then a short explaination of what the excerpt actaully contains. It gives us page numbers, however generally being randomly chosen within the document. Expected to drop once given full corpus. 