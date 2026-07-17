# VeritasRAG 🔍
### Legal Document Q&A with Hallucination Detection
> *Veritas* — Latin for **truth**. VeritasRAG doesn't just retrieve answers from documents. It verifies them.


## What This Is

Most RAG systems stop at generating an answer. VeritasRAG goes one step further — after every answer, it automatically scores how faithful that answer is to the source document, catching hallucinations before they reach the user.

Built from scratch on a real-world Master SaaS Agreement (Armorblox/Cisco), without copying from tutorials.


## Architecture

```
PDF
 └─► Text cleaning + semantic chunking
      └─► BGE-large-en-v1.5 embeddings → ChromaDB

Query
 ├─► BM25 keyword search     ─┐
 └─► BGE-large vector search  ├─► 20 candidates
                               │
                     Cross-encoder reranker
                               │
                          Top 5 chunks
                               │
                    Grounded prompt + [SOURCE X] citations
                               │
                      Groq Llama3 → Answer
                               │
              ┌────────────────┴──────────────────┐
        Failsafe scanner              BGE semantic similarity
        (speculation keywords)        faithfulness scorer
              └────────────────┬──────────────────┘
                     Faithfulness score (0.0 – 1.0)
                     per-claim breakdown + matched context
```


## Key Features

**Hybrid Retrieval**
Combines BM25 (exact keyword matching) with BGE-large vector search (semantic similarity) in parallel. Legal documents contain precise terminology — BM25 catches exact phrases like "indemnification" and "nonrefundable" that vector search sometimes misses. Both arms retrieve 10 candidates each, merged and deduplicated to ~20 chunks.

**Cross-Encoder Reranking**
Rather than relying on cosine distance alone, every (query, chunk) pair is scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) which reads both together and produces a true relevance score. Top 5 chunks are passed to the LLM.

**Grounded Prompting with Citations**
The LLM is instructed via both system prompt and user prompt to only use provided sources, cite every claim with `[SOURCE X]` notation, and explicitly say "NOT FOUND IN DOCUMENT" when information is absent. Temperature is set to 0.0 for fully deterministic output.

**Three-Layer Hallucination Detection**

| Layer | Method | What it catches |
|---|---|---|
| Failsafe scanner | Keyword matching | Speculation language ("probably", "typically", "I believe") |
| Citation check | String matching | Answers with no `[SOURCE X]` markers |
| Faithfulness scorer | BGE-large semantic similarity | Claims not semantically supported by retrieved chunks |

**Per-Claim Faithfulness Scoring**
The answer is split into individual claims. Each claim is encoded with BGE-large and compared against every sentence in the retrieved context. A similarity score ≥ 0.72 means the claim is grounded. The final faithfulness score is the fraction of supported claims.


## Example Output

```
QUERY: What are the payment terms?

ANSWER:
According to [SOURCE 2], the payment terms are as follows:
* Customer agrees to pay amounts invoiced by Armorblox in U.S. dollars ("Fees").
* Order Forms are noncancellable and Fees are nonrefundable.
* Payment from Customer is due thirty (30) days after Customer's receipt of each invoice.

FAITHFULNESS EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Claim 1 → ✅ SUPPORTED | similarity: 0.937
Claim 2 → ✅ SUPPORTED | similarity: 0.975
Claim 3 → ✅ SUPPORTED | similarity: 0.989

FAITHFULNESS SCORE: 1.0 (3/3 claims supported)
✅ Fully grounded in the source document.
```


## Project Structure

```
VeritasRAG/
├── ingest.py        # PDF → clean text → semantic chunks → ChromaDB
├── retriever.py     # Hybrid BM25 + vector search → cross-encoder reranking
├── generator.py     # Grounded prompt → Groq Llama3 → cited answer + failsafe
├── evaluator.py     # Per-claim faithfulness scoring via BGE semantic similarity
├── app.py           # Interactive Q&A loop tying all modules together
└── data/            # Place your PDF documents here
```


## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5` |
| Vector DB | ChromaDB |
| Keyword search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Faithfulness model | BGE-large semantic similarity |
| LLM | Groq API — Llama 3 8B |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| PDF parsing | `pypdf` |

## Design Decisions Worth Noting

**Why BGE-large over all-MiniLM?**
BGE-large is specifically trained for retrieval tasks and significantly outperforms MiniLM on domain-specific text like legal documents. It supports query instruction prefixes which distinguish between "this is a question" and "this is a document chunk" at encoding time.

**Why hybrid retrieval?**
Legal documents use highly specific terminology. A query like "nonrefundable fees" needs exact keyword matching (BM25) as well as semantic understanding. Vector search alone missed the payment terms section in early testing — BM25 + vector together did not.

**Why cross-encoder reranking?**
Cosine similarity compares vectors independently. A cross-encoder reads the query and chunk together, giving it full context to judge relevance. This is slower but far more accurate — the reranker consistently promoted the correct chunk to position 1 even when vector search ranked it 4th or 5th.

**Why semantic similarity for faithfulness instead of NLI?**
Tested `cross-encoder/nli-deberta-v3-small` and `facebook/bart-large-mnli` — both returned "neutral" even for claims word-for-word identical to the source. BGE semantic similarity with a 0.72 threshold proved more reliable and consistent for this use case.



## What's Next

- [ ] Streamlit UI for non-technical users
- [ ] Multi-document support with document-level metadata filtering
- [ ] RAGAS integration for automated evaluation benchmarking
- [ ] Support for scanned PDFs via OCR (pytesseract)
- [ ] Contradiction detection — flag when two chunks say opposite things



## What I Learned Building This

This project started as a RAG tutorial follow-along and became something I had to debug, redesign, and rebuild from first principles multiple times:

- Fixed mid-sentence chunking by cleaning PDF text before splitting
- Replaced broken NLI models with BGE semantic similarity after testing showed consistent false negatives
- Discovered that chunk size directly determines answer quality — 400 tokens produced truncated answers, 800 tokens fixed it  
- Learned that vector search alone misses exact legal terminology — hybrid retrieval was the fix
- Understood why cross-encoder reranking matters by seeing it promote the correct chunk from position 4 to position 1
