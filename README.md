# 🧠 DocuMind — Advanced PDF Research Assistant

> A production-grade Retrieval-Augmented Generation (RAG) pipeline
> with Semantic Chunking, Hybrid Search, Cross-Encoder Reranking, and Query Rewriting.

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## What This Does

Upload any PDF. Ask questions in natural language. Get precise, cited answers.

The pipeline goes well beyond naive "embed → retrieve → generate" RAG, implementing
several techniques from current academic research.

---

## Pipeline Architecture

```
PDF Upload (bytes)
      │
      ▼
Text Extraction (pypdf — fresh BytesIO per engine, no shared file-pointer)
      │
      ▼
Semantic Chunking
  Adaptive threshold: mean(pairwise_sim) − 0.5·std, clamped [0.30, 0.65]
  Fallback: 400-word sliding window for scanned/table-heavy PDFs
      │
      ├─────────────────────┐
      ▼                     ▼
BM25 Index            ChromaDB (Dense Embeddings)
(Lexical)             (Sentence Transformers)
Collection name = docs-{model}-{file_hash8}
      │                     │
      └──────────┬──────────┘
                 ▼
   Reciprocal Rank Fusion (RRF)
   score(d) = Σ 1/(k + rank(d)), k=60
   [Cormack et al. 2009]
                 │
          [Optional HyDE]
    Generate hypothetical answer →
    embed it → retrieve against
    that vector instead of raw query
    [Gao et al. 2022, arXiv:2212.10496]
                 │
                 ▼
   Cross-Encoder Reranking
   [ms-marco-MiniLM-L-6-v2]
   sigmoid(logit) → [0,1] confidence
                 │
                 ▼
   Groq Llama 3.3 70B (Generation)
   Inline [Source N] citations
                 │
                 ▼
   Hallucination Guard
   Second LLM call checks each factual
   claim against retrieved evidence
                 │
                 ▼
   Cited Answer + Confidence Score + Evidence Chunks
```

---

## Key Concepts

### 1. Semantic Chunking with Adaptive Threshold

Word-count windows split mid-sentence and mid-concept.
We embed every sentence, compute pairwise cosine similarities between
consecutive sentences, then derive a document-specific split threshold:

```
threshold = clip(mean(sims) − 0.5·std(sims), 0.30, 0.65)
```

Dense technical text has high baseline similarity — it needs a lower
threshold to detect topic shifts. Narrative prose varies more naturally —
it needs a higher one. The adaptive formula adjusts per document
without manual tuning.

### 2. Hybrid Search

BM25 excels on exact/rare terms; semantic search excels on paraphrasing
and synonyms. Running both and fusing results outperforms either alone.

### 3. Reciprocal Rank Fusion (RRF)

Merges ranked lists without requiring comparable score scales.
Uses rank positions, not raw scores — scale-invariant by design.

```
RRF(doc) = Σ 1 / (k + rank(doc))    k = 60
```

### 4. Cross-Encoder Reranking

Bi-encoders (retrieval) encode query and document independently —
fast but lose cross-attention. Cross-encoders process `[query, document]`
jointly — higher precision at higher cost. Strategy: bi-encoder retrieves
top-12 candidates fast; cross-encoder reranks to top-4 accurately.

### 5. HyDE (Hypothetical Document Embedding)

Instead of rephrasing the question, generate a *hypothetical answer* and
embed that. Hypothetical answers live closer to real document passages in
embedding space than question surface forms do.

### 6. Confidence Signal

```
confidence = mean(sigmoid(reranker_logit)) over top-k chunks
```

This measures *retrieval quality*, not *answer quality*. Deliberately
separate concerns: a low-confidence answer that correctly says "not in
the document" is better than a high-confidence hallucination.

---

## Tech Stack

| Component         | Technology                            |
|-------------------|---------------------------------------|
| LLM               | Groq Llama 3.3 70B                    |
| Embedding Models  | Sentence Transformers (HuggingFace)   |
| Vector Store      | ChromaDB (EphemeralClient)            |
| Lexical Index     | BM25Okapi (rank-bm25)                 |
| Reranker          | CrossEncoder ms-marco-MiniLM-L-6-v2  |
| UI                | Streamlit                             |
| PDF Parsing       | pypdf                                 |

All embedding and reranking models run locally. No OpenAI dependency.

---

## Setup

```bash
git clone https://github.com/YOURUSERNAME/docuMind
cd docuMind
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Paste your free Groq API key from console.groq.com into .env

streamlit run app.py
```

### Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Model Comparison Mode

Side-by-side comparison of two embedding models on the same query:

- `all-MiniLM-L6-v2` vs `BAAI/bge-small-en-v1.5`
- Same pipeline settings, same document
- Compare retrieval confidence, answer quality, faithfulness verdict

Both engines read from the same pre-loaded `bytes` object — file-pointer
exhaustion (a common RAG bug) is not possible by design.

---

## Engineering Notes

### Why `pdf_bytes: bytes` instead of a file object

Streamlit's `UploadedFile` is a file-like object with a shared internal
pointer. After `PdfReader(uploaded)` reads the stream, the pointer is at
EOF. A second `PdfReader(uploaded)` call reads nothing — exactly what
happened in Model Comparison mode in earlier versions. Accepting `bytes`
and constructing a fresh `BytesIO` inside each call eliminates the issue.

### Why the collection name includes a file hash

ChromaDB collection names previously encoded only the embedding model name.
Uploading a second PDF deleted the first PDF's collection even while its
engine was still cached in `st.session_state`. The file-content hash makes
each `(model, file)` pair own its own collection permanently.

### Why models are loaded with `lru_cache`

`SentenceTransformer` and `CrossEncoder` load 100-400 MB of weights from
disk. Previously instantiated inside `RAGEngine.__init__`, they were
reloaded on every Streamlit widget interaction (any sidebar toggle
triggers a full script rerun). Module-level `lru_cache` ensures weights
are loaded exactly once per Python process.

---

## References

- Cormack, Clarke & Buettcher (2009) — Reciprocal Rank Fusion
- Gao et al. (2022) — Precise Zero-Shot Dense Retrieval (HyDE) — arXiv:2212.10496
- Nogueira & Cho (2019) — Passage Re-ranking with BERT
- MS MARCO Dataset — Cross-encoder training data
- Hugging Face MTEB Leaderboard — Embedding model benchmarks
- Kamradt (2023) — 5 Levels of Text Splitting