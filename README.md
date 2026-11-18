# DocuMind

Upload a PDF, ask questions in plain language, and get answers with citations to the passages they
came from. A second model call checks every claim in the answer against the retrieved evidence.

Live demo: https://devmalaiyatx01-commits-documind-app-nubxyi.streamlit.app/

## Pipeline

```
PDF bytes
  -> text extraction (pypdf, a fresh BytesIO per engine)
  -> semantic chunking (split where neighbouring sentences stop being similar)
  -> BM25 index (exact terms)  +  ChromaDB with sentence-transformer embeddings (meaning)
  -> Reciprocal Rank Fusion of the two ranked lists
  -> optional HyDE: embed a hypothetical answer instead of the question
  -> cross-encoder reranking (ms-marco-MiniLM-L-6-v2), top 12 -> top 4
  -> Llama 3.3 70B on Groq writes the answer with [Source N] citations
  -> hallucination guard: a second call checks each claim against the evidence
  -> answer + confidence score + the evidence chunks
```

## Design choices

**Semantic chunking with an adaptive threshold.** Every sentence is embedded and consecutive
sentences are compared. The split threshold is `clip(mean - 0.5 * std, 0.30, 0.65)` of those
similarities, so dense technical text and loose prose each get a threshold that fits them. Scanned
or table-heavy PDFs fall back to a 400-word sliding window.

**Hybrid search with RRF.** BM25 is good at exact and rare terms, embeddings at paraphrase. RRF
merges the two lists by rank, `score(d) = sum(1 / (60 + rank))`, so the two score scales never
have to be compared.

**Cross-encoder reranking.** The bi-encoder retrieves fast but encodes query and passage
separately. The cross-encoder reads them together, which is slower and more precise, so it only
reranks the top 12.

**Confidence is about retrieval, not the answer.** `confidence = mean(sigmoid(reranker logit))`
over the top chunks. An honest "not in the document" with low confidence beats a confident
hallucination.

**Model comparison mode.** Runs `all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` side by side on
the same question and document to compare retrieval confidence and the faithfulness verdict.

## Bugs fixed along the way

* **Empty second read.** Streamlit's uploaded file has one shared read pointer, so the second engine
  in comparison mode read nothing. Engines now take `bytes` and build their own `BytesIO`.
* **Collections deleting each other.** ChromaDB collection names used only the model name, so a
  second PDF deleted the first one's index. Names now include a hash of the file content.
* **Models reloading on every click.** Streamlit reruns the script on every widget change, which
  reloaded 100-400 MB of weights. Models are now loaded once per process with `lru_cache`.

## Stack

Groq (Llama 3.3 70B), sentence-transformers, ChromaDB, rank-bm25, cross-encoder reranker, pypdf,
Streamlit. Embedding and reranking run locally.

## Run it

```bash
git clone https://github.com/devmalaiyatx01-commits/docuMind
cd docuMind
pip install -r requirements.txt
cp .env.example .env          # add a free Groq API key from console.groq.com
streamlit run app.py
python -m pytest test_rag_engine.py -v
```

## References

* Cormack, Clarke and Buettcher (2009), Reciprocal Rank Fusion
* Gao et al. (2022), Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE), arXiv:2212.10496
* Nogueira and Cho (2019), Passage Re-ranking with BERT
