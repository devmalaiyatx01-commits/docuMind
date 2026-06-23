# rag_engine.py — DocuMind Production RAG Pipeline
#
# ── Changelog from v1 ─────────────────────────────────────────────────────────
#
#   FIX 1  pdf_bytes: bytes replaces file-object parameter
#          PdfReader leaves the file pointer at EOF after one read.
#          In Model Comparison mode both engines called index(uploaded) with the
#          same UploadedFile object — the second read returned an empty document.
#          Accepting bytes means callers do uploaded.getvalue() once and each
#          engine constructs its own BytesIO — safe for any number of re-reads.
#
#   FIX 2  ChromaDB collection name now includes a file-content hash
#          Old name: "docs-{model}" — shared across ALL files for a given model.
#          Uploading PDF-B deleted PDF-A's collection while the cached PDF-A
#          engine still held a reference to it → InvalidCollectionException.
#          New name: "docs-{model}-{file_hash8}" — each (model, file) pair owns
#          its own collection; switching PDFs never corrupts another engine.
#
#   FIX 3  _with_retry decorator on every Groq API call
#          429 / 503 / transient network errors previously crashed the whole page.
#          Exponential backoff with 3 attempts handles transient failures cleanly.
#
#   FIX 4  lru_cache on SentenceTransformer and CrossEncoder loaders
#          Models were re-constructed on every Streamlit rerun (widget click).
#          Module-level lru_cache ensures each model is loaded exactly once per
#          process — regardless of how many RAGEngine instances are created.
#
#   FIX 5  fetch_k hard-bounded to len(all_chunks) before any retrieval call
#          max(top_k, ...) could return 4 when only 2 chunks exist.
#          _semantic_retrieve had its own inner guard but _bm25_retrieve did not,
#          so the two result lists had unequal lengths entering _rrf_fuse.
#
#   FIX 6  _chunk_map: dict[int, Chunk] replaces positional list indexing
#          self.all_chunks[chunk_idx] assumed chunk.index == list position.
#          The invariant held in practice but was one reorder away from silently
#          returning the wrong chunk. Dict lookup is O(1) and explicit.
#
#   FIX 7  Empty-document guard raises ValueError with actionable message
#          Scanned/image-only PDFs produced pages=[] → all_chunks=[] →
#          BM25([]) + collection.query(n_results=0) → obscure library crash.
#          Now raises immediately with a user-readable message.
#
#   FIX 8  max_tokens raised: generation 800→1500, faithfulness 80→150
#          80 tokens truncated the VERDICT+REASON pair mid-sentence, causing
#          the regex to miss and reason to default to raw partial output.
#
#   FIX 9  Session state key now uses MD5(file_bytes) not filename
#          Two different files named "report.pdf" previously hit the same cache
#          slot, returning stale results from the first file.
#          (This fix lives in app.py — documented here for traceability.)
#
#   NEW    Adaptive similarity threshold in semantic chunking
#          Hardcoded 0.45 was too tight for dense technical text and too loose
#          for narrative prose. Threshold is now mean(sims) - 0.5·std(sims),
#          clamped to [0.30, 0.65] — adapts to each document's own similarity
#          distribution without requiring manual tuning per document type.
#
#   NEW    Structured logging with stage-level context throughout
# ──────────────────────────────────────────────────────────────────────────────

import hashlib
import io
import os
import re
import time
import logging
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Optional

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from groq import Groq

logger = logging.getLogger(__name__)

# ── Shared ChromaDB client ─────────────────────────────────────────────────────
# One EphemeralClient per process. All engines share one namespace so that
# uniquely-named collections are visible across engine instances.
_CHROMA_CLIENT = chromadb.EphemeralClient()


# ── Model loaders (lru_cache = loaded once per process) ──────────────────────
#
# Previously models were constructed inside RAGEngine.__init__, which runs on
# every Streamlit rerun. 100-400 MB weights were reloaded from disk on every
# widget interaction. lru_cache keys on the model name string so the same
# weights are shared across all engine instances and all reruns.

@lru_cache(maxsize=8)
def _load_embedding_model(name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


@lru_cache(maxsize=1)
def _load_reranker(name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    logger.info("Loading reranker: %s", name)
    return CrossEncoder(name)


# ── Retry decorator ───────────────────────────────────────────────────────────

def _with_retry(max_attempts: int = 3, backoff_base: float = 1.5):
    """
    Retry with exponential backoff for transient API failures.

    Groq returns HTTP 429 on rate-limit and 503 on transient overload.
    Without retry, either error crashes the entire Streamlit page with an
    unhandled exception. Three attempts with 1.5x backoff adds at most
    ~3.4 s of latency in the worst case — acceptable for a research tool.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        wait = backoff_base ** attempt
                        logger.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            fn.__name__, attempt + 1, max_attempts, exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            fn.__name__, max_attempts, exc,
                        )
            raise last_exc
        return wrapper
    return decorator


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A chunk is more than raw text.
    Page provenance lets citations say "Page 4" not just "Source 2".
    chunk_id is the stable lookup key stored in ChromaDB.
    """
    text: str
    page: int
    chunk_id: int   # renamed from 'index' to make the identity role explicit

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, page={self.page}, text='{preview}...')"


@dataclass
class RetrievalResult:
    """Everything the UI needs to render one evidence card."""
    chunk: Chunk
    score: float
    retrieval_method: str   # "hybrid" | "semantic" | "bm25"


@dataclass
class QueryResult:
    """
    Full pipeline output — structured so the UI never parses strings.
    """
    answer: str
    sources: list[RetrievalResult]
    rewritten_query: Optional[str]
    confidence: float               # 0.0–1.0 proxy for retrieval quality
    faithfulness_ok: bool           # did the LLM self-check pass?
    faithfulness_note: str
    latency_ms: dict = field(default_factory=dict)


# ── Engine ─────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Production RAG pipeline.

    Design decisions worth discussing in interviews:

    SEMANTIC CHUNKING
        Word-count windows split mid-sentence and mid-concept.
        We embed every sentence, then group by cosine-similarity drops:
        when consecutive sentences diverge past an *adaptive* threshold
        (mean − 0.5·std of pairwise similarities) we start a new chunk.
        This yields topic-coherent chunks — retrieved evidence reads as
        coherent text, not mid-sentence fragments.

    HyDE (HYPOTHETICAL DOCUMENT EMBEDDING)
        Standard rewriting rephrases the question.
        HyDE asks the LLM to *answer* the question, then embeds that
        hypothetical answer. A hypothetical answer lives closer in embedding
        space to real document passages than the question surface form does.
        (Gao et al. 2022, arXiv:2212.10496)

    HALLUCINATION GUARD
        Second LLM call checks whether factual claims in the answer are
        grounded in retrieved sources — lightweight RAGAS faithfulness.

    CONFIDENCE SIGNAL
        Mean sigmoid(reranker_logit) over top-k chunks.
        Honest proxy for retrieval quality, not answer quality.
        Deliberately separated: a low-confidence answer that correctly says
        "not in the document" is better than a high-confidence hallucination.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_reranker: bool = True,
        semantic_chunk: bool = True,
    ):
        self.model_name = model_name
        self.use_reranker = use_reranker
        self.semantic_chunk = semantic_chunk

        # FIX 4: models loaded via lru_cache — not reinstantiated per engine
        self.embedding_model: SentenceTransformer = _load_embedding_model(model_name)
        self.reranker: Optional[CrossEncoder] = (
            _load_reranker() if use_reranker else None
        )

        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.chroma_client = _CHROMA_CLIENT
        self.collection = None

        self.all_chunks: list[Chunk] = []
        # FIX 6: explicit dict — no positional assumption
        self._chunk_map: dict[int, Chunk] = {}
        self.bm25: Optional[BM25Okapi] = None

    # ── STAGE 1: EXTRACTION ────────────────────────────────────────────────────

    def _extract_pages(self, pdf_bytes: bytes) -> list[tuple[str, int]]:
        """
        Returns list of (page_text, page_number).

        Accepts bytes so the caller can pass the same bytes to multiple
        engines without worrying about file-pointer state.
        """
        # FIX 1: construct a fresh BytesIO per call — no shared pointer state
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[tuple[str, int]] = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, i + 1))
        return pages

    # ── STAGE 2a: NAIVE CHUNKING ───────────────────────────────────────────────

    @staticmethod
    def _word_window_chunks(
        pages: list[tuple[str, int]],
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> list[Chunk]:
        """
        Classic sliding-window fallback.
        Used when sentence segmentation is unreliable (tables, code, scanned PDFs).
        """
        chunks: list[Chunk] = []
        idx = 0
        for page_text, page_num in pages:
            words = page_text.split()
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                text = " ".join(words[i: i + chunk_size])
                if len(text.strip()) > 50:
                    chunks.append(Chunk(text=text, page=page_num, chunk_id=idx))
                    idx += 1
        return chunks

    # ── STAGE 2b: SEMANTIC CHUNKING ────────────────────────────────────────────

    def _semantic_chunks(
        self,
        pages: list[tuple[str, int]],
        min_chunk_words: int = 60,
        max_chunk_words: int = 500,
    ) -> list[Chunk]:
        """
        Topic-boundary chunking via embedding similarity drops.

        Algorithm:
        1. Split document into sentences.
        2. Embed all sentences in one batch.
        3. Compute cosine similarity between consecutive sentence pairs.
        4. Adaptive threshold: mean(sims) − 0.5·std(sims), clamped [0.30, 0.65].
           This adapts to each document's own similarity distribution — dense
           technical text has high baseline similarity and needs a lower threshold;
           narrative prose varies more and needs a higher one.
        5. When similarity drops below threshold → topic shift → start new chunk.
        6. Enforce min/max word counts to prevent micro/mega chunks.

        Reference: Greg Kamradt's "5 levels of text splitting" (2023)
        """
        # 1. Sentence segmentation
        all_sentences: list[tuple[str, int]] = []
        for page_text, page_num in pages:
            raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', page_text)
            for s in raw:
                s = s.strip()
                if len(s) > 20:
                    all_sentences.append((s, page_num))

        if len(all_sentences) < 2:
            logger.warning("Too few sentences for semantic chunking — falling back to word-window")
            return self._word_window_chunks(pages)

        texts = [s for s, _ in all_sentences]
        pages_map = [p for _, p in all_sentences]

        # 2. Embed in one pass
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=False, batch_size=64
        )

        # 3. Pairwise similarity between consecutive sentences
        sims = [
            float(cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0])
            for i in range(len(embeddings) - 1)
        ]

        # 4. Adaptive threshold — data-driven, not hand-tuned
        sim_arr = np.array(sims)
        raw_threshold = float(sim_arr.mean() - 0.5 * sim_arr.std())
        similarity_threshold = float(np.clip(raw_threshold, 0.30, 0.65))
        logger.debug(
            "Semantic chunk threshold: %.3f (mean=%.3f, std=%.3f)",
            similarity_threshold, sim_arr.mean(), sim_arr.std(),
        )

        # 5. Build chunks
        chunks: list[Chunk] = []
        idx = 0
        group: list[str] = [texts[0]]
        group_page = pages_map[0]
        word_count = len(texts[0].split())

        for i, sim in enumerate(sims):
            next_words = len(texts[i + 1].split())
            over_max = (word_count + next_words) > max_chunk_words
            topic_shift = sim < similarity_threshold

            if (topic_shift or over_max) and word_count >= min_chunk_words:
                chunks.append(Chunk(text=" ".join(group), page=group_page, chunk_id=idx))
                idx += 1
                group = [texts[i + 1]]
                group_page = pages_map[i + 1]
                word_count = next_words
            else:
                group.append(texts[i + 1])
                word_count += next_words

        # Flush tail
        if group:
            tail_text = " ".join(group)
            if len(tail_text.split()) >= min_chunk_words // 2:
                chunks.append(Chunk(text=tail_text, page=group_page, chunk_id=idx))
            elif chunks:
                prev = chunks[-1]
                chunks[-1] = Chunk(
                    text=prev.text + " " + tail_text,
                    page=prev.page,
                    chunk_id=prev.chunk_id,
                )

        return chunks if chunks else self._word_window_chunks(pages)

    # ── STAGE 3: INDEXING ─────────────────────────────────────────────────────

    def index(self, pdf_bytes: bytes) -> int:
        """
        Dual-index: BM25 (lexical) + ChromaDB (dense). Returns chunk count.

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF content. Callers obtain this via uploaded.getvalue() once
            and pass the same bytes to each engine — no file-pointer concerns.
        """
        # FIX 2: include file hash in collection name so engines for different
        # files never collide, and switching PDFs never corrupts a cached engine.
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
        safe_model = re.sub(r'[^a-z0-9-]', '-', self.model_name.lower())
        cname = f"docs-{safe_model}-{file_hash}"[:63]
        logger.info("Indexing into collection '%s'", cname)

        pages = self._extract_pages(pdf_bytes)

        # FIX 7: guard empty document early with an actionable message
        if not pages:
            raise ValueError(
                "No extractable text found in this PDF. "
                "It may be a scanned or image-only document. "
                "Try a PDF with selectable text."
            )

        chunks = (
            self._semantic_chunks(pages) if self.semantic_chunk
            else self._word_window_chunks(pages)
        )

        if not chunks:
            raise ValueError("Document produced zero chunks after splitting. Check PDF content.")

        self.all_chunks = chunks
        # FIX 6: build the explicit lookup dict
        self._chunk_map = {c.chunk_id: c for c in chunks}

        # BM25 index
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # ChromaDB index — delete stale collection for this exact (model, file) pair
        try:
            self.chroma_client.delete_collection(cname)
        except Exception:
            pass  # collection didn't exist yet — that's fine

        self.collection = self.chroma_client.create_collection(
            name=cname, metadata={"hnsw:space": "cosine"}
        )

        embeddings = self.embedding_model.encode(
            [c.text for c in chunks],
            show_progress_bar=False, batch_size=32,
        ).tolist()

        self.collection.add(
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            ids=[f"chunk_{c.chunk_id}" for c in chunks],
            metadatas=[{"page": c.page} for c in chunks],
        )

        chunk_type = "semantic" if self.semantic_chunk else "fixed-window"
        logger.info("Indexed %d %s chunks", len(chunks), chunk_type)
        return len(chunks)

    # ── RETRIEVAL: BM25 ───────────────────────────────────────────────────────

    def _bm25_retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # top_k is already bounded to len(all_chunks) by the caller
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.all_chunks[i], float(scores[i])) for i in top_idx]

    # ── RETRIEVAL: SEMANTIC ───────────────────────────────────────────────────

    def _semantic_retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        # top_k already bounded by caller; guard once more for safety
        n = min(top_k, len(self.all_chunks))
        if n == 0:
            return []
        q_emb = self.embedding_model.encode([query]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=n)
        results: list[tuple[Chunk, float]] = []
        for dist, cid in zip(res["distances"][0], res["ids"][0]):
            sim = round(max(0.0, min(1.0, 1 - dist)), 4)
            chunk_id = int(cid.split("_")[1])
            # FIX 6: safe dict lookup — KeyError is explicit, not silent wrong chunk
            chunk = self._chunk_map[chunk_id]
            results.append((chunk, sim))
        return results

    # ── FUSION: RRF ───────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_res: list[tuple[Chunk, float]],
        sem_res: list[tuple[Chunk, float]],
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """
        Reciprocal Rank Fusion — rank-position fusion, scale-invariant.
        Formula: score(d) = Σ 1/(k + rank(d))    [Cormack et al. 2009]
        """
        rrf: dict[int, float] = {}
        chunk_map: dict[int, Chunk] = {}

        for rank, (chunk, _) in enumerate(bm25_res):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(sem_res):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], round(score, 6)) for cid, score in ranked]

    # ── RERANKING ─────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_n: int,
    ) -> list[tuple[Chunk, float]]:
        """
        Cross-encoder reranking. Raw logits → sigmoid → [0, 1].

        Bi-encoders encode query and document independently.
        Cross-encoders process [query, document] jointly — token-level
        attention across the pair yields much higher precision.

        Strategy: bi-encoder retrieves top-12 candidates fast;
        cross-encoder reranks to top-4 accurately.
        """
        if not self.reranker or not candidates:
            return candidates[:top_n]

        pairs = [[query, c.text] for c, _ in candidates]
        logits = self.reranker.predict(pairs)
        sig = [round(float(1 / (1 + np.exp(-s))), 4) for s in logits]

        ranked = sorted(
            zip([c for c, _ in candidates], sig),
            key=lambda x: x[1], reverse=True,
        )
        return list(ranked[:top_n])

    # ── QUERY REWRITING: HyDE ─────────────────────────────────────────────────

    @_with_retry(max_attempts=3)
    def _hyde_query(self, question: str) -> str:
        """
        Hypothetical Document Embedding (HyDE).

        Generate a hypothetical answer, embed it, retrieve against that vector.
        The hypothetical answer is semantically closer to real document passages
        than the question surface form — even when factually wrong.

        Reference: Gao et al. (2022) arXiv:2212.10496
        """
        resp = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    "Write a short, dense factual passage (3–5 sentences) that "
                    "would directly answer the following question. "
                    "Write as if you are an expert author of a technical document. "
                    "Do not say 'the answer is' — just write the passage.\n\n"
                    f"Question: {question}"
                )
            }],
            temperature=0.3,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()

    # ── HALLUCINATION GUARD ────────────────────────────────────────────────────

    @_with_retry(max_attempts=3)
    def _check_faithfulness(
        self,
        question: str,
        answer: str,
        chunks: list[Chunk],
    ) -> tuple[bool, str]:
        """
        LLM self-check: does the answer contain claims unsupported by sources?

        Lightweight implementation of the RAGAS faithfulness metric.
        A second LLM call fact-checks the answer against retrieved evidence.
        In production, a dedicated NLI model (DeBERTa-v3) would be faster and
        cheaper — but an LLM call is more accessible and surprisingly effective.

        Returns: (is_faithful: bool, explanation: str)
        """
        context = "\n\n".join(
            f"[Source {i+1}]\n{c.text}" for i, c in enumerate(chunks)
        )
        prompt = (
            "You are a strict hallucination detector for a RAG system. "
            "Your job is to catch GENUINE hallucinations only — factual claims "
            "in the answer that cannot be traced to ANY of the source passages.\n\n"
            "DO NOT flag:\n"
            "- 'This information is not in the document' (valid response)\n"
            "- Paraphrasing or summarising source content in different words\n"
            "- [Source N] citation labels\n"
            "- Connector phrases and reasoning\n\n"
            "DO flag:\n"
            "- Specific numbers, names, or dates not in any source\n"
            "- Claims that contradict source content\n"
            "- Invented details presented as fact\n\n"
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Respond in exactly this format:\n"
            "VERDICT: FAITHFUL or UNFAITHFUL\n"
            "REASON: one sentence. If UNFAITHFUL, quote the specific hallucinated claim."
        )
        resp = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,   # FIX 8: was 80 — too tight for VERDICT+REASON
        )
        raw = resp.choices[0].message.content.strip()
        is_faithful = "UNFAITHFUL" not in raw.upper()
        reason_match = re.search(r'REASON:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else raw
        return is_faithful, reason

    # ── GENERATION ────────────────────────────────────────────────────────────

    @_with_retry(max_attempts=3)
    def _generate(self, question: str, chunks: list[Chunk]) -> str:
        """
        Grounded generation with inline citations.

        Source labels are plain [Source N] — page numbers are surfaced in
        the UI evidence cards, not mixed into LLM context labels.
        Mixing page numbers into context caused the faithfulness checker
        to flag [Page 6] as a hallucination.
        """
        context = "\n\n".join(
            f"[Source {i+1}]\n{c.text}" for i, c in enumerate(chunks)
        )
        prompt = (
            "You are a precise research assistant. "
            "Answer ONLY using the provided source passages. "
            "Cite inline as [Source 1], [Source 2], etc. "
            "Do NOT invent page numbers or any information not in the sources. "
            "If the answer is absent from sources, say exactly: "
            '"This information is not present in the document."\n\n'
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        resp = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,  # FIX 8: was 800 — truncated multi-source answers
        )
        return resp.choices[0].message.content

    # ── CONFIDENCE SIGNAL ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(sources: list[RetrievalResult]) -> float:
        """
        Retrieval confidence = mean sigmoid(reranker logit) over top results.

        This is an honest proxy for retrieval quality, not answer quality.
        We deliberately separate these two concerns:
        - Low confidence + correct "not in document" → good pipeline behaviour
        - High confidence + hallucinated answer → faithfulness guard should catch it

        Threshold guide (surfaced in UI):
        > 0.70  → strong match
        0.40–0.70 → partial match
        < 0.40  → weak/no match
        """
        if not sources:
            return 0.0
        return round(float(np.mean([s.score for s in sources])), 4)

    # ── FULL PIPELINE ─────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 4,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        do_hyde: bool = False,
        check_faithfulness: bool = True,
    ) -> QueryResult:
        """
        Full pipeline with per-stage latency tracking.

        raw query
          → [HyDE: hypothetical answer embed]
          → hybrid BM25 + semantic retrieval
          → RRF fusion
          → cross-encoder rerank
          → grounded generation
          → faithfulness self-check
          → QueryResult
        """
        # Input guard
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if not self.all_chunks:
            raise RuntimeError("Engine has not been indexed yet. Call index() first.")

        timings: dict[str, float] = {}
        rewritten: Optional[str] = None
        retrieval_query = question

        # Stage: HyDE query rewriting
        if do_hyde:
            t0 = time.perf_counter()
            rewritten = self._hyde_query(question)
            retrieval_query = rewritten
            timings["hyde_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            logger.debug("HyDE query: %s", rewritten[:80])

        # FIX 5: bound fetch_k to actual corpus size before calling any retriever
        n_chunks = len(self.all_chunks)
        fetch_k = min(max(top_k, top_k * 3), n_chunks)

        # Stage: retrieval
        t0 = time.perf_counter()
        if use_hybrid and self.bm25:
            bm25_res = self._bm25_retrieve(retrieval_query, fetch_k)
            sem_res = self._semantic_retrieve(retrieval_query, fetch_k)
            candidates = self._rrf_fuse(bm25_res, sem_res)
            method = "hybrid"
        else:
            candidates = self._semantic_retrieve(retrieval_query, fetch_k)
            method = "semantic"
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Stage: reranking
        t0 = time.perf_counter()
        if use_rerank and self.reranker and candidates:
            final = self._rerank(question, candidates, top_n=top_k)
            score_source = "reranker"
        else:
            final = candidates[:top_k]
            score_source = "retrieval"
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        sources = [
            RetrievalResult(
                chunk=chunk,
                score=score,
                retrieval_method=f"{method}+rerank" if score_source == "reranker" else method,
            )
            for chunk, score in final
        ]

        # Stage: generation
        t0 = time.perf_counter()
        answer = self._generate(question, [s.chunk for s in sources])
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Stage: faithfulness check
        faithful, faith_note = True, "Check skipped"
        if check_faithfulness:
            t0 = time.perf_counter()
            faithful, faith_note = self._check_faithfulness(
                question, answer, [s.chunk for s in sources]
            )
            timings["faithfulness_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        confidence = self._compute_confidence(sources)
        logger.info(
            "Query complete — confidence=%.4f, faithful=%s, stages=%s",
            confidence, faithful, timings,
        )

        return QueryResult(
            answer=answer,
            sources=sources,
            rewritten_query=rewritten,
            confidence=confidence,
            faithfulness_ok=faithful,
            faithfulness_note=faith_note,
            latency_ms=timings,
        )