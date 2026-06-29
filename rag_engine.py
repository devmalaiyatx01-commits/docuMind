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

_CHROMA_CLIENT = chromadb.EphemeralClient()

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=8)
def _load_embedding_model(name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


@lru_cache(maxsize=4)
def _load_reranker(name: str) -> CrossEncoder:
    logger.info("Loading reranker: %s", name)
    return CrossEncoder(name)


def _with_retry(max_attempts: int = 3, backoff_base: float = 1.5):
    """Exponential backoff for transient Groq API failures (429, 503)."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception = RuntimeError("no attempts made")
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        wait = backoff_base ** attempt
                        logger.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.1fs",
                            fn.__name__, attempt + 1, max_attempts, exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error("%s failed after %d attempts: %s", fn.__name__, max_attempts, exc)
            raise last_exc
        return wrapper
    return decorator


@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, page={self.page}, text='{preview}...')"


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    retrieval_method: str


@dataclass
class QueryResult:
    answer: str
    sources: list[RetrievalResult]
    rewritten_query: Optional[str]
    confidence: float
    faithfulness_ok: bool
    faithfulness_note: str
    latency_ms: dict = field(default_factory=dict)


class RAGEngine:
    """
    PDF question-answering pipeline.

    Indexing: semantic chunking (adaptive cosine threshold) or fixed word-window
    fallback → dual BM25 + ChromaDB index.

    Retrieval: optional HyDE query rewriting → BM25 + dense retrieval →
    Reciprocal Rank Fusion → cross-encoder reranking.

    Generation: Groq Llama 3.3 70B with inline source citations, followed by
    an optional faithfulness self-check against retrieved evidence.
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

        self.embedding_model: SentenceTransformer = _load_embedding_model(model_name)
        self.reranker: Optional[CrossEncoder] = (
            _load_reranker(RERANKER_MODEL) if use_reranker else None
        )

        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.chroma_client = _CHROMA_CLIENT
        self.collection = None

        self.all_chunks: list[Chunk] = []
        self._chunk_map: dict[int, Chunk] = {}
        self.bm25: Optional[BM25Okapi] = None

    # ── Text extraction ────────────────────────────────────────────────────────

    def _extract_pages(self, pdf_bytes: bytes) -> list[tuple[str, int]]:
        """Return (page_text, page_number) for every page with extractable text."""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[tuple[str, int]] = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, i + 1))
        return pages

    # ── Chunking: fixed window ─────────────────────────────────────────────────

    @staticmethod
    def _word_window_chunks(
        pages: list[tuple[str, int]],
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> list[Chunk]:
        """
        Sliding word-window chunking. Used as a fallback for scanned PDFs or
        documents where sentence segmentation produces too few segments.
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

    # ── Chunking: semantic ─────────────────────────────────────────────────────

    def _semantic_chunks(
        self,
        pages: list[tuple[str, int]],
        min_chunk_words: int = 60,
        max_chunk_words: int = 500,
    ) -> list[Chunk]:
        """
        Topic-boundary chunking using embedding cosine similarity.

        Sentences are embedded in one batch, then grouped by consecutive-pair
        similarity. A topic shift (similarity below threshold) or word-count
        ceiling starts a new chunk.

        The split threshold is derived from the document's own similarity
        distribution: mean(sims) - 0.5 * std(sims), clamped to [0.30, 0.65].
        Dense technical text has a high baseline similarity and needs a lower
        threshold to detect topic changes; narrative prose varies more, so
        it naturally gets a higher threshold.
        """
        all_sentences: list[tuple[str, int]] = []
        for page_text, page_num in pages:
            raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', page_text)
            for s in raw:
                s = s.strip()
                if len(s) > 20:
                    all_sentences.append((s, page_num))

        if len(all_sentences) < 2:
            logger.warning("Too few sentences for semantic chunking — using word-window fallback")
            return self._word_window_chunks(pages)

        texts = [s for s, _ in all_sentences]
        pages_map = [p for _, p in all_sentences]

        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, batch_size=64)

        sims = [
            float(cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1),
            )[0][0])
            for i in range(len(embeddings) - 1)
        ]

        sim_arr = np.array(sims)
        raw_threshold = float(sim_arr.mean() - 0.5 * sim_arr.std())
        threshold = float(np.clip(raw_threshold, 0.30, 0.65))
        logger.debug("Chunk threshold: %.3f (mean=%.3f, std=%.3f)", threshold, sim_arr.mean(), sim_arr.std())

        chunks: list[Chunk] = []
        idx = 0
        group: list[str] = [texts[0]]
        group_page = pages_map[0]
        word_count = len(texts[0].split())

        for i, sim in enumerate(sims):
            next_words = len(texts[i + 1].split())
            over_max = (word_count + next_words) > max_chunk_words
            topic_shift = sim < threshold

            if (topic_shift or over_max) and word_count >= min_chunk_words:
                chunks.append(Chunk(text=" ".join(group), page=group_page, chunk_id=idx))
                idx += 1
                group = [texts[i + 1]]
                group_page = pages_map[i + 1]
                word_count = next_words
            else:
                group.append(texts[i + 1])
                word_count += next_words

        # Flush remaining sentences
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

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index(self, pdf_bytes: bytes) -> int:
        """
        Build a BM25 index and a ChromaDB dense vector index from a PDF.

        The ChromaDB collection name encodes both the embedding model and a
        hash of the file content. This means two PDFs with the same filename
        get separate collections, and switching files never corrupts a cached
        engine that is still in session state.

        Returns the number of chunks indexed.
        """
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
        safe_model = re.sub(r'[^a-z0-9-]', '-', self.model_name.lower())
        cname = f"docs-{safe_model}-{file_hash}"[:63]
        logger.info("Indexing into collection '%s'", cname)

        pages = self._extract_pages(pdf_bytes)
        if not pages:
            raise ValueError(
                "No extractable text found in this PDF. "
                "It may be a scanned or image-only document — try a PDF with selectable text."
            )

        chunks = (
            self._semantic_chunks(pages) if self.semantic_chunk
            else self._word_window_chunks(pages)
        )

        if not chunks:
            raise ValueError("Document produced zero chunks after splitting. Check PDF content.")

        self.all_chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}

        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        try:
            self.chroma_client.delete_collection(cname)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=cname, metadata={"hnsw:space": "cosine"}
        )

        embeddings = self.embedding_model.encode(
            [c.text for c in chunks], show_progress_bar=False, batch_size=32
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

    # ── Retrieval: BM25 ───────────────────────────────────────────────────────

    def _bm25_retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        # Use _chunk_map rather than positional indexing — chunk_id != list position
        return [(self._chunk_map[i], float(scores[i])) for i in top_idx if i in self._chunk_map]

    # ── Retrieval: dense ──────────────────────────────────────────────────────

    def _semantic_retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        n = min(top_k, len(self.all_chunks))
        if n == 0:
            return []
        q_emb = self.embedding_model.encode([query]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=n)
        results: list[tuple[Chunk, float]] = []
        for dist, cid in zip(res["distances"][0], res["ids"][0]):
            sim = round(max(0.0, min(1.0, 1 - dist)), 4)
            chunk_id = int(cid.split("_")[1])
            results.append((self._chunk_map[chunk_id], sim))
        return results

    # ── Fusion: RRF ───────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_res: list[tuple[Chunk, float]],
        sem_res: list[tuple[Chunk, float]],
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """
        Reciprocal Rank Fusion merges ranked lists without requiring comparable
        score scales. Score = sum of 1/(k + rank) across retrieval methods.

        A chunk that appears in both BM25 and semantic results accumulates
        score from both lists, which is why hybrid retrieval typically beats
        either method alone. (Cormack, Clarke & Buettcher, 2009)
        """
        rrf: dict[int, float] = {}
        chunk_ref: dict[int, Chunk] = {}

        for rank, (chunk, _) in enumerate(bm25_res):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1 / (k + rank + 1)
            chunk_ref[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(sem_res):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1 / (k + rank + 1)
            chunk_ref[chunk.chunk_id] = chunk

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_ref[cid], round(score, 6)) for cid, score in ranked]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_n: int,
    ) -> list[tuple[Chunk, float]]:
        """
        Cross-encoder reranking over bi-encoder candidates.

        Bi-encoders encode query and document independently, so they miss
        fine-grained query-document interactions. The cross-encoder processes
        the [query, document] pair jointly, giving much higher precision at
        the cost of speed. We use it only on the top-N candidates from RRF
        to keep latency manageable.

        Raw logits are passed through sigmoid to produce a [0, 1] confidence
        score, which also serves as the retrieval confidence signal.
        """
        if not self.reranker or not candidates:
            return candidates[:top_n]

        pairs = [[query, c.text] for c, _ in candidates]
        logits = self.reranker.predict(pairs)
        sigmoid_scores = [round(float(1 / (1 + np.exp(-s))), 4) for s in logits]

        ranked = sorted(
            zip([c for c, _ in candidates], sigmoid_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return list(ranked[:top_n])

    # ── Query rewriting: HyDE ─────────────────────────────────────────────────

    @_with_retry(max_attempts=3)
    def _hyde_query(self, question: str) -> str:
        """
        Hypothetical Document Embedding (Gao et al., 2022 — arXiv:2212.10496).

        Instead of embedding the question directly, we ask the LLM to write a
        short hypothetical answer and embed that. A hypothetical answer lives
        closer to real document passages in embedding space than a question
        does, even when the answer is factually wrong.
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
                ),
            }],
            temperature=0.3,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()

    # ── Faithfulness check ────────────────────────────────────────────────────

    @_with_retry(max_attempts=3)
    def _check_faithfulness(
        self,
        question: str,
        answer: str,
        chunks: list[Chunk],
    ) -> tuple[bool, str]:
        """
        LLM self-check: are the factual claims in the answer grounded in
        the retrieved sources?

        This is a lightweight version of the RAGAS faithfulness metric. A
        dedicated NLI model (e.g. DeBERTa-v3) would be faster and cheaper
        in production, but an LLM call is easier to deploy and works well
        for a portfolio tool.
        """
        context = "\n\n".join(f"[Source {i+1}]\n{c.text}" for i, c in enumerate(chunks))
        prompt = (
            "You are a strict hallucination detector for a RAG system. "
            "Your job is to catch GENUINE hallucinations only — factual claims "
            "in the answer that cannot be traced to ANY of the source passages.\n\n"
            "DO NOT flag:\n"
            "- 'This information is not in the document' (a valid response)\n"
            "- Paraphrasing or summarising source content in different words\n"
            "- [Source N] citation labels\n"
            "- Connector phrases and reasoning\n\n"
            "DO flag:\n"
            "- Specific numbers, names, or dates not present in any source\n"
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
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        is_faithful = "UNFAITHFUL" not in raw.upper()
        match = re.search(r'REASON:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
        reason = match.group(1).strip() if match else raw
        return is_faithful, reason

    # ── Generation ────────────────────────────────────────────────────────────

    @_with_retry(max_attempts=3)
    def _generate(self, question: str, chunks: list[Chunk]) -> str:
        """
        Generate a grounded answer with inline [Source N] citations.

        Source labels are kept simple — page numbers are shown in the UI
        evidence cards rather than in the LLM context, which prevents the
        faithfulness checker from incorrectly flagging [Page N] labels as
        hallucinations.
        """
        context = "\n\n".join(f"[Source {i+1}]\n{c.text}" for i, c in enumerate(chunks))
        prompt = (
            "You are a precise research assistant. "
            "Answer ONLY using the provided source passages. "
            "Cite inline as [Source 1], [Source 2], etc. "
            "Do NOT invent page numbers or any information not in the sources. "
            "If the answer is absent from the sources, say exactly: "
            '"This information is not present in the document."\n\n'
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        resp = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        return resp.choices[0].message.content

    # ── Confidence ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(sources: list[RetrievalResult]) -> float:
        """
        Mean sigmoid(reranker logit) across the top-k chunks.

        This measures retrieval quality, not answer quality — the two are
        deliberately separate. A low confidence score that produces a correct
        "not found in document" answer is a better outcome than a high
        confidence score paired with a hallucinated answer.

        Threshold guide:
          > 0.70  — strong evidence match
          0.40–0.70 — partial match
          < 0.40  — weak match; consider the answer may be outside the document
        """
        if not sources:
            return 0.0
        return round(float(np.mean([s.score for s in sources])), 4)

    # ── Full pipeline ──────────────────────────────────────────────────────────

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
        Run the full RAG pipeline and return a structured QueryResult.

        Stages (each timed independently):
          1. HyDE query rewriting (optional)
          2. BM25 + dense retrieval, RRF fusion (or dense-only)
          3. Cross-encoder reranking (optional)
          4. Grounded generation
          5. Faithfulness self-check (optional)
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if not self.all_chunks:
            raise RuntimeError("Engine has not been indexed yet — call index() first.")

        timings: dict[str, float] = {}
        rewritten: Optional[str] = None
        retrieval_query = question

        if do_hyde:
            t0 = time.perf_counter()
            rewritten = self._hyde_query(question)
            retrieval_query = rewritten
            timings["hyde_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            logger.debug("HyDE query: %s", rewritten[:80])

        # Bound fetch_k to the actual corpus size
        n_chunks = len(self.all_chunks)
        fetch_k = min(top_k * 3, n_chunks)

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

        t0 = time.perf_counter()
        answer = self._generate(question, [s.chunk for s in sources])
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        faithful, faith_note = True, "Check skipped"
        if check_faithfulness:
            t0 = time.perf_counter()
            faithful, faith_note = self._check_faithfulness(
                question, answer, [s.chunk for s in sources]
            )
            timings["faithfulness_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        confidence = self._compute_confidence(sources)
        logger.info(
            "Query done — confidence=%.4f faithful=%s stages=%s",
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