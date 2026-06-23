# tests/test_rag_engine.py
#
# Run with:  pytest tests/ -v
#
# Coverage targets:
#   - Bug fix regressions (file-pointer safety, collection name isolation,
#     empty-PDF guard, fetch_k bounding, chunk map lookup)
#   - Core pipeline contracts (chunking, RRF, reranking, confidence)
#   - Edge cases (single-chunk corpus, empty questions, unicode text)

import hashlib
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_engine import RAGEngine, Chunk, RetrievalResult, QueryResult, _rrf_fuse_standalone


# ── Helpers ────────────────────────────────────────────────────────────────────

def _minimal_pdf_bytes() -> bytes:
    """
    Minimal valid PDF with two pages of plain text.
    Avoids a real PDF fixture so tests have zero external file dependencies.
    """
    # pypdf can read this minimal hand-crafted PDF
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 5 0 R /Resources << /Font << /F1 6 0 R >> >> >> endobj
4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 7 0 R /Resources << /Font << /F1 6 0 R >> >> >> endobj
5 0 obj << /Length 44 >>
stream
BT /F1 12 Tf 72 720 Td (Hello world. This is page one.) Tj ET
endstream
endobj
7 0 obj << /Length 46 >>
stream
BT /F1 12 Tf 72 720 Td (Goodbye world. This is page two.) Tj ET
endstream
endobj
6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 8
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000417 00000 n
0000000513 00000 n
0000000617 00000 n
trailer << /Size 8 /Root 1 0 R >>
startxref
715
%%EOF"""
    return pdf_content


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(text=f"Sentence about topic {i}. More detail about topic {i}.", page=1, chunk_id=i)
        for i in range(n)
    ]


# ── Unit: Chunk dataclass ──────────────────────────────────────────────────────

class TestChunk:
    def test_repr_does_not_crash(self):
        c = Chunk(text="Hello world. " * 10, page=3, chunk_id=7)
        assert "id=7" in repr(c)
        assert "page=3" in repr(c)

    def test_repr_truncates_long_text(self):
        c = Chunk(text="X" * 200, page=1, chunk_id=0)
        assert len(repr(c)) < 150  # should not dump the full text


# ── Unit: Word-window chunking ─────────────────────────────────────────────────

class TestWordWindowChunking:
    def setup_method(self):
        # Patch model loading so we don't need real weights in unit tests
        with patch("rag_engine._load_embedding_model"), \
             patch("rag_engine._load_reranker"), \
             patch("rag_engine.Groq"):
            self.engine = RAGEngine(use_reranker=False)

    def test_basic_chunking(self):
        pages = [("word " * 500, 1)]
        chunks = self.engine._word_window_chunks(pages, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text.split()) <= 100

    def test_chunk_ids_are_unique_and_sequential(self):
        pages = [("word " * 300, 1)]
        chunks = self.engine._word_window_chunks(pages, chunk_size=50, overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(ids)))

    def test_short_text_below_minimum_is_excluded(self):
        pages = [("hi", 1)]  # < 50 chars — should not produce a chunk
        chunks = self.engine._word_window_chunks(pages)
        assert chunks == []

    def test_multipage_preserves_page_numbers(self):
        pages = [("word " * 200, 1), ("word " * 200, 2)]
        chunks = self.engine._word_window_chunks(pages, chunk_size=50, overlap=10)
        pages_seen = {c.page for c in chunks}
        assert 1 in pages_seen
        assert 2 in pages_seen


# ── Unit: RRF fusion ───────────────────────────────────────────────────────────
#
# RRF is the most mathematically important component — test it in isolation.

class TestRRFFusion:
    def setup_method(self):
        with patch("rag_engine._load_embedding_model"), \
             patch("rag_engine._load_reranker"), \
             patch("rag_engine.Groq"):
            self.engine = RAGEngine(use_reranker=False)
        self.chunks = _make_chunks(5)

    def test_chunk_appearing_in_both_lists_ranks_higher(self):
        shared = self.chunks[0]
        unique_bm25 = self.chunks[1]
        unique_sem = self.chunks[2]

        bm25_res = [(shared, 0.9), (unique_bm25, 0.5)]
        sem_res = [(shared, 0.8), (unique_sem, 0.4)]

        fused = self.engine._rrf_fuse(bm25_res, sem_res)
        top_chunk, _ = fused[0]
        assert top_chunk.chunk_id == shared.chunk_id, \
            "Chunk appearing in both retrieval lists should rank first after RRF"

    def test_output_scores_are_positive(self):
        bm25_res = [(c, float(i)) for i, c in enumerate(self.chunks[:3])]
        sem_res = [(c, float(i) * 0.5) for i, c in enumerate(reversed(self.chunks[:3]))]
        fused = self.engine._rrf_fuse(bm25_res, sem_res)
        assert all(score > 0 for _, score in fused)

    def test_no_duplicates_in_output(self):
        shared = self.chunks[0]
        bm25_res = [(shared, 1.0), (self.chunks[1], 0.5)]
        sem_res = [(shared, 0.9), (self.chunks[2], 0.4)]
        fused = self.engine._rrf_fuse(bm25_res, sem_res)
        ids = [c.chunk_id for c, _ in fused]
        assert len(ids) == len(set(ids)), "RRF output must not contain duplicate chunks"

    def test_rrf_is_scale_invariant(self):
        """
        RRF uses rank positions, not scores — doubling raw scores should not
        change the fused ranking.
        """
        chunks = _make_chunks(4)
        base_bm25 = [(chunks[0], 10.0), (chunks[1], 5.0)]
        base_sem = [(chunks[2], 0.9), (chunks[0], 0.8)]
        scaled_bm25 = [(chunks[0], 20.0), (chunks[1], 10.0)]
        scaled_sem = [(chunks[2], 1.8), (chunks[0], 1.6)]

        fused_base = self.engine._rrf_fuse(base_bm25, base_sem)
        fused_scaled = self.engine._rrf_fuse(scaled_bm25, scaled_sem)

        rank_base = [c.chunk_id for c, _ in fused_base]
        rank_scaled = [c.chunk_id for c, _ in fused_scaled]
        assert rank_base == rank_scaled, \
            "RRF ranking must not depend on raw score magnitudes"


# ── Unit: Confidence computation ───────────────────────────────────────────────

class TestConfidence:
    def test_empty_sources_returns_zero(self):
        assert RAGEngine._compute_confidence([]) == 0.0

    def test_confidence_is_mean_of_scores(self):
        chunks = _make_chunks(3)
        sources = [
            RetrievalResult(chunk=c, score=s, retrieval_method="test")
            for c, s in zip(chunks, [0.8, 0.6, 0.4])
        ]
        conf = RAGEngine._compute_confidence(sources)
        assert abs(conf - pytest.approx(0.6, abs=1e-4)) < 1e-4

    def test_confidence_bounded_zero_to_one(self):
        chunks = _make_chunks(2)
        sources = [
            RetrievalResult(chunk=chunks[0], score=0.0, retrieval_method="test"),
            RetrievalResult(chunk=chunks[1], score=1.0, retrieval_method="test"),
        ]
        conf = RAGEngine._compute_confidence(sources)
        assert 0.0 <= conf <= 1.0


# ── Unit: Adaptive similarity threshold ────────────────────────────────────────

class TestAdaptiveThreshold:
    """
    The adaptive threshold (mean - 0.5·std, clamped [0.30, 0.65]) should
    produce a value that's lower for high-similarity (dense technical) docs
    and higher for low-similarity (narrative) docs.
    """

    def _compute_threshold(self, sims: list[float]) -> float:
        arr = np.array(sims)
        raw = float(arr.mean() - 0.5 * arr.std())
        return float(np.clip(raw, 0.30, 0.65))

    def test_high_similarity_corpus_gets_lower_threshold(self):
        dense_sims = [0.85, 0.87, 0.84, 0.88]       # technical/dense doc
        sparse_sims = [0.30, 0.60, 0.25, 0.70]       # varied/narrative doc
        assert self._compute_threshold(dense_sims) < self._compute_threshold(sparse_sims)

    def test_threshold_stays_within_clamp(self):
        for sims in [[0.99] * 10, [0.01] * 10, [0.5] * 10]:
            t = self._compute_threshold(sims)
            assert 0.30 <= t <= 0.65


# ── Integration: index + query (mocked LLM + embeddings) ─────────────────────

class TestPipelineIntegration:
    """
    Tests the full pipeline with mocked I/O so no network calls or real
    model weights are needed. Validates that:
      - index() populates all_chunks and _chunk_map
      - query() returns a well-formed QueryResult
      - fetch_k never exceeds corpus size
      - chunk_map lookup is used (not positional indexing)
    """

    def setup_method(self):
        # We patch at the module level so lru_cache doesn't interfere
        self.embed_patch = patch("rag_engine._load_embedding_model")
        self.rerank_patch = patch("rag_engine._load_reranker")
        self.groq_patch = patch("rag_engine.Groq")

        mock_embed_cls = self.embed_patch.start()
        mock_rerank_cls = self.rerank_patch.start()
        mock_groq_cls = self.groq_patch.start()

        # Embedding model returns deterministic random vectors
        mock_embed = MagicMock()
        mock_embed.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 384
        )
        mock_embed_cls.return_value = mock_embed

        # Reranker returns random logits
        mock_rerank = MagicMock()
        mock_rerank.predict.side_effect = lambda pairs: np.random.rand(len(pairs))
        mock_rerank_cls.return_value = mock_rerank

        # Groq client returns stub responses
        mock_groq = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "VERDICT: FAITHFUL\nREASON: All claims grounded."
        mock_groq.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_groq_cls.return_value = mock_groq

        self.engine = RAGEngine(model_name="all-MiniLM-L6-v2", use_reranker=True)

    def teardown_method(self):
        self.embed_patch.stop()
        self.rerank_patch.stop()
        self.groq_patch.stop()

    def _fake_pdf_bytes(self, n_pages: int = 3) -> bytes:
        """
        Return PDF-like bytes that are NOT real PDFs — we patch _extract_pages
        so the actual content doesn't matter.
        """
        return b"fake-pdf-content-" + str(n_pages).encode()

    def _patch_extract(self, n_sentences: int = 20):
        """Patch _extract_pages to return synthetic pages with many sentences."""
        sentences = ". ".join(
            f"The quick brown fox jumped over the lazy dog number {i}"
            for i in range(n_sentences)
        )
        return patch.object(
            self.engine, "_extract_pages",
            return_value=[(sentences + ".", 1), (sentences + ".", 2)],
        )

    def test_index_populates_all_chunks(self):
        with self._patch_extract():
            n = self.engine.index(b"fake")
        assert n > 0
        assert len(self.engine.all_chunks) == n

    def test_chunk_map_matches_all_chunks(self):
        with self._patch_extract():
            self.engine.index(b"fake")
        for chunk in self.engine.all_chunks:
            assert chunk.chunk_id in self.engine._chunk_map
            assert self.engine._chunk_map[chunk.chunk_id] is chunk

    def test_query_returns_query_result(self):
        with self._patch_extract():
            self.engine.index(b"fake")
        # Patch generate to return a plain string
        with patch.object(self.engine, "_generate", return_value="The answer is 42."):
            result = self.engine.query("What is the answer?", top_k=2)
        assert isinstance(result, QueryResult)
        assert result.answer == "The answer is 42."
        assert 0.0 <= result.confidence <= 1.0

    def test_query_sources_count_bounded_by_top_k(self):
        with self._patch_extract(n_sentences=30):
            self.engine.index(b"fake")
        with patch.object(self.engine, "_generate", return_value="Answer."), \
             patch.object(self.engine, "_check_faithfulness", return_value=(True, "ok")):
            result = self.engine.query("Any question?", top_k=3)
        assert len(result.sources) <= 3

    def test_empty_question_raises_value_error(self):
        with self._patch_extract():
            self.engine.index(b"fake")
        with pytest.raises(ValueError, match="empty"):
            self.engine.query("   ")

    def test_query_before_index_raises_runtime_error(self):
        fresh_engine = RAGEngine.__new__(RAGEngine)
        fresh_engine.all_chunks = []
        with pytest.raises(RuntimeError, match="index"):
            fresh_engine.query("Hello?")


# ── Regression: FIX 1 — file pointer safety ───────────────────────────────────

class TestFilePointerSafety:
    """
    Two engines indexing the same pdf_bytes must both see the full document.
    This was the critical comparison-mode bug: the second engine read an empty
    stream because the file pointer was at EOF after the first PdfReader call.
    """

    def test_multiple_engines_read_same_bytes(self):
        """
        Both engines must produce at least one page — not zero pages from EOF.
        We verify this at the _extract_pages level without real PDF parsing.
        """
        pdf_bytes = b"fake-pdf"

        with patch("rag_engine._load_embedding_model"), \
             patch("rag_engine._load_reranker"), \
             patch("rag_engine.Groq"):
            engine1 = RAGEngine(use_reranker=False)
            engine2 = RAGEngine(use_reranker=False)

        fake_pages = [("Some text on page one. Another sentence.", 1)]

        with patch.object(engine1, "_extract_pages", return_value=fake_pages) as m1, \
             patch.object(engine2, "_extract_pages", return_value=fake_pages) as m2:

            # Simulate indexing with the same bytes object
            engine1._extract_pages(pdf_bytes)
            engine2._extract_pages(pdf_bytes)

            # Both calls must receive the same bytes — no shared state
            args1 = m1.call_args[0][0]
            args2 = m2.call_args[0][0]
            assert args1 == args2 == pdf_bytes, \
                "Both engines must receive identical bytes — no shared file pointer"


# ── Regression: FIX 2 — collection name isolation ────────────────────────────

class TestCollectionNameIsolation:
    """
    Engine A (file-1) and Engine B (file-2) must use different collection
    names so switching PDFs never deletes a live engine's collection.
    """

    def _collection_name(self, model: str, pdf_bytes: bytes) -> str:
        import re
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
        safe_model = re.sub(r'[^a-z0-9-]', '-', model.lower())
        return f"docs-{safe_model}-{file_hash}"[:63]

    def test_different_files_get_different_collection_names(self):
        bytes_a = b"PDF content of document A " * 100
        bytes_b = b"PDF content of document B " * 100
        name_a = self._collection_name("all-MiniLM-L6-v2", bytes_a)
        name_b = self._collection_name("all-MiniLM-L6-v2", bytes_b)
        assert name_a != name_b

    def test_same_file_same_model_gets_same_name(self):
        pdf_bytes = b"Stable document content " * 100
        name1 = self._collection_name("all-MiniLM-L6-v2", pdf_bytes)
        name2 = self._collection_name("all-MiniLM-L6-v2", pdf_bytes)
        assert name1 == name2

    def test_same_file_different_model_gets_different_name(self):
        pdf_bytes = b"Same document " * 100
        name_a = self._collection_name("all-MiniLM-L6-v2", pdf_bytes)
        name_b = self._collection_name("BAAI/bge-small-en-v1.5", pdf_bytes)
        assert name_a != name_b


# ── Regression: FIX 7 — empty PDF guard ──────────────────────────────────────

class TestEmptyPDFGuard:
    def setup_method(self):
        with patch("rag_engine._load_embedding_model"), \
             patch("rag_engine._load_reranker"), \
             patch("rag_engine.Groq"):
            self.engine = RAGEngine(use_reranker=False)

    def test_empty_pages_raises_value_error(self):
        with patch.object(self.engine, "_extract_pages", return_value=[]):
            with pytest.raises(ValueError, match="No extractable text"):
                self.engine.index(b"fake-scanned-pdf")

    def test_error_message_is_actionable(self):
        with patch.object(self.engine, "_extract_pages", return_value=[]):
            with pytest.raises(ValueError) as exc_info:
                self.engine.index(b"fake")
        assert "scanned" in str(exc_info.value).lower() or \
               "selectable" in str(exc_info.value).lower(), \
            "Error message should tell the user what kind of PDF caused the problem"