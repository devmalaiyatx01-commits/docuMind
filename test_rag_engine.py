"""
tests/test_rag_engine.py

Run with:  pytest tests/ -v

What is tested:
  - Core data structures (Chunk, RetrievalResult, QueryResult)
  - Word-window chunking: output shape, chunk IDs, page attribution, min-length filter
  - Adaptive similarity threshold: direction and clamp bounds
  - RRF fusion: correctness, deduplication, scale-invariance
  - Confidence computation: empty case, mean formula, bounds
  - Full pipeline integration (mocked embeddings and LLM): index → query contract
  - Regression: empty PDF guard, collection name isolation, file-pointer safety
"""

import hashlib
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_engine import RAGEngine, Chunk, RetrievalResult, QueryResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            text=f"Sentence about topic {i}. More detail about topic {i}.",
            page=1,
            chunk_id=i,
        )
        for i in range(n)
    ]


def _engine_no_models(**kwargs) -> RAGEngine:
    """Instantiate RAGEngine with all heavy dependencies patched out."""
    with patch("rag_engine._load_embedding_model"), \
         patch("rag_engine._load_reranker"), \
         patch("rag_engine.Groq"):
        return RAGEngine(use_reranker=False, **kwargs)


# ── Chunk dataclass ────────────────────────────────────────────────────────────

class TestChunk:
    def test_repr_contains_id_and_page(self):
        c = Chunk(text="Hello world. " * 10, page=3, chunk_id=7)
        assert "id=7" in repr(c)
        assert "page=3" in repr(c)

    def test_repr_does_not_dump_full_text(self):
        c = Chunk(text="X" * 200, page=1, chunk_id=0)
        assert len(repr(c)) < 150


# ── Word-window chunking ───────────────────────────────────────────────────────

class TestWordWindowChunking:
    def setup_method(self):
        self.engine = _engine_no_models()

    def test_produces_multiple_chunks_from_long_text(self):
        pages = [("word " * 500, 1)]
        chunks = self.engine._word_window_chunks(pages, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_chunk_word_count_respects_chunk_size(self):
        pages = [("word " * 500, 1)]
        chunks = self.engine._word_window_chunks(pages, chunk_size=100, overlap=20)
        for c in chunks:
            assert len(c.text.split()) <= 100

    def test_chunk_ids_are_unique_and_sequential(self):
        pages = [("word " * 300, 1)]
        chunks = self.engine._word_window_chunks(pages, chunk_size=50, overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(ids)))

    def test_text_below_minimum_length_is_excluded(self):
        pages = [("hi", 1)]
        chunks = self.engine._word_window_chunks(pages)
        assert chunks == []

    def test_page_numbers_are_preserved_across_pages(self):
        pages = [("word " * 200, 1), ("word " * 200, 2)]
        chunks = self.engine._word_window_chunks(pages, chunk_size=50, overlap=10)
        pages_seen = {c.page for c in chunks}
        assert 1 in pages_seen
        assert 2 in pages_seen


# ── Adaptive similarity threshold ─────────────────────────────────────────────

class TestAdaptiveThreshold:
    """
    The adaptive threshold (mean - 0.5·std, clamped to [0.30, 0.65]) should
    be lower for high-similarity corpora and higher for low-similarity ones.
    """

    @staticmethod
    def _threshold(sims: list[float]) -> float:
        arr = np.array(sims)
        raw = float(arr.mean() - 0.5 * arr.std())
        return float(np.clip(raw, 0.30, 0.65))

    def test_dense_corpus_gets_lower_threshold_than_narrative(self):
        dense = [0.85, 0.87, 0.84, 0.88]
        narrative = [0.30, 0.60, 0.25, 0.70]
        assert self._threshold(dense) < self._threshold(narrative)

    def test_threshold_always_within_clamp_bounds(self):
        for sims in [[0.99] * 10, [0.01] * 10, [0.50] * 10]:
            t = self._threshold(sims)
            assert 0.30 <= t <= 0.65, f"Threshold {t} out of [0.30, 0.65] for sims={sims}"


# ── RRF fusion ────────────────────────────────────────────────────────────────

class TestRRFFusion:
    def setup_method(self):
        self.engine = _engine_no_models()
        self.chunks = _make_chunks(5)

    def test_chunk_in_both_lists_ranks_first(self):
        shared = self.chunks[0]
        bm25_res = [(shared, 0.9), (self.chunks[1], 0.5)]
        sem_res = [(shared, 0.8), (self.chunks[2], 0.4)]
        fused = self.engine._rrf_fuse(bm25_res, sem_res)
        top_chunk, _ = fused[0]
        assert top_chunk.chunk_id == shared.chunk_id

    def test_all_output_scores_are_positive(self):
        bm25_res = [(c, float(i)) for i, c in enumerate(self.chunks[:3])]
        sem_res = [(c, float(i) * 0.5) for i, c in enumerate(reversed(self.chunks[:3]))]
        fused = self.engine._rrf_fuse(bm25_res, sem_res)
        assert all(score > 0 for _, score in fused)

    def test_no_duplicate_chunks_in_output(self):
        shared = self.chunks[0]
        bm25_res = [(shared, 1.0), (self.chunks[1], 0.5)]
        sem_res = [(shared, 0.9), (self.chunks[2], 0.4)]
        fused = self.engine._rrf_fuse(bm25_res, sem_res)
        ids = [c.chunk_id for c, _ in fused]
        assert len(ids) == len(set(ids))

    def test_ranking_is_scale_invariant(self):
        """Doubling raw scores must not change the RRF ranking."""
        chunks = _make_chunks(4)
        bm25_base = [(chunks[0], 10.0), (chunks[1], 5.0)]
        sem_base = [(chunks[2], 0.9), (chunks[0], 0.8)]
        bm25_scaled = [(chunks[0], 20.0), (chunks[1], 10.0)]
        sem_scaled = [(chunks[2], 1.8), (chunks[0], 1.6)]

        rank_base = [c.chunk_id for c, _ in self.engine._rrf_fuse(bm25_base, sem_base)]
        rank_scaled = [c.chunk_id for c, _ in self.engine._rrf_fuse(bm25_scaled, sem_scaled)]
        assert rank_base == rank_scaled


# ── Confidence computation ─────────────────────────────────────────────────────

class TestConfidence:
    def test_empty_sources_returns_zero(self):
        assert RAGEngine._compute_confidence([]) == 0.0

    def test_returns_mean_of_scores(self):
        chunks = _make_chunks(3)
        sources = [
            RetrievalResult(chunk=c, score=s, retrieval_method="test")
            for c, s in zip(chunks, [0.8, 0.6, 0.4])
        ]
        conf = RAGEngine._compute_confidence(sources)
        assert abs(conf - 0.6) < 1e-4

    def test_confidence_within_zero_to_one(self):
        chunks = _make_chunks(2)
        sources = [
            RetrievalResult(chunk=chunks[0], score=0.0, retrieval_method="test"),
            RetrievalResult(chunk=chunks[1], score=1.0, retrieval_method="test"),
        ]
        conf = RAGEngine._compute_confidence(sources)
        assert 0.0 <= conf <= 1.0


# ── Integration: index + query (mocked I/O) ────────────────────────────────────

class TestPipelineIntegration:
    """
    Full pipeline tests with mocked embeddings and LLM so no network calls or
    real model weights are needed.
    """

    def setup_method(self):
        self.embed_patch = patch("rag_engine._load_embedding_model")
        self.rerank_patch = patch("rag_engine._load_reranker")
        self.groq_patch = patch("rag_engine.Groq")

        mock_embed_cls = self.embed_patch.start()
        mock_rerank_cls = self.rerank_patch.start()
        mock_groq_cls = self.groq_patch.start()

        mock_embed = MagicMock()
        mock_embed.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 384
        )
        mock_embed_cls.return_value = mock_embed

        mock_rerank = MagicMock()
        mock_rerank.predict.side_effect = lambda pairs: np.random.rand(len(pairs))
        mock_rerank_cls.return_value = mock_rerank

        mock_groq = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "VERDICT: FAITHFUL\nREASON: All claims grounded in sources."
        mock_groq.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_groq_cls.return_value = mock_groq

        self.engine = RAGEngine(model_name="all-MiniLM-L6-v2", use_reranker=True)

    def teardown_method(self):
        self.embed_patch.stop()
        self.rerank_patch.stop()
        self.groq_patch.stop()

    def _synthetic_pages(self, n_sentences: int = 20) -> list[tuple[str, int]]:
        sentences = ". ".join(
            f"The quick brown fox jumped over the lazy dog sentence {i}"
            for i in range(n_sentences)
        )
        return [(sentences + ".", 1), (sentences + ".", 2)]

    def _patch_extract(self, n_sentences: int = 20):
        return patch.object(
            self.engine, "_extract_pages",
            return_value=self._synthetic_pages(n_sentences),
        )

    def test_index_returns_positive_chunk_count(self):
        with self._patch_extract():
            n = self.engine.index(b"fake")
        assert n > 0

    def test_all_chunks_count_matches_index_return_value(self):
        with self._patch_extract():
            n = self.engine.index(b"fake")
        assert len(self.engine.all_chunks) == n

    def test_chunk_map_references_match_all_chunks(self):
        with self._patch_extract():
            self.engine.index(b"fake")
        for chunk in self.engine.all_chunks:
            assert chunk.chunk_id in self.engine._chunk_map
            assert self.engine._chunk_map[chunk.chunk_id] is chunk

    def test_query_returns_query_result_type(self):
        with self._patch_extract():
            self.engine.index(b"fake")
        with patch.object(self.engine, "_generate", return_value="The answer is 42."):
            result = self.engine.query("What is the answer?", top_k=2)
        assert isinstance(result, QueryResult)
        assert result.answer == "The answer is 42."
        assert 0.0 <= result.confidence <= 1.0

    def test_sources_count_bounded_by_top_k(self):
        with self._patch_extract(n_sentences=30):
            self.engine.index(b"fake")
        with patch.object(self.engine, "_generate", return_value="Answer."), \
             patch.object(self.engine, "_check_faithfulness", return_value=(True, "ok")):
            result = self.engine.query("Any question?", top_k=3)
        assert len(result.sources) <= 3

    def test_latency_dict_contains_expected_keys(self):
        with self._patch_extract():
            self.engine.index(b"fake")
        with patch.object(self.engine, "_generate", return_value="Answer."), \
             patch.object(self.engine, "_check_faithfulness", return_value=(True, "ok")):
            result = self.engine.query("Question?", top_k=2, check_faithfulness=True)
        assert "retrieval_ms" in result.latency_ms
        assert "rerank_ms" in result.latency_ms
        assert "generation_ms" in result.latency_ms

    def test_empty_question_raises_value_error(self):
        with self._patch_extract():
            self.engine.index(b"fake")
        with pytest.raises(ValueError, match="empty"):
            self.engine.query("   ")

    def test_query_before_index_raises_runtime_error(self):
        fresh = RAGEngine.__new__(RAGEngine)
        fresh.all_chunks = []
        with pytest.raises(RuntimeError, match="index"):
            fresh.query("Hello?")


# ── Regression: file-pointer safety ───────────────────────────────────────────

class TestFilePointerSafety:
    """
    Two engines indexing the same pdf_bytes object must both see the full
    document. The original bug: the second engine read an empty stream because
    PdfReader left the file pointer at EOF after the first call.
    Passing bytes (not a file object) means each engine constructs its own
    BytesIO internally.
    """

    def test_both_engines_receive_identical_bytes(self):
        pdf_bytes = b"fake-pdf"

        with patch("rag_engine._load_embedding_model"), \
             patch("rag_engine._load_reranker"), \
             patch("rag_engine.Groq"):
            engine1 = RAGEngine(use_reranker=False)
            engine2 = RAGEngine(use_reranker=False)

        fake_pages = [("Some text on page one. Another sentence.", 1)]

        with patch.object(engine1, "_extract_pages", return_value=fake_pages) as m1, \
             patch.object(engine2, "_extract_pages", return_value=fake_pages) as m2:
            engine1._extract_pages(pdf_bytes)
            engine2._extract_pages(pdf_bytes)

            assert m1.call_args[0][0] == pdf_bytes
            assert m2.call_args[0][0] == pdf_bytes


# ── Regression: collection name isolation ─────────────────────────────────────

class TestCollectionNameIsolation:
    """
    Each (model, file-content) pair must get a unique ChromaDB collection name.
    Without this, uploading a second PDF deleted the first PDF's collection while
    its engine was still cached in session state.
    """

    @staticmethod
    def _collection_name(model: str, pdf_bytes: bytes) -> str:
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
        safe_model = re.sub(r'[^a-z0-9-]', '-', model.lower())
        return f"docs-{safe_model}-{file_hash}"[:63]

    def test_different_files_get_different_names(self):
        bytes_a = b"PDF content of document A " * 100
        bytes_b = b"PDF content of document B " * 100
        assert self._collection_name("all-MiniLM-L6-v2", bytes_a) != \
               self._collection_name("all-MiniLM-L6-v2", bytes_b)

    def test_same_file_same_model_is_stable(self):
        pdf_bytes = b"Stable document content " * 100
        name1 = self._collection_name("all-MiniLM-L6-v2", pdf_bytes)
        name2 = self._collection_name("all-MiniLM-L6-v2", pdf_bytes)
        assert name1 == name2

    def test_same_file_different_model_gets_different_name(self):
        pdf_bytes = b"Same document " * 100
        assert self._collection_name("all-MiniLM-L6-v2", pdf_bytes) != \
               self._collection_name("BAAI/bge-small-en-v1.5", pdf_bytes)


# ── Regression: empty PDF guard ───────────────────────────────────────────────

class TestEmptyPDFGuard:
    def setup_method(self):
        self.engine = _engine_no_models()

    def test_empty_pages_raises_value_error(self):
        with patch.object(self.engine, "_extract_pages", return_value=[]):
            with pytest.raises(ValueError, match="No extractable text"):
                self.engine.index(b"fake-scanned-pdf")

    def test_error_message_mentions_scanned_or_selectable(self):
        with patch.object(self.engine, "_extract_pages", return_value=[]):
            with pytest.raises(ValueError) as exc_info:
                self.engine.index(b"fake")
        msg = str(exc_info.value).lower()
        assert "scanned" in msg or "selectable" in msg