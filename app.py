# app.py — DocuMind UI
#
# ── Changelog from v1 ─────────────────────────────────────────────────────────
#
#   FIX 1  pdf_bytes = uploaded.getvalue() called ONCE at the top of the script.
#          Previously, `uploaded` (a Streamlit UploadedFile / file-like object)
#          was passed directly to engine.index(). PdfReader reads the stream and
#          leaves the file pointer at EOF. In Model Comparison mode, the second
#          engine.index() call read an empty stream → empty all_chunks → silent
#          wrong answers. Now bytes are captured once; each engine constructs its
#          own BytesIO internally (see rag_engine.py FIX 1).
#
#   FIX 9  Session state key now uses MD5(pdf_bytes) not just filename.
#          Two different files named "report.pdf" previously shared one cache
#          slot and returned stale results. The content hash is the true identity.
#
#   NEW    try/except wraps both index() and query() calls.
#          ValueError (empty PDF, empty question) and RuntimeError (API failure
#          after retries) are now shown as st.error() messages instead of
#          crashing the page with a Python traceback.
#
#   NEW    Logging configured at INFO level so rag_engine's stage logs appear
#          in the terminal during development.
# ──────────────────────────────────────────────────────────────────────────────

import hashlib
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine, QueryResult

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 DocuMind")
st.markdown(
    "**Advanced PDF Research Assistant** — "
    "Semantic Chunking · Hybrid Search · RRF · Cross-Encoder Reranking · "
    "HyDE Query Rewriting · Hallucination Guard"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    env_key = os.environ.get("GROQ_API_KEY", "")
    if env_key:
        api_key = env_key
        st.success("✅ API key loaded from environment")
    else:
        api_key = st.text_input(
            "Groq API Key", type="password", help="Free at console.groq.com"
        )

    st.divider()
    st.subheader("🔬 Mode")
    mode = st.radio(
        "Select Mode",
        ["Standard Q&A", "Model Comparison"],
        help="Model Comparison runs two embedding models side-by-side",
    )

    if mode == "Standard Q&A":
        selected_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "paraphrase-MiniLM-L3-v2"],
        )

    st.divider()
    st.subheader("🚀 Pipeline Settings")

    use_semantic_chunk = st.toggle(
        "Semantic Chunking",
        value=True,
        help=(
            "ON: split on topic-boundary similarity drops (smarter, adaptive threshold)\n"
            "OFF: fixed 400-word sliding window (faster)"
        ),
    )
    use_hybrid = st.toggle("Hybrid Search (BM25 + Semantic)", value=True)
    use_rerank = st.toggle("Cross-Encoder Reranking", value=True)
    do_hyde = st.toggle(
        "HyDE Query Rewriting",
        value=False,
        help=(
            "Hypothetical Document Embedding: generates a hypothetical answer, "
            "embeds it, and retrieves against that vector instead of the raw query. "
            "(Gao et al. 2022 — arXiv:2212.10496)"
        ),
    )
    check_faith = st.toggle(
        "Hallucination Guard",
        value=True,
        help="Second LLM call checks whether the answer is grounded in retrieved sources",
    )
    top_k = st.slider("Final chunks to use", 2, 6, 4)

    st.divider()
    with st.expander("ℹ️ How the pipeline works"):
        st.markdown("""
**Indexing:**
1. Extract text, preserve page numbers
2. Semantic chunking: embed sentences → group by cosine-similarity
   (adaptive threshold: mean − 0.5·std, clamped [0.30, 0.65])
3. Build BM25 (keyword) + ChromaDB (dense) indexes in parallel

**Retrieval:**
1. Optional HyDE: generate hypothetical answer, embed it
2. BM25 keyword retrieval + semantic retrieval (fetch_k = min(3×top_k, n_chunks))
3. RRF rank fusion (Cormack et al. 2009)
4. Cross-encoder reranking (joint query+doc scoring)

**Generation:**
5. Groq Llama 3.3 70B with page-aware citations
6. Hallucination guard: second LLM checks answer faithfulness

**Confidence score:** mean sigmoid(reranker logit) across top-k chunks.
Measures retrieval quality, not answer quality — low confidence on a
well-formed answer means the document simply doesn't contain the answer.
        """)

    st.divider()
    st.caption("Built with Groq · ChromaDB · SentenceTransformers · Streamlit")


if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

uploaded = st.file_uploader("📄 Upload a PDF document", type=["pdf"])
if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

# FIX 1 + FIX 9: read bytes once; derive a content-hash for the cache key.
# Using getvalue() means the file pointer is never shared between engines.
# Using the MD5 hash (not filename) means two different files named "report.pdf"
# never collide in session state.
pdf_bytes: bytes = uploaded.getvalue()
file_hash: str = hashlib.md5(pdf_bytes).hexdigest()[:12]

COMPARISON_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]


def get_or_create_engine(
    model_name: str,
    with_reranker: bool = True,
) -> RAGEngine | None:
    """
    Return a cached RAGEngine for this (model, file-content) pair.

    Cache key uses the file content hash (not filename) so different files
    with the same name produce different engine instances.
    """
    cache_key = f"engine::{model_name}::{file_hash}::{use_semantic_chunk}"
    if cache_key not in st.session_state:
        with st.spinner(f"Indexing with `{model_name}`..."):
            try:
                engine = RAGEngine(
                    model_name=model_name,
                    use_reranker=with_reranker,
                    semantic_chunk=use_semantic_chunk,
                )
                # FIX 1: pass bytes, not the file object
                n = engine.index(pdf_bytes)
            except ValueError as exc:
                st.error(f"❌ Could not index PDF: {exc}")
                return None
            except Exception as exc:
                st.error(f"❌ Unexpected error during indexing: {exc}")
                return None

        st.session_state[cache_key] = engine
        st.session_state[f"n_chunks::{cache_key}"] = n

    n = st.session_state[f"n_chunks::{cache_key}"]
    chunk_type = "semantic" if use_semantic_chunk else "fixed-window"
    st.success(
        f"✅ Indexed **{n} {chunk_type} chunks** using `{model_name}`"
    )
    return st.session_state[cache_key]


if mode == "Standard Q&A":
    engine = get_or_create_engine(selected_model, with_reranker=use_rerank)
    if engine is None:
        st.stop()
else:
    engines: dict[str, RAGEngine] = {}
    for m in COMPARISON_MODELS:
        eng = get_or_create_engine(m)
        if eng is None:
            st.stop()
        engines[m] = eng

st.divider()

question = st.text_input(
    "❓ Ask a question about the document",
    placeholder="e.g. What is the formula for RRF? How does B-spline continuity work?",
)
if not question:
    st.stop()


def _confidence_label(conf: float) -> str:
    """
    Translate raw confidence into a human-readable signal.

    Low confidence ≠ wrong answer. It means the retrieved evidence weakly
    matches the query. The correct response is 'this isn't in the document',
    which is itself a correct and faithful answer.
    """
    if conf > 0.70:
        return "🟢 Strong match"
    elif conf > 0.40:
        return "🟡 Partial match"
    elif conf > 0.10:
        return "🔴 Weak match — answer may be absent from document"
    else:
        return "⬛ No match — query likely outside document scope"


def render_result(result: QueryResult, model_name: str) -> None:
    """Render a QueryResult — used by both Standard and Comparison modes."""
    conf = result.confidence
    faith_icon = "✅ Grounded" if result.faithfulness_ok else "⚠️ Flagged"

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Embedding", model_name.split("/")[-1])
    c2.metric("Search", "Hybrid" if use_hybrid else "Semantic")
    c3.metric("Reranking", "ON ✅" if use_rerank else "OFF")
    c4.metric("Retrieval Confidence", f"{conf:.4f}")
    c5.metric("Faithfulness", faith_icon)

    # Confidence interpretation
    conf_msg = _confidence_label(conf)
    if conf <= 0.10:
        st.info(
            f"**{conf_msg}**  \n"
            "The reranker scored all retrieved chunks near zero against this query — "
            "the document likely does not contain a direct answer. "
            "A correct response here is *'not found in document'*, not a pipeline failure. "
            "Try a more specific question about content visible in the evidence below."
        )
    elif conf <= 0.40:
        st.warning(f"**{conf_msg}** — retrieved evidence partially matches the query.")

    # Faithfulness warning
    if not result.faithfulness_ok:
        st.error(
            f"⚠️ **Hallucination Guard:** {result.faithfulness_note}  \n"
            "Treat this answer with caution."
        )

    # HyDE trace
    if result.rewritten_query:
        with st.expander("🔬 HyDE: hypothetical passage used for retrieval"):
            st.caption(
                "Generated by the LLM and embedded as the retrieval vector. "
                "Never shown to the user or used in the answer — only for retrieval."
            )
            st.markdown(f"_{result.rewritten_query}_")

    # Answer
    st.subheader("📝 Answer")
    st.markdown(result.answer)

    # Latency
    if result.latency_ms:
        total = sum(result.latency_ms.values())
        with st.expander(f"⏱️ Latency breakdown — total {total:.0f} ms"):
            for stage, ms in result.latency_ms.items():
                pct = (ms / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 5)
                st.text(f"{stage:<22} {ms:>6.0f} ms  {bar} {pct:.0f}%")

    # Evidence
    st.subheader("🔎 Retrieved Evidence")
    for i, src in enumerate(result.sources):
        score_bar = "█" * max(1, int(src.score * 20))
        label = (
            f"Source {i+1}  ·  Page {src.chunk.page}  ·  "
            f"Relevance: {src.score:.4f} {score_bar}  ·  [{src.retrieval_method}]"
        )
        with st.expander(label):
            st.markdown(f"```\n{src.chunk.text}\n```")


# ── Standard Q&A ──────────────────────────────────────────────────────────────
if mode == "Standard Q&A":
    with st.spinner("Running pipeline..."):
        try:
            result = engine.query(
                question,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                do_hyde=do_hyde,
                check_faithfulness=check_faith,
            )
        except ValueError as exc:
            st.error(f"❌ Invalid query: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Pipeline error: {exc}")
            st.stop()

    st.subheader("🔍 Pipeline Trace")
    render_result(result, selected_model)

# ── Model Comparison ──────────────────────────────────────────────────────────
else:
    st.subheader("🔬 Embedding Model Comparison")
    st.caption(
        "Same question · Same pipeline settings · Different embedding models. "
        "Low confidence on both = query is outside document scope, not a bug."
    )

    results_map: dict[str, QueryResult] = {}
    col1, col2 = st.columns(2)

    for col, model_name in zip([col1, col2], COMPARISON_MODELS):
        with col:
            st.subheader(f"`{model_name}`")
            with st.spinner(f"Querying {model_name}..."):
                try:
                    r = engines[model_name].query(
                        question,
                        top_k=top_k,
                        use_hybrid=use_hybrid,
                        use_rerank=True,
                        do_hyde=do_hyde,
                        check_faithfulness=check_faith,
                    )
                except Exception as exc:
                    st.error(f"❌ {model_name} pipeline error: {exc}")
                    continue
            results_map[model_name] = r
            render_result(r, model_name)

    # Cross-model insight
    if len(results_map) == 2:
        st.divider()
        st.subheader("📊 Model Comparison Insight")
        m1, m2 = COMPARISON_MODELS
        r1, r2 = results_map[m1], results_map[m2]

        conf_diff = abs(r1.confidence - r2.confidence)
        winner = m1.split("/")[-1] if r1.confidence > r2.confidence else m2.split("/")[-1]
        loser = m2.split("/")[-1] if r1.confidence > r2.confidence else m1.split("/")[-1]

        if conf_diff < 0.05:
            st.info(
                f"Both models retrieved with similar confidence "
                f"({r1.confidence:.4f} vs {r2.confidence:.4f}). "
                "For this query, the choice of embedding model doesn't significantly "
                "affect retrieval quality."
            )
        else:
            st.info(
                f"**{winner}** retrieved with higher confidence than **{loser}** "
                f"({max(r1.confidence, r2.confidence):.4f} vs "
                f"{min(r1.confidence, r2.confidence):.4f}, Δ={conf_diff:.4f}). "
                "This suggests the higher-confidence model's embedding space better "
                "represents this document's vocabulary for this query type."
            )

        faith_both = r1.faithfulness_ok and r2.faithfulness_ok
        faith_neither = not r1.faithfulness_ok and not r2.faithfulness_ok
        if not faith_both and not faith_neither:
            flagged = m1.split("/")[-1] if not r1.faithfulness_ok else m2.split("/")[-1]
            ok = m2.split("/")[-1] if not r1.faithfulness_ok else m1.split("/")[-1]
            st.warning(
                f"**{flagged}** was flagged by the Hallucination Guard while **{ok}** was not. "
                "This can indicate the flagged model retrieved lower-quality evidence, "
                "causing the LLM to fill gaps with invented content."
            )

st.divider()
st.caption("DocuMind · Advanced RAG Portfolio Project")