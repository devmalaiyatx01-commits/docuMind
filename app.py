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
    "Upload a PDF and ask questions about it. "
    "Answers are grounded in retrieved passages with inline source citations."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    env_key = os.environ.get("GROQ_API_KEY", "")
    if env_key:
        api_key = env_key
        st.success("API key loaded from environment")
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Free at console.groq.com",
        )

    st.divider()
    st.subheader("Mode")
    mode = st.radio(
        "Select mode",
        ["Standard Q&A", "Model Comparison"],
        help="Model Comparison runs two embedding models on the same query side-by-side.",
    )

    if mode == "Standard Q&A":
        selected_model = st.selectbox(
            "Embedding model",
            ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "paraphrase-MiniLM-L3-v2"],
        )

    st.divider()
    st.subheader("Pipeline settings")

    use_semantic_chunk = st.toggle(
        "Semantic chunking",
        value=True,
        help=(
            "ON: splits on topic-boundary similarity drops using an adaptive threshold.\n"
            "OFF: fixed 400-word sliding window (faster, less accurate)."
        ),
    )
    use_hybrid = st.toggle(
        "Hybrid search",
        value=True,
        help="Combines BM25 keyword search with dense vector search via Reciprocal Rank Fusion.",
    )
    use_rerank = st.toggle(
        "Cross-encoder reranking",
        value=True,
        help="Re-scores retrieved candidates with a cross-encoder for higher precision.",
    )
    do_hyde = st.toggle(
        "HyDE query rewriting",
        value=False,
        help=(
            "Generates a hypothetical answer, embeds it, and retrieves against that vector "
            "instead of the raw question. Helps when the question phrasing differs significantly "
            "from document language. (Gao et al., 2022)"
        ),
    )
    check_faith = st.toggle(
        "Hallucination guard",
        value=True,
        help="A second LLM call checks whether the answer is supported by the retrieved sources.",
    )
    top_k = st.slider("Chunks to use for generation", 2, 6, 4)

    st.divider()
    with st.expander("How the pipeline works"):
        st.markdown("""
**Indexing**
1. Extract text from each page, preserving page numbers.
2. Semantic chunking: embed all sentences, group by cosine similarity drops
   (threshold = mean − 0.5·std, clamped to [0.30, 0.65]).
3. Build a BM25 keyword index and a ChromaDB dense vector index.

**Retrieval**
1. Optional HyDE: generate a hypothetical answer and embed it as the query vector.
2. BM25 and dense retrieval each fetch `min(3×top_k, n_chunks)` candidates.
3. Reciprocal Rank Fusion merges the two ranked lists.
4. Cross-encoder reranks the fused candidates to `top_k`.

**Generation**
5. Groq Llama 3.3 70B generates an answer with inline [Source N] citations.
6. Optional faithfulness check: a second LLM call verifies claims against sources.

**Confidence score**
Mean sigmoid(reranker logit) over the top-k chunks. Measures retrieval quality,
not answer quality — a low score means the document may not contain the answer,
not that the pipeline failed.
        """)

    st.divider()
    st.caption("Groq · ChromaDB · SentenceTransformers · Streamlit")


if not api_key:
    st.info("Enter your Groq API key in the sidebar to get started.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

# Read bytes once. Using content hash (not filename) as the cache key means
# two different files named "report.pdf" never collide in session state.
pdf_bytes: bytes = uploaded.getvalue()
file_hash: str = hashlib.md5(pdf_bytes).hexdigest()[:12]

COMPARISON_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]


def get_or_create_engine(model_name: str, with_reranker: bool = True) -> RAGEngine | None:
    """
    Return a cached RAGEngine for this (model, file-content, chunking-mode) combination.

    The cache key includes the chunking setting so toggling semantic chunking on/off
    re-indexes rather than returning a stale engine.
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
                n = engine.index(pdf_bytes)
            except ValueError as exc:
                st.error(f"Could not index PDF: {exc}")
                return None
            except Exception as exc:
                st.error(f"Unexpected error during indexing: {exc}")
                return None

        st.session_state[cache_key] = engine
        st.session_state[f"n_chunks::{cache_key}"] = n

    n = st.session_state[f"n_chunks::{cache_key}"]
    chunk_type = "semantic" if use_semantic_chunk else "fixed-window"
    st.success(f"Indexed **{n} {chunk_type} chunks** using `{model_name}`")
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
    "Ask a question about the document",
    placeholder="e.g. What are the main conclusions? How does the model handle edge cases?",
)
if not question:
    st.stop()


def _confidence_label(conf: float) -> str:
    if conf > 0.70:
        return "🟢 Strong match"
    elif conf > 0.40:
        return "🟡 Partial match"
    elif conf > 0.10:
        return "🔴 Weak match — answer may be absent from the document"
    else:
        return "⬛ No match — query likely outside document scope"


def render_result(result: QueryResult, model_name: str) -> None:
    conf = result.confidence
    faith_icon = "✅ Grounded" if result.faithfulness_ok else "⚠️ Flagged"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Embedding", model_name.split("/")[-1])
    c2.metric("Search", "Hybrid" if use_hybrid else "Semantic")
    c3.metric("Reranking", "ON" if use_rerank else "OFF")
    c4.metric("Retrieval confidence", f"{conf:.4f}")
    c5.metric("Faithfulness", faith_icon)

    conf_msg = _confidence_label(conf)
    if conf <= 0.10:
        st.info(
            f"**{conf_msg}**\n\n"
            "All retrieved chunks scored near zero against this query — "
            "the document likely does not contain a direct answer. "
            "A correct response here is 'not found in document', not a pipeline failure."
        )
    elif conf <= 0.40:
        st.warning(f"**{conf_msg}** — retrieved evidence only partially matches the query.")

    if not result.faithfulness_ok:
        st.error(
            f"⚠️ **Hallucination guard flagged this answer:** {result.faithfulness_note}\n\n"
            "Treat this response with caution."
        )

    if result.rewritten_query:
        with st.expander("HyDE: hypothetical passage used for retrieval"):
            st.caption(
                "This passage was generated by the LLM and embedded as the retrieval vector. "
                "It is never shown to the user or used in the final answer — only for retrieval."
            )
            st.markdown(f"_{result.rewritten_query}_")

    st.subheader("Answer")
    st.markdown(result.answer)

    if result.latency_ms:
        total = sum(result.latency_ms.values())
        with st.expander(f"Latency breakdown — {total:.0f} ms total"):
            for stage, ms in result.latency_ms.items():
                pct = (ms / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 5)
                st.text(f"{stage:<22} {ms:>6.0f} ms  {bar} {pct:.0f}%")

    st.subheader("Retrieved evidence")
    for i, src in enumerate(result.sources):
        score_bar = "█" * max(1, int(src.score * 20))
        label = (
            f"Source {i+1}  ·  Page {src.chunk.page}  ·  "
            f"Score: {src.score:.4f} {score_bar}  ·  [{src.retrieval_method}]"
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
            st.error(f"Invalid query: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    render_result(result, selected_model)

# ── Model Comparison ──────────────────────────────────────────────────────────
else:
    st.subheader("Embedding model comparison")
    st.caption(
        "Same question, same pipeline settings, two embedding models. "
        "Low confidence on both means the query is outside the document's scope."
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
                    st.error(f"{model_name} error: {exc}")
                    continue
            results_map[model_name] = r
            render_result(r, model_name)

    if len(results_map) == 2:
        st.divider()
        st.subheader("Comparison summary")
        m1, m2 = COMPARISON_MODELS
        r1, r2 = results_map[m1], results_map[m2]

        conf_diff = abs(r1.confidence - r2.confidence)
        winner = m1.split("/")[-1] if r1.confidence >= r2.confidence else m2.split("/")[-1]
        loser = m2.split("/")[-1] if r1.confidence >= r2.confidence else m1.split("/")[-1]

        if conf_diff < 0.05:
            st.info(
                f"Both models retrieved with similar confidence "
                f"({r1.confidence:.4f} vs {r2.confidence:.4f}). "
                "For this query the choice of embedding model has little effect."
            )
        else:
            st.info(
                f"**{winner}** retrieved with higher confidence than **{loser}** "
                f"({max(r1.confidence, r2.confidence):.4f} vs "
                f"{min(r1.confidence, r2.confidence):.4f}, Δ={conf_diff:.4f}). "
                "The higher-confidence model's embedding space likely represents "
                "this document's vocabulary better for this query type."
            )

        if not r1.faithfulness_ok and r2.faithfulness_ok:
            flagged, ok = m1.split("/")[-1], m2.split("/")[-1]
            st.warning(
                f"**{flagged}** was flagged by the hallucination guard while **{ok}** was not. "
                "This sometimes happens when lower-quality retrieved evidence causes the LLM "
                "to fill gaps with unsupported claims."
            )
        elif r1.faithfulness_ok and not r2.faithfulness_ok:
            flagged, ok = m2.split("/")[-1], m1.split("/")[-1]
            st.warning(
                f"**{flagged}** was flagged by the hallucination guard while **{ok}** was not."
            )

st.divider()
st.caption("DocuMind · PDF Research Assistant")