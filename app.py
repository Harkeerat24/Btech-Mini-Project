"""
app.py — GraphRAG Streamlit UI
================================
3-column layout: Knowledge Graph | Chat | Source Trace
Sidebar: PDF upload + ingestion controls + system status
Toggle: Compare Mode (Standard RAG vs GraphRAG)

Run: streamlit run app.py
"""

import os
import pickle
import logging
from pathlib import Path

import streamlit as st
import networkx as nx
from pyvis.network import Network

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="GraphRAG System",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

CHUNKS_PATH    = DATA_DIR / "chunks.pkl"
GRAPH_PATH     = DATA_DIR / "graph.pkl"
FAISS_PATH     = DATA_DIR / "faiss_index.bin"
CHUNK_MAP_PATH = DATA_DIR / "chunk_map.pkl"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def artifacts_exist() -> bool:
    return all(p.exists() for p in [CHUNKS_PATH, GRAPH_PATH, FAISS_PATH, CHUNK_MAP_PATH])


@st.cache_resource
def get_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_retriever(model_name: str, provider: str):
    from retriever import GraphRAGRetriever
    return GraphRAGRetriever(ollama_model=model_name, llm_provider=provider, verbose=True)


# ─── Build Pyvis graph HTML ───────────────────────────────────────────────────
def build_pyvis_html(G: nx.DiGraph, highlight_nodes: list = None) -> str:
    net = Network(height="700px", width="100%", directed=True,
                  bgcolor="#0d0d1a", font_color="#e0e0f0")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    highlight_nodes = set(highlight_nodes or [])
    color_map = {
        "PROTOCOL":  "#3498DB",
        "ALGORITHM": "#2ECC71",
        "CONCEPT":   "#9B59B6",
        "PERSON":    "#F39C12",
        "ORG":       "#1ABC9C",
        "PRODUCT":   "#E67E22",
        "GPE":       "#E74C3C",
        "LOC":       "#E74C3C",
        "DEFAULT":   "#95A5A6",
    }

    for node, attrs in G.nodes(data=True):
        label = attrs.get("label", node)[:20]
        etype = attrs.get("entity_type", "DEFAULT")
        color = "#FFD700" if node in highlight_nodes else color_map.get(etype, "#95A5A6")
        size  = 12 + attrs.get("betweenness_centrality", 0) * 80
        size  = min(max(size, 10), 50)
        title = (
            f"<b>{attrs.get('label', node)}</b><br>"
            f"Type: {etype}<br>"
            f"Centrality: {attrs.get('betweenness_centrality', 0):.4f}<br>"
            f"Chunks: {len(attrs.get('chunk_ids', []))}"
        )
        net.add_node(node, label=label, color=color, size=size, title=title)

    for src, dst, edata in G.edges(data=True):
        if src in G.nodes() and dst in G.nodes():
            net.add_edge(src, dst, title=edata.get("predicate", ""),
                         width=edata.get("weight", 1))

    # Write to a fixed persistent file (avoids WinError 32 / file-lock on Windows)
    render_path = DATA_DIR / "graph_render.html"
    net.save_graph(str(render_path))
    with open(render_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕸️ GraphRAG")
    st.markdown("*Graph-Enhanced Retrieval System*")
    st.divider()

    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    ollama_model = st.selectbox(
        "LLM Model",
        ["llama3.2", "phi3", "llama3", "mistral"],
        index=0,
    )
    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "openai"],
        index=0,
    )
    chunk_size    = st.slider("Chunk Size",    256, 1024, 512, 64)
    chunk_overlap = st.slider("Chunk Overlap",  32,  256,  64, 16)

    ingest_btn = st.button("⚡ Ingest & Build Index", use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ Retrieval")
    compare_mode = st.toggle("📊 Compare Mode (RAG vs GraphRAG)", value=False)
    top_k = st.slider("Vector Top-K", 3, 10, 5)

    st.divider()
    st.markdown("### 🔍 System Status")
    if artifacts_exist():
        st.success("✅ Index Ready")
        with open(GRAPH_PATH, "rb") as f:
            _g = pickle.load(f)
        st.caption(f"Nodes: {_g.number_of_nodes()} | Edges: {_g.number_of_edges()}")
        with open(CHUNKS_PATH, "rb") as f:
            _c = pickle.load(f)
        st.caption(f"Chunks: {len(_c)}")
    else:
        st.warning("⚠️ No index — upload & ingest a PDF first.")


# ─── Ingestion Logic ──────────────────────────────────────────────────────────
if ingest_btn:
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF first.")
    else:
        pdf_path = UPLOAD_DIR / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("🔄 Running ingestion pipeline..."):
            progress = st.sidebar.progress(0, text="Loading spaCy model...")
            try:
                from ingestion import ingest
                progress.progress(20, text="Chunking & NER...")
                chunks, triplets = ingest(str(pdf_path), chunk_size, chunk_overlap)

                progress.progress(50, text="Building graph...")
                from graph_engine import build as build_graph
                build_graph()

                progress.progress(65, text="Building FAISS index...")
                from vector_engine import build as build_vector
                build_vector()

                progress.progress(80, text="Building keyword index...")
                from retriever import build_lexical_index
                build_lexical_index()

                progress.progress(100, text="Done!")
                st.sidebar.success(
                    f"✅ Ingested! Chunks: {len(chunks)} | Triplets: {len(triplets)}"
                )
                st.cache_resource.clear()
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {e}")
                log.exception("Ingestion error")


# ─── Main Title ───────────────────────────────────────────────────────────────
st.title("🕸️ GraphRAG System")
st.caption("Graph-Enhanced Retrieval-Augmented Generation")
st.divider()


# ─── Main 3-column layout ─────────────────────────────────────────────────────
col_graph, col_chat, col_source = st.columns([1.1, 1.3, 1.0], gap="medium")


# ── Column 1: Knowledge Graph ──────────────────────────────────────────────────
with col_graph:
    st.subheader("🗺️ Knowledge Graph")

    if not artifacts_exist():
        st.info("Upload a PDF and click **Ingest** to build the graph.")
    else:
        G = get_graph()
        highlight = st.session_state.get("last_traversed_nodes", [])

        with st.spinner("Rendering graph..."):
            try:
                graph_html = build_pyvis_html(G, highlight_nodes=highlight)
                st.components.v1.html(graph_html, height=720, scrolling=False)
            except Exception as e:
                st.error(f"Graph render error: {e}")

        st.caption("🟡 Highlighted: query-traversed nodes | Drag to explore")

        with st.expander("🏆 Top Hub Nodes"):
            nodes_sorted = sorted(
                G.nodes(data=True),
                key=lambda x: x[1].get("betweenness_centrality", 0),
                reverse=True,
            )[:8]
            for node, attrs in nodes_sorted:
                etype = attrs.get("entity_type", "DEFAULT")
                btw   = attrs.get("betweenness_centrality", 0)
                st.markdown(
                    f"**{attrs.get('label', node)}** — `{etype}` — centrality: `{btw:.4f}`"
                )


# ── Column 2: Chat Interface ───────────────────────────────────────────────────
with col_chat:
    st.subheader("💬 Chat Interface")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None
    if "last_compare" not in st.session_state:
        st.session_state.last_compare = None

    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.messages:
            st.info("Ask a question about your document 👇")
        for msg in st.session_state.messages:
            role = msg["role"]
            with st.chat_message(role):
                st.write(msg["content"])

    user_query = st.chat_input(
        "Ask about your document...", disabled=not artifacts_exist()
    )

    if user_query:
        if not artifacts_exist():
            st.error("Please ingest a PDF first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.spinner("🧠 Retrieving + Generating..."):
                try:
                    retriever = get_retriever(ollama_model, llm_provider)
                    retriever.top_k_vector = top_k
                    result = retriever.query(user_query, compare_mode=compare_mode)

                    answer = result["graphrag_answer"]
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.last_trace   = result["trace"]
                    st.session_state.last_compare = result if compare_mode else None
                    st.session_state.last_traversed_nodes = result["trace"].get(
                        "traversed_nodes", []
                    )

                except Exception as e:
                    err = f"Error: {e}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
                    log.exception("Query error")

            st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.session_state.last_compare = None
            st.session_state.last_traversed_nodes = []
            st.rerun()


# ── Column 3: Source Context + Retrieval Trace ────────────────────────────────
with col_source:
    st.subheader("🔍 Source & Trace")

    trace          = st.session_state.get("last_trace")
    compare_result = st.session_state.get("last_compare")

    if trace is None:
        st.info("Retrieval trace will appear here after your first query.")
    else:
        entities = trace.get("identified_entities", [])
        matched  = trace.get("matched_graph_nodes", [])
        trav     = trace.get("traversed_nodes", [])
        vcids    = trace.get("vector_chunk_ids", [])
        gcids    = trace.get("graph_chunk_ids", [])
        fcids    = trace.get("final_chunk_ids", [])

        st.markdown("**🔎 Retrieval Trace**")
        st.code(
            f"Entities detected : {', '.join(entities) if entities else 'none'}\n"
            f"Graph nodes matched: {', '.join(matched) if matched else 'none'}\n"
            f"Nodes traversed (BFS): {len(trav)}\n"
            f"Vector chunks  : {len(vcids)}\n"
            f"Graph chunks   : {len(gcids)}\n"
            f"Final merged   : {len(fcids)}",
            language="yaml",
        )

        if trav:
            st.markdown("**Traversed nodes:**")
            st.write(", ".join(trav[:15]))

        st.divider()

        with st.expander("📄 Source Chunks", expanded=True):
            ctx_text = trace.get("context", "")
            if ctx_text:
                for sec in ctx_text.split("\n\n")[:6]:
                    st.info(sec)
            else:
                st.caption("No context available.")

        if compare_mode and compare_result:
            st.divider()
            st.subheader("📊 Compare Mode")
            st.caption("GraphRAG uses richer context through graph traversal.")

            tab_rag, tab_graphrag = st.tabs(["⚡ Standard RAG", "🕸️ GraphRAG"])

            with tab_rag:
                std_answer = compare_result.get("standard_rag_answer", "N/A")
                st.write(std_answer)
                std_chunks = trace.get("standard_rag_chunks", [])
                st.caption(f"Based on {len(std_chunks)} vector chunks only.")

            with tab_graphrag:
                grag_answer = compare_result.get("graphrag_answer", "N/A")
                st.write(grag_answer)
                st.caption(f"Based on {len(fcids)} hybrid chunks (vector + graph).")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "GraphRAG System · B.Tech Independent Minor Project · "
    "spaCy + NetworkX + FAISS + Ollama + Streamlit"
)
