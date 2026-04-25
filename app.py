"""
app.py — GraphRAG Streamlit UI
================================
3-column layout: Knowledge Graph | Chat | Source Trace
Sidebar: PDF upload + ingestion controls
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


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* Column headers */
.col-header {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid rgba(167,139,250,0.3);
    padding-bottom: 0.4rem;
}

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 0.4rem 2rem;
    font-size: 0.93rem;
}
.chat-bot {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px 12px 12px 2px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 2rem 0.4rem 0;
    font-size: 0.93rem;
}

/* Trace box */
.trace-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 10px;
    padding: 0.8rem;
    font-size: 0.8rem;
    font-family: 'Courier New', monospace;
    color: #a5f3fc;
    margin-bottom: 0.6rem;
}

/* Entity badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

/* Compare cards */
.compare-rag  { border-left: 3px solid #f59e0b; }
.compare-graphrag { border-left: 3px solid #10b981; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Status indicators */
.status-ok  { color: #10b981; font-weight: 600; }
.status-bad { color: #f87171; font-weight: 600; }

/* Graph legend */
.legend-item {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.8rem; margin: 3px 0;
}
.legend-dot {
    width: 12px; height: 12px;
    border-radius: 50%; display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper: check artifacts exist ────────────────────────────────────────────
def artifacts_exist() -> bool:
    return all(p.exists() for p in [CHUNKS_PATH, GRAPH_PATH, FAISS_PATH, CHUNK_MAP_PATH])


# ─── Helper: load graph (cached) ──────────────────────────────────────────────
@st.cache_resource
def get_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


# ─── Helper: load retriever (cached) ──────────────────────────────────────────
@st.cache_resource
def get_retriever(model_name: str):
    from retriever import GraphRAGRetriever
    return GraphRAGRetriever(ollama_model=model_name, verbose=True)


# ─── Build Pyvis graph HTML ───────────────────────────────────────────────────
def build_pyvis_html(G: nx.DiGraph, highlight_nodes: list = None) -> str:
    net = Network(height="480px", width="100%", directed=True, bgcolor="#0d0d1a", font_color="#e0e0f0")
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
            net.add_edge(src, dst, title=edata.get("predicate", ""), width=edata.get("weight", 1))

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

    # PDF Upload
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    ollama_model = st.selectbox(
        "LLM Model",
        ["llama3.2", "phi3", "llama3", "mistral"],
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
    # System status
    st.markdown("### 🔍 System Status")
    if artifacts_exist():
        st.markdown('<span class="status-ok">✅ Index Ready</span>', unsafe_allow_html=True)
        with open(GRAPH_PATH, "rb") as f:
            _g = pickle.load(f)
        st.caption(f"Nodes: {_g.number_of_nodes()} | Edges: {_g.number_of_edges()}")
        with open(CHUNKS_PATH, "rb") as f:
            _c = pickle.load(f)
        st.caption(f"Chunks: {len(_c)}")
    else:
        st.markdown('<span class="status-bad">⚠️ No index — upload & ingest</span>', unsafe_allow_html=True)

    st.divider()
    # Color legend
    st.markdown("### 🎨 Graph Legend")
    legend = [
        ("PROTOCOL",  "#3498DB"),
        ("ALGORITHM", "#2ECC71"),
        ("CONCEPT",   "#9B59B6"),
        ("PERSON",    "#F39C12"),
        ("ORG",       "#1ABC9C"),
        ("DEFAULT",   "#95A5A6"),
    ]
    for name, color in legend:
        st.markdown(
            f'<div class="legend-item"><span class="legend-dot" style="background:{color}"></span>{name}</div>',
            unsafe_allow_html=True,
        )


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

                progress.progress(75, text="Building FAISS index...")
                from vector_engine import build as build_vector
                build_vector()

                progress.progress(100, text="Done!")
                st.sidebar.success(f"✅ Ingested! Chunks: {len(chunks)} | Triplets: {len(triplets)}")
                # Clear cached resources so they reload
                st.cache_resource.clear()
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {e}")
                log.exception("Ingestion error")


# ─── Main Title ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; background: linear-gradient(135deg, #a78bfa, #60a5fa); "
    "-webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.2rem;'>"
    "🕸️ GraphRAG System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:0;'>"
    "Graph-Enhanced Retrieval-Augmented Generation</p>",
    unsafe_allow_html=True,
)
st.divider()


# ─── Main 3-column layout ─────────────────────────────────────────────────────
col_graph, col_chat, col_source = st.columns([1.1, 1.3, 1.0], gap="medium")

# ── Column 1: Knowledge Graph ──────────────────────────────────────────────────
with col_graph:
    st.markdown('<div class="col-header">🗺️ Knowledge Graph</div>', unsafe_allow_html=True)

    if not artifacts_exist():
        st.info("Upload a PDF and click **Ingest** to build the graph.")
    else:
        G = get_graph()
        highlight = st.session_state.get("last_traversed_nodes", [])

        with st.spinner("Rendering graph..."):
            try:
                graph_html = build_pyvis_html(G, highlight_nodes=highlight)
                st.components.v1.html(graph_html, height=500, scrolling=False)
            except Exception as e:
                st.error(f"Graph render error: {e}")

        st.caption(f"🟡 Highlighted: query-traversed nodes | Drag to explore")

        # Top hub nodes
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
    st.markdown('<div class="col-header">💬 Chat Interface</div>', unsafe_allow_html=True)

    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None
    if "last_compare" not in st.session_state:
        st.session_state.last_compare = None

    # Chat history
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "<div style='text-align:center; color:#64748b; margin-top:4rem;'>"
                "Ask a question about your document 👇</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            role  = msg["role"]
            content = msg["content"]
            css_class = "chat-user" if role == "user" else "chat-bot"
            icon = "🧑" if role == "user" else "🤖"
            st.markdown(
                f'<div class="{css_class}">{icon} {content}</div>',
                unsafe_allow_html=True,
            )

    # Input
    user_query = st.chat_input("Ask about your document...", disabled=not artifacts_exist())

    if user_query:
        if not artifacts_exist():
            st.error("Please ingest a PDF first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.spinner("🧠 Retrieving + Generating..."):
                try:
                    retriever = get_retriever(ollama_model)
                    retriever.top_k_vector = top_k
                    result = retriever.query(user_query, compare_mode=compare_mode)

                    answer = result["graphrag_answer"]
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_trace   = result["trace"]
                    st.session_state.last_compare = result if compare_mode else None

                    # Store traversed nodes for graph highlight
                    st.session_state.last_traversed_nodes = result["trace"].get("traversed_nodes", [])

                except Exception as e:
                    err = f"Error: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    log.exception("Query error")

            st.rerun()

    # Clear history button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.session_state.last_compare = None
            st.session_state.last_traversed_nodes = []
            st.rerun()


# ── Column 3: Source Context + Retrieval Trace ────────────────────────────────
with col_source:
    st.markdown('<div class="col-header">🔍 Source & Trace</div>', unsafe_allow_html=True)

    trace = st.session_state.get("last_trace")
    compare_result = st.session_state.get("last_compare")

    if trace is None:
        st.info("Retrieval trace will appear here after your first query.")
    else:
        # Retrieval trace
        entities = trace.get("identified_entities", [])
        matched  = trace.get("matched_graph_nodes", [])
        trav     = trace.get("traversed_nodes", [])
        vcids    = trace.get("vector_chunk_ids", [])
        gcids    = trace.get("graph_chunk_ids", [])
        fcids    = trace.get("final_chunk_ids", [])

        st.markdown("**🔎 Retrieval Trace**")

        trace_html = (
            f"<div class='trace-box'>"
            f"<b>Entities detected:</b> {', '.join(entities) if entities else 'none'}<br>"
            f"<b>Graph nodes matched:</b> {', '.join(matched) if matched else 'none'}<br>"
            f"<b>Nodes traversed (BFS):</b> {len(trav)}<br>"
            f"<b>Vector chunks:</b> {len(vcids)}<br>"
            f"<b>Graph chunks:</b> {len(gcids)}<br>"
            f"<b>Final merged chunks:</b> {len(fcids)}"
            f"</div>"
        )
        st.markdown(trace_html, unsafe_allow_html=True)

        # Traversed node badges
        if trav:
            badge_html = "<b>Traversed nodes:</b><br>"
            for n in trav[:15]:
                badge_html += f'<span class="badge" style="background:rgba(99,102,241,0.25);border:1px solid #6366f1;">{n}</span>'
            st.markdown(badge_html, unsafe_allow_html=True)

        st.divider()

        # Source chunks
        with st.expander("📄 Source Chunks", expanded=True):
            ctx_text = trace.get("context", "")
            if ctx_text:
                sections = ctx_text.split("\n\n")
                for sec in sections[:6]:
                    st.markdown(
                        f'<div class="glass-card" style="font-size:0.78rem; color:#cbd5e1;">{sec}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No context available.")

        # Compare Mode Panel
        if compare_mode and compare_result:
            st.divider()
            st.markdown("### 📊 Compare Mode")
            st.caption("GraphRAG uses richer context through graph traversal.")

            tab_rag, tab_graphrag = st.tabs(["⚡ Standard RAG", "🕸️ GraphRAG"])

            with tab_rag:
                std_answer = compare_result.get("standard_rag_answer", "N/A")
                st.markdown(
                    f'<div class="glass-card compare-rag">{std_answer}</div>',
                    unsafe_allow_html=True,
                )
                std_chunks = trace.get("standard_rag_chunks", [])
                st.caption(f"Based on {len(std_chunks)} vector chunks only.")

            with tab_graphrag:
                grag_answer = compare_result.get("graphrag_answer", "N/A")
                st.markdown(
                    f'<div class="glass-card compare-graphrag">{grag_answer}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Based on {len(fcids)} hybrid chunks (vector + graph).")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.78rem;'>"
    "GraphRAG System · B.Tech Independent Minor Project · "
    "spaCy + NetworkX + FAISS + Ollama + Streamlit</p>",
    unsafe_allow_html=True,
)
