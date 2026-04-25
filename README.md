# 🕸️ GraphRAG System — B.Tech Independent Minor Project

A full-stack **Graph-Enhanced Retrieval-Augmented Generation (GraphRAG)** system that extracts knowledge graphs from documents and uses hybrid graph + vector retrieval to answer questions more accurately than standard RAG.

---

## 🏗️ Architecture

```
PDF
 │
 ▼
[ingestion.py] ──► Chunks + Entities/Triplets (spaCy NER)
      │
      ├─────────────────────────────────────┐
      ▼                                     ▼
[graph_engine.py]                 [vector_engine.py]
NetworkX DiGraph                   FAISS Index
(Concept Map + BFS)                (Semantic Memory)
      │                                     │
      └─────────────┬───────────────────────┘
                    ▼
             [retriever.py]
         Hybrid: Vector + 2-hop Graph BFS
                    │
                    ▼
             [app.py] (Streamlit)
     ┌──────────────────────────────────┐
     │  Knowledge Graph │ Chat │ Trace  │
     └──────────────────────────────────┘
```

---

## ✨ Features

- **Interactive Knowledge Graph** — Pyvis-rendered graph visualization with node sizing by betweenness centrality
- **Hybrid Retrieval** — FAISS vector similarity + 2-hop graph BFS traversal, merged and deduplicated
- **BFS Trace Panel** — See every graph hop used to retrieve context in real time
- **Compare Mode** — Side-by-side Standard RAG vs GraphRAG answers
- **Ollama LLM** — Runs `llama3.2` (or any Ollama model) fully locally — no API keys needed

---

## 📁 File Structure

```
graphrag/
├── app.py              ← Streamlit 3-column UI
├── ingestion.py        ← PDF → chunks + NER + triplets
├── graph_engine.py     ← NetworkX graph builder + BFS
├── vector_engine.py    ← FAISS index builder
├── retriever.py        ← Hybrid retrieval + Ollama LLM
├── requirements.txt    ← All dependencies
├── sample_data/
│   ├── computer_networks.pdf   ← Full networking document
│   └── sample_story.pdf        ← Short story demo PDF
└── data/               ← Auto-generated at runtime (gitignored)
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 3. Pull the LLM Model
```bash
ollama pull llama3.2
```

### 4. Run the App
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5. Ingest a Document
1. Upload any PDF via the sidebar
2. Click **⚡ Ingest & Build Index**
3. Ask questions in the chat panel

---

## 📊 Verified Performance

| Stage | Result |
|---|---|
| Ingestion | 25 chunks · 168 entities · 146 triplets |
| Graph | 206 nodes · 145 edges |
| FAISS Index | 25 vectors · 384-dim |
| Hybrid Retrieval | 5 vector + 7 graph → 8 merged chunks |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| NLP / NER | spaCy `en_core_web_trf` |
| Knowledge Graph | NetworkX `DiGraph` |
| Vector Store | FAISS `IndexFlatL2` |
| Embeddings | `all-MiniLM-L6-v2` |
| LLM | Ollama `llama3.2` |
| UI | Streamlit |
| Graph Viz | Pyvis |

---

*B.Tech Independent Minor Project — IIIT Pune*
