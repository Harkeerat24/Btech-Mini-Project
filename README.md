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
         Hybrid: Vector + Inverted Index + 2-hop Graph BFS
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
- **Hybrid Retrieval** — FAISS vector similarity + inverted-index keyword retrieval + 2-hop graph BFS traversal, merged and deduplicated
- **Knowledge Graph Construction** — NER entities become graph nodes; extracted S-V-O triplets and entity co-occurrence become edges
- **BFS Trace Panel** — See every graph hop used to retrieve context in real time
- **Compare Mode** — Side-by-side Standard RAG vs GraphRAG answers
- **Ollama / OpenAI LLM** — Runs local Llama-style models through Ollama or OpenAI through `OPENAI_API_KEY`

---

## 📁 File Structure

```
graphrag/
├── app.py              ← Streamlit 3-column UI
├── ingestion.py        ← PDF → chunks + NER + triplets
├── graph_engine.py     ← NetworkX graph builder + BFS
├── vector_engine.py    ← FAISS index builder
├── lexical_engine.py   ← Inverted index builder
├── retriever.py        ← Hybrid retrieval + Ollama/OpenAI LLM
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

Optional OpenAI mode:
```bash
set OPENAI_API_KEY=your_key_here
set OPENAI_MODEL=gpt-4o-mini
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
| Text Splitting | LangChain Text Splitter (`langchain-text-splitters`) |
| Knowledge Graph | NetworkX `DiGraph` (in-memory, no external DB needed) |
| Vector Store | FAISS `IndexFlatL2` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` via `sentence-transformers` |
| LLM | Ollama `llama3.2` |
| UI | Streamlit |
| Graph Viz | Pyvis |

> **Note:** The retrieval pipeline (graph traversal, vector search, keyword search) is built from scratch without LangChain chains — this gives full transparency and control over the retrieval logic.

---

*B.Tech Independent Minor Project — IIIT Pune*
