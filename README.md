# GraphRAG System — B.Tech Independent Minor Project

> **Graph-Enhanced Retrieval-Augmented Generation**
> A terminal-based Q&A system that builds a knowledge graph
> from any PDF and answers questions using hybrid graph + vector retrieval.

---

## What it does

Standard LLMs hallucinate because they have no document memory.
Standard RAG improves this with vector search, but misses
relationships between concepts. This system builds a **knowledge
graph** from the document — so retrieval follows concept connections
the way a human would reason, not just keyword similarity.

---

## Architecture

```
PDF Document
     │
     ▼
[ingestion.py] ── spaCy NER + LangChain chunking + dep-parse triplets
     │
     ├─────────────────────────────────┐
     ▼                                 ▼
[graph_engine.py]              [vector_engine.py]
 NetworkX DiGraph               FAISS IndexFlatL2
 BFS traversal                  HuggingFace MiniLM
     │                                 │
     └──────────────┬──────────────────┘
                    ▼
             [retriever.py]
      Hybrid: Vector + Inverted Index + 2-hop Graph BFS
                    │
                    ▼
               [demo.py]
         Terminal Q&A interface
```

---

## Tech Stack

| Layer        | Technology                                  |
|--------------|---------------------------------------------|
| NLP / NER    | spaCy `en_core_web_trf`                     |
| Text split   | LangChain Text Splitters                    |
| Knowledge graph | NetworkX `DiGraph` + BFS traversal       |
| Vector store | FAISS `IndexFlatL2`                         |
| Embeddings   | HuggingFace `all-MiniLM-L6-v2`             |
| LLM          | Ollama `llama3.2` (local, offline)          |
| Terminal UI  | Python + colorama                           |
| Graph viz    | Neo4j Browser (screenshot only, for PPT)    |

---

## Project Structure

```
graphrag/
├── demo.py              ← Entry point: terminal Q&A loop
├── ingestion.py         ← PDF → chunks + NER + triplets
├── graph_engine.py      ← NetworkX graph builder + BFS
├── vector_engine.py     ← FAISS index builder
├── lexical_engine.py    ← Inverted index (keyword retrieval)
├── retriever.py         ← Hybrid retrieval + LLM call
├── neo4j_export.py      ← One-time Neo4j export for PPT screenshot
├── requirements.txt
├── sample_data/
│   └── computer_networks.pdf
└── data/                ← Auto-generated at runtime (gitignored)
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 3. Pull the LLM
```bash
ollama pull llama3.2
```

### 4. Run ingestion (builds index from PDF)
```bash
python ingestion.py --pdf sample_data/computer_networks.pdf
python graph_engine.py
python vector_engine.py
```

### 5. Start the terminal Q&A
```bash
python demo.py
```
Press **Enter on a blank input** to exit.

---

## Neo4j Graph Visualisation (for PPT only)

To generate a publication-quality knowledge graph screenshot:

1. Download [Neo4j Desktop](https://neo4j.com/download/) and start a local database
2. Install the driver: `pip install neo4j`
3. Run: `python neo4j_export.py`
4. Open `http://localhost:7474` and run:
   ```cypher
   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150
   ```
5. Take a screenshot — use it on your PPT knowledge graph slide

> Neo4j does **not** need to be running during the presentation.

---

## Student Details

| Field       | Value                              |
|-------------|-------------------------------------|
| Name        | Harkeerat Singh                    |
| Roll No.    | 112415075                          |
| Project     | B.Tech Independent Minor (BTP)     |
| Topic       | AI + ML + DSA (GraphRAG)           |
| Institution | IIIT Pune                          |

---
