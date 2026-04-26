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

```text
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

| Layer           | Technology                               |
| --------------- | ---------------------------------------- |
| NLP / NER       | spaCy `en_core_web_trf`                  |
| Text split      | LangChain Text Splitters                 |
| Knowledge graph | NetworkX `DiGraph` + BFS traversal       |
| Vector store    | FAISS `IndexFlatL2`                      |
| Embeddings      | HuggingFace `all-MiniLM-L6-v2`           |
| LLM             | Ollama `llama3.2` (local, offline)       |
| Terminal UI     | Python + colorama                        |
| Graph viz       | Neo4j Browser (screenshot only, for PPT) |

---

## Project Structure

```text
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
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

Notes:

- `en_core_web_sm` is enough for most BTP demos.
- `en_core_web_md` usually improves NER quality.
- `en_core_web_trf` is optional (slower and larger).

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

### Using another PDF as the knowledge base

To switch the project to a different document, rerun the full build pipeline with your new file path:

```bash
python ingestion.py --pdf sample_data/your_new_file.pdf
python graph_engine.py
python vector_engine.py
python neo4j_export.py
python demo.py
```

Notes:

- `ingestion.py` extracts chunks/entities/triplets from the PDF you pass.
- `graph_engine.py` rebuilds `data/graph.pkl` from those extracted triplets.
- `vector_engine.py` rebuilds the FAISS vectors for the same content.
- `neo4j_export.py` pushes the latest graph to Neo4j for PPT screenshots.
- If you skip `graph_engine.py`, export can show `0 nodes, 0 edges`.

---

## Neo4j Graph Visualisation (for PPT only)

To generate a publication-quality knowledge graph screenshot (Aura compatible):

1. Install the driver: `pip install neo4j`
2. Set environment variables:
   - `set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687`
   - If that times out, try `set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:443`
     - `set NEO4J_USERNAME=9b183348`
     - `set NEO4J_DATABASE=9b183348`
   - `set NEO4J_PASSWORD=<your-password>`
     - Or put the same values in a local `.env` file
3. Run: `python neo4j_export.py`
4. Open Neo4j Browser from Aura Console and run `MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150`.
5. Take a screenshot — use it on your PPT knowledge graph slide

> Neo4j does **not** need to be running during the presentation.

### Troubleshooting: Neo4j export says `Exported 0 nodes, 0 edges`

This usually means `data/graph.pkl` is empty (or was built from no extracted triplets), not a Neo4j auth problem.

Check graph contents directly:

```bash
python -c "import pickle; G = pickle.load(open('data/graph.pkl','rb')); print(type(G)); print('Nodes:', G.number_of_nodes()); print('Edges:', G.number_of_edges())"
```

Expected: non-zero node/edge counts before running `neo4j_export.py`.

If counts are zero, rebuild in order:

```bash
python ingestion.py --pdf sample_data/computer_networks.pdf
python graph_engine.py
python vector_engine.py
```

Then rerun:

```bash
python neo4j_export.py
```

### Troubleshooting: HuggingFace warning about unauthenticated requests

If you see a warning like "You are sending unauthenticated requests to the HF Hub", ensure `HF_TOKEN` exists in `.env`.

This project loads `.env` at startup in `ingestion.py` and `vector_engine.py` before HuggingFace imports, so `HF_TOKEN` is available early.

Example `.env` keys:

```dotenv
HF_TOKEN=hf_xxx
LLM_MODEL=llama3.2
RETRIEVAL_TOP_K=5
```

---

## Student Details

| Field       | Value                          |
| ----------- | ------------------------------ |
| Name        | Harkeerat Singh                |
| Roll No.    | 112415075                      |
| Project     | B.Tech Independent Minor (BTP) |
| Topic       | AI + ML + DSA (GraphRAG)       |
| Institution | IIIT Pune                      |

---
