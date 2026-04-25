# GraphRAG System — Context & Progress File
> **Purpose**: Keep this file open. If the AI agent runs out of tokens,
> paste this file into a new session to restore full context instantly.

---

## Project Identity
- **Name**: Graph-Enhanced RAG (GraphRAG) System
- **Goal**: B.Tech IV Sem Independent Minor Project
- **Stack**: Python 3.10+, LangChain, spaCy, NetworkX, FAISS, Streamlit, Ollama
- **Ollama Model**: `llama3.2` (primary), `phi3` (fallback)
- **Project Root**: `h:\IIITP\Sem4\BTP\graphrag\`

---

## Key Decisions Made
| Decision | Choice | Reason |
|---|---|---|
| Triplet extraction | Rule-based spaCy (dep parse) | Explainable in viva, fast, offline |
| LLM | llama3.2:3b via Ollama | Low latency for live demo |
| Compare Mode | YES (hard yes) | S-grade feature — proves GraphRAG > RAG |
| Embeddings | all-MiniLM-L6-v2 | Fast, 384-dim, offline |
| FAISS index | IndexFlatL2 | Simple, no training needed |
| NER model | en_core_web_trf | Transformer-based, best accuracy |

---

## Architecture Pipeline
```
PDF
 └─► [ingestion.py]
       ├─ Chunking (RecursiveCharacterTextSplitter, 512/64)
       ├─ NER (spaCy en_core_web_trf + custom EntityRuler)
       └─ Triplet Extraction (dep parse: SUBJ + ROOT + OBJ)
            │
            ├─► [graph_engine.py]  →  graph.pkl  (NetworkX DiGraph)
            │     ├─ Node: {label, entity_type, chunk_ids, color, centrality}
            │     └─ Edge: {predicate, weight}
            │
            └─► [vector_engine.py] →  faiss_index.bin + chunk_map.pkl
                  └─ SentenceTransformers (all-MiniLM-L6-v2)

[retriever.py] — Hybrid query engine
  ├─ Step A: FAISS top-5 vector chunks
  ├─ Step B: spaCy NER on query → BFS 2-hop graph traversal (logged)
  ├─ Step C: Merge + deduplicate chunk_ids
  └─ Step D: Ollama LLM call with merged context

[app.py] — Streamlit UI
  ├─ Sidebar:  Upload PDF | Ingest | Settings
  ├─ Col 1:    Pyvis interactive graph
  ├─ Col 2:    Chat interface (session history)
  ├─ Col 3:    Source context + BFS retrieval trace
  └─ Toggle:   Compare Mode (Standard RAG vs GraphRAG)
```

---

## File Status
| File | Status | Notes |
|---|---|---|
| `requirements.txt` | ✅ DONE | All deps pinned |
| `ingestion.py` | ✅ DONE | PDF→chunks (512/64)→spaCy NER→dep-parse triplets |
| `graph_engine.py` | ✅ DONE | DiGraph + centrality + BFS w/ step logging |
| `vector_engine.py` | ✅ DONE | FAISS IndexFlatL2 + all-MiniLM-L6-v2 |
| `retriever.py` | ✅ DONE | Hybrid: FAISS top-5 + 2-hop BFS + Ollama + compare |
| `app.py` | ✅ DONE | 3-col Streamlit UI + glass design + compare tabs |
| `context.md` | ✅ DONE | This file |

---

## Data Files (auto-generated on ingest)
```
graphrag/data/
├── chunks.pkl       # list of {chunk_id, text, page, metadata}
├── triplets.pkl     # list of {subject, predicate, object, chunk_id}
├── graph.pkl        # NetworkX DiGraph object
├── faiss_index.bin  # FAISS binary index
└── chunk_map.pkl    # {faiss_int_id → chunk_id} mapping
```

---

## How to Run (Quick Reference)
```bash
# 1. Setup
cd h:\IIITP\Sem4\BTP\graphrag
pip install -r requirements.txt
python -m spacy download en_core_web_trf

# 2. Make sure Ollama is running
ollama serve
ollama pull llama3.2

# 3. Launch app (ingestion happens inside UI)
streamlit run app.py

# OR: Run ingestion manually from CLI
python ingestion.py --pdf "path/to/your.pdf"
python graph_engine.py
python vector_engine.py
```

---

## Viva Talking Points (Prepared)
1. **Why spaCy dep parse for triplets?**
   "We look for syntactic (SUBJ, ROOT, OBJ) patterns in the dependency tree.
   A ROOT verb connects a SUBJ noun to an OBJ noun — that's our triplet."

2. **Why FAISS + Graph? Why not just FAISS?**
   "Vector search captures semantic similarity but misses structural relationships.
   The graph lets us traverse 'what is connected to this entity' — giving us
   context that doesn't directly mention the query term."

3. **What is 2-hop BFS?**
   "From a matched entity node, we visit all direct neighbors (hop 1),
   then all of their neighbors (hop 2). Each neighbor contributes its
   chunk_ids, which we retrieve from the text corpus."

4. **How do you prevent the context from being too large?**
   "We deduplicate chunk_ids from both retrieval paths and cap at top-8 chunks.
   The merged context is then passed to the LLM."

---

## Future Enhancements (If Time Allows)
- [ ] Relation extraction with LLM (Option B) as an optional `--mode llm` flag
- [ ] Graph persistence to Neo4j for production scale
- [ ] Named Entity Linking (NEL) to disambiguate nodes (e.g., "Python" language vs snake)
- [ ] Evaluation harness: RAGAS metrics (faithfulness, answer relevancy)
- [ ] Multi-document ingestion (merge graphs from multiple PDFs)
- [ ] Export graph as `.graphml` for external visualization tools

---

## If Resuming in a New Agent Session
1. Paste this entire file as the first message.
2. Say: "Resume the GraphRAG project. All files are written. The next step is: [X]"
3. Reference `h:\IIITP\Sem4\BTP\graphrag\` as the project root.
