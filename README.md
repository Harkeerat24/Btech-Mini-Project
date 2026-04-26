# GraphRAG System — B.Tech Independent Minor Project

> **Graph-Enhanced Retrieval-Augmented Generation**
> A terminal-based AI system that reads any PDF, builds a knowledge graph of its concepts, and answers questions using a hybrid of graph traversal + semantic vector search — more accurately than a plain LLM or standard RAG.

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

## What Problem Does This Solve?

A standard Large Language Model (LLM) like Llama or GPT has no memory of your specific document. If you ask it a question, it either makes up an answer (hallucination) or says it doesn't know.

**Standard RAG** (Retrieval-Augmented Generation) improves this by searching your document using vector similarity — find the chunks of text most similar to the question, and give them to the LLM as context. This works for simple factual questions but fails for questions that require understanding _relationships_ between concepts. For example: "How does OSPF use Dijkstra's algorithm?" requires knowing that OSPF → Dijkstra is a relationship, not just two similar-sounding words.

**GraphRAG** (this project) solves this by also building a **knowledge graph** — a network of concepts and their relationships extracted directly from the document. When you ask a question, the system:

1. Finds which concepts (entities) appear in the question
2. Traverses the knowledge graph with BFS (Breadth-First Search) to find related concepts
3. Retrieves text chunks associated with those concepts
4. Combines that with vector search results
5. Gives the merged context to the LLM for a grounded, accurate answer

The result: better answers on relationship questions, and correct "I don't know" responses for questions outside the document (no hallucination).

---

## Architecture — How Data Flows

```
Your PDF Document
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ingestion.py                                                        │
│                                                                      │
│  1. Extracts raw text from the PDF page by page (pypdf)             │
│  2. Splits text into overlapping chunks of ~512 tokens              │
│     (LangChain RecursiveCharacterTextSplitter)                      │
│  3. Runs Named Entity Recognition (NER) on each chunk               │
│     using spaCy to extract entities: OSPF, TCP, Dijkstra, etc.     │
│  4. Extracts Subject-Verb-Object triplets using dependency parsing  │
│     e.g. (OSPF) --[uses]--> (Dijkstra's algorithm)                 │
│  5. Saves: chunks.pkl, triplets.pkl                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┴────────────────┐
               │                                │
               ▼                                ▼
┌──────────────────────────┐    ┌───────────────────────────────────┐
│  graph_engine.py          │    │  vector_engine.py                  │
│                           │    │                                   │
│  Builds a NetworkX        │    │  Embeds every chunk using         │
│  DiGraph (directed graph) │    │  HuggingFace all-MiniLM-L6-v2     │
│                           │    │  (384-dimensional vectors)        │
│  Nodes = entities         │    │                                   │
│  Edges = relationships    │    │  Stores in FAISS IndexFlatL2      │
│  (from triplets + NER     │    │  for fast approximate nearest     │
│   co-occurrence)          │    │  neighbour search                 │
│                           │    │                                   │
│  Computes:                │    │  Also builds an inverted index    │
│  - Degree centrality      │    │  (lexical_engine.py) for exact    │
│  - Betweenness centrality │    │  keyword matching                 │
│                           │    │                                   │
│  Saves: graph.pkl         │    │  Saves: faiss_index.bin,          │
│                           │    │  chunk_map.pkl, inverted_index.pkl│
└──────────────┬────────────┘    └──────────────────┬──────────────┘
               │                                    │
               └────────────────┬───────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  retriever.py  — Hybrid Retrieval Engine                             │
│                                                                      │
│  When you ask a question, three things happen in parallel:          │
│                                                                      │
│  1. VECTOR SEARCH (ML)                                              │
│     Embed the question → search FAISS → return top-5 similar chunks │
│                                                                      │
│  2. KEYWORD SEARCH (DSA — Inverted Index)                           │
│     Tokenise question → look up tokens in inverted index            │
│     → return chunks containing exact keywords                       │
│                                                                      │
│  3. GRAPH TRAVERSAL (DSA — BFS on NetworkX DiGraph)                 │
│     Extract entities from question using spaCy NER                  │
│     → find matching nodes in knowledge graph                        │
│     → run 2-hop BFS: visit neighbours and their neighbours          │
│     → collect all source chunks attached to traversed nodes         │
│                                                                      │
│  All three result sets are merged and deduplicated → top-8 chunks  │
│  This merged context is passed to the LLM as grounded information  │
│                                                                      │
│  Then: calls Ollama API (llama3.2 running locally) with the prompt: │
│  "Answer using ONLY the context below. If unsure, say so."         │
│  → returns the LLM's answer + the retrieval trace for display      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  demo.py  — Terminal Interface                                        │
│                                                                      │
│  Loads all saved artifacts from data/ directory                     │
│  Prints a summary: nodes, edges, chunks                             │
│  Loops: accepts your question → calls retriever → prints trace      │
│  + answer → waits for next question                                 │
│  Exits cleanly when you press Enter on a blank line                │
└─────────────────────────────────────────────────────────────────────┘
                               │
                    (optional, run once for PPT)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  neo4j_export.py  — Graph Visualisation Export                       │
│                                                                      │
│  Reads graph.pkl → connects to Neo4j Aura (cloud) via bolt driver  │
│  Clears old data → batch-inserts all nodes and edges                │
│  You then open Neo4j Browser → run a Cypher query → screenshot     │
│  That screenshot is used on your PPT knowledge graph slide          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
graphrag/
│
├── demo.py              ← MAIN ENTRY POINT — run this to ask questions
├── ingestion.py         ← Step 1: PDF → text chunks + NER entities + triplets
├── graph_engine.py      ← Step 2: Build NetworkX knowledge graph
├── vector_engine.py     ← Step 3: Build FAISS vector index + inverted index
├── lexical_engine.py    ← Helper: inverted index (keyword search) — used by retriever
├── retriever.py         ← Core engine: hybrid retrieval + LLM call
├── neo4j_export.py      ← Optional: export graph to Neo4j Aura for PPT screenshot
│
├── requirements.txt     ← All Python dependencies
├── .env                 ← Your secrets (Neo4j credentials, HF token) — never commit this
├── .env.example         ← Template showing which keys are needed
├── .gitignore           ← Excludes .venv/, data/, .env from git
│
├── sample_data/
│   ├── computer_networks.pdf   ← Main demo document (recommended)
│   └── sample_story.pdf        ← Short story for quick testing
│
└── data/                ← Auto-generated at runtime — do NOT edit manually
    ├── chunks.pkl          ← List of text chunks from the PDF
    ├── triplets.pkl        ← List of (subject, verb, object) triplets
    ├── graph.pkl           ← NetworkX DiGraph object
    ├── faiss_index.bin     ← FAISS vector index
    ├── chunk_map.pkl       ← Maps FAISS integer IDs back to chunk IDs
    └── inverted_index.pkl  ← Token → list of chunk IDs mapping
```

> **The `data/` folder is gitignored.** It is created automatically when you run the pipeline. You never need to touch it manually.

---

## Tech Stack

| Layer           | Technology                                        | Why used                               |
| --------------- | ------------------------------------------------- | -------------------------------------- |
| PDF Parsing     | `pypdf`                                           | Extract raw text from any PDF          |
| Text Splitting  | `LangChain RecursiveCharacterTextSplitter`        | Smart overlapping chunking             |
| NLP / NER       | `spaCy en_core_web_sm` (or `_md`, `_trf`)         | Named Entity Recognition               |
| Knowledge Graph | `NetworkX DiGraph`                                | Store + traverse concept relationships |
| Graph Traversal | BFS (custom, built on NetworkX)                   | Multi-hop concept retrieval            |
| Vector Store    | `FAISS IndexFlatL2`                               | Fast semantic similarity search        |
| Embeddings      | `HuggingFace all-MiniLM-L6-v2`                    | Convert text to 384-dim vectors        |
| Keyword Search  | Custom inverted index (dict of token → chunk IDs) | Exact keyword matching                 |
| LLM             | `Ollama llama3.2` (runs locally, offline)         | Generate grounded answers              |
| Graph Export    | `Neo4j Aura` + Python neo4j driver                | Visual graph screenshot for PPT        |
| Terminal UI     | Python + `colorama`                               | Coloured terminal output               |

---

## Prerequisites

Before you start, install these on your machine:

1. **Python 3.10 or higher** — check with `python --version`
2. **Ollama** — download from [https://ollama.com](https://ollama.com) and install it
3. **Git** — to clone the repo

You do NOT need Neo4j installed unless you want the graph screenshot for your PPT. See the Neo4j section below.

---

## Setup — Do This Once

### Step 1 — Clone the repository

```bash
git clone https://github.com/Harkeerat24/Btech-Mini-Project.git
cd Btech-Mini-Project
```

### Step 2 — Create a virtual environment

A virtual environment keeps this project's dependencies isolated from your global Python installation. This prevents version conflicts.

```bash
python -m venv .venv
```

Now activate it. **You must activate it every time you open a new terminal.**

On Windows (PowerShell or Git Bash):

```bash
source .venv/Scripts/activate
```

On macOS / Linux:

```bash
source .venv/bin/activate
```

You will see `(.venv)` or `graphrag` in your terminal prompt — this means the environment is active.

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs everything: spaCy, FAISS, sentence-transformers, networkx, ollama, colorama, etc.

### Step 4 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

This downloads the small English NLP model used for entity recognition. If you want higher accuracy (slower), use `en_core_web_md` instead.

### Step 5 — Download the LLM

```bash
ollama pull llama3.2
```

This downloads the Llama 3.2 model (~2GB) to your local machine. Ollama runs it locally — no internet needed when answering questions.

### Step 6 — Set up your `.env` file

Copy the example file:

```bash
cp .env.example .env
```

Open `.env` and fill in your values. The minimum required to run `demo.py` is:

```
LLM_MODEL=llama3.2
RETRIEVAL_TOP_K=5
```

The Neo4j and HuggingFace fields are only needed for `neo4j_export.py` and are optional for the Q&A demo.

---

## Running the Project — Commands in Order

### Every time you want to use a new PDF, run Steps A through D:

**Step A — Ingest the PDF (extract chunks + entities + triplets)**

```bash
python ingestion.py --pdf sample_data/computer_networks.pdf
```

What this does: reads the PDF, splits it into chunks, runs NER to find entities, extracts subject-verb-object triplets, saves `chunks.pkl` and `triplets.pkl` to the `data/` folder.

Expected output: shows first 3 chunks and first 8 triplets as a preview.

**Step B — Build the knowledge graph**

```bash
python graph_engine.py
```

What this does: reads `triplets.pkl` and `chunks.pkl`, creates a NetworkX directed graph where nodes are entities and edges are relationships, computes centrality scores, saves `graph.pkl`.

Expected output: logs the number of nodes and edges built.

**Step C — Build the vector index**

```bash
python vector_engine.py
```

What this does: reads `chunks.pkl`, embeds every chunk using the HuggingFace MiniLM model, builds a FAISS index, also builds the inverted keyword index, saves `faiss_index.bin`, `chunk_map.pkl`, `inverted_index.pkl`.

Expected output: shows embedding progress bar, confirms FAISS index size.

**Step D — Start asking questions**

```bash
python demo.py
```

What this does: loads all saved artifacts from `data/`, shows a summary of the index, then enters a loop where you type questions and get answers with a retrieval trace.

To exit: press **Enter on a blank line** (just hit Enter without typing anything).

---

## ⚡ Already ingested a PDF and want to ask more questions?

**You do NOT need to repeat Steps A, B, C.**

The `data/` folder stores all built indexes permanently on disk. As long as you haven't changed the PDF or deleted the `data/` folder, just run:

```bash
python demo.py
```

It loads everything from disk in a few seconds and you're ready to ask questions immediately. The index only needs to be rebuilt when you switch to a different PDF.

---

## Optional — Export Graph to Neo4j for PPT Screenshot

This step is only needed once to get a high-quality knowledge graph image for your presentation slide. Neo4j does **not** need to be running during your presentation.

### Setup Neo4j Aura (cloud, free tier)

1. Go to [https://console.neo4j.io](https://console.neo4j.io) and create a free Aura instance
2. Download the credentials file when prompted — it contains your URI, username, and password
3. Add these to your `.env`:

```
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=your-instance-id
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=your-instance-id
```

### Install the Neo4j driver (only needed for this script)

```bash
pip install neo4j
```

### Run the export

```bash
python neo4j_export.py
```

This pushes all nodes and edges from `graph.pkl` into your Aura instance.

### Get the screenshot

1. Open your Aura Console at [https://console.neo4j.io](https://console.neo4j.io)
2. Click **Open** next to your instance → opens Neo4j Browser
3. In the query box, run:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150
```

4. You will see your knowledge graph visualised as an interactive network
5. Arrange nodes to your liking → take a full-resolution screenshot
6. Use this screenshot on your **Knowledge Graph** PPT slide

---

## Verified Performance — computer_networks.pdf

After running the full pipeline on `computer_networks.pdf`:

| Metric        | Value                                |
| ------------- | ------------------------------------ |
| Chunks        | 25                                   |
| Graph nodes   | 259                                  |
| Graph edges   | 641                                  |
| FAISS vectors | 25 (384-dimensional)                 |
| BFS depth     | 2 hops                               |
| Retrieval     | 5 vector + up to 23 graph → 8 merged |

### Sample questions and answers (all verified working)

| Question                                              | Answer quality                                                                    |
| ----------------------------------------------------- | --------------------------------------------------------------------------------- |
| What is OSPF?                                         | Correct, concise                                                                  |
| Which algorithm does OSPF use?                        | "Dijkstra's Algorithm." — perfect                                                 |
| What is the maximum hop count in RIP?                 | "15, with 16 = unreachable" — correct                                             |
| How are OSPF and Dijkstra's algorithm related?        | Graph traversal finds direct connection, answered accurately                      |
| How does OSPF detect and recover from a link failure? | Multi-hop reasoning, full explanation of Dead Interval + LSA flood + SPF rerun    |
| Why does RIP suffer from count-to-infinity?           | Correctly explains the problem and the split horizon / route poisoning mitigation |
| What is the capital of France?                        | "I don't have enough context" — correct refusal, no hallucination                 |
| Who invented the telephone?                           | "I don't have enough context" — correct refusal, no hallucination                 |

---

## Troubleshooting

**"No spaCy English model found"**

```bash
python -m spacy download en_core_web_sm
```

**"0 nodes · 0 edges" in demo.py header**
You forgot to run `graph_engine.py` after ingestion, or your spaCy model was missing during ingestion (which means no entities were extracted → no graph nodes). Re-run Steps A and B after installing spaCy.

**"Index not found" when starting demo.py**
You need to run the full pipeline (Steps A, B, C) at least once before running demo.py.

**HuggingFace "unauthenticated requests" warning**
This is harmless — it just means rate limiting is lower. To suppress it, set `HF_TOKEN` in your `.env` to a valid token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Ollama connection error**
Make sure Ollama is running. Open a separate terminal and run:

```bash
ollama serve
```

Then try again.

**Neo4j "Database not found" error**
Make sure your `.env` uses `neo4j+s://` (not `bolt+s://`) and that `NEO4J_USERNAME` and `NEO4J_DATABASE` match exactly what's in the credentials file downloaded from Aura Console.

---

## How the Virtual Environment Works

A virtual environment (`.venv/`) is an isolated Python installation that lives inside your project folder. When activated, `pip install` puts packages into `.venv/Lib/` instead of your global Python — so this project's dependencies never conflict with other Python projects on your machine.

The `.venv/` folder is excluded from git (listed in `.gitignore`) so it's never pushed to GitHub. Anyone who clones the repo creates their own fresh venv with `python -m venv .venv` and installs dependencies with `pip install -r requirements.txt`.

You **must** activate the venv every time you open a new terminal window before running any project commands. If you forget, you'll get import errors because the packages won't be found.

```bash
# Windows
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate

# Verify it's active — you should see the venv name in your prompt
# and this should print the path inside .venv:
which python
```

---

_GraphRAG System · B.Tech Independent Minor Project · Harkeerat Singh · Roll No. 112415075 · IIIT Pune_
