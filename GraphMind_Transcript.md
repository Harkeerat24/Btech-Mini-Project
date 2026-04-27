# GraphMind — Presentation Transcript (≤ 10 Minutes)
**Slide-by-slide speaker notes | B.Tech Minor Project | Harkeerat Singh**

---

## Slide 1 — Title Slide
**⏱ 0:00 – 0:30 (30 sec)**

**What to SAY:**
> "Good morning / afternoon. My name is Harkeerat Singh, Roll Number 112415075, and today I'm presenting my B.Tech Independent Minor Project titled **GraphMind** — a Graph-Enhanced Retrieval-Augmented Generation system.
> The project sits at the intersection of AI, Machine Learning, and Data Structures & Algorithms. The core stack includes spaCy, NetworkX, FAISS, HuggingFace Sentence Transformers, and Ollama — all running fully offline on a local machine."

**What to SHOW:**
- Point to the five library logos at the bottom.
- Let the title and subtitle speak visually. Don't rush.

---

## Slide 2 — The Problem with Standard LLMs & RAG
**⏱ 0:30 – 1:30 (60 sec)**

**What to SAY:**
> "So — why build this at all? Let's look at the problem.
>
> A **plain LLM** has no memory of your document. It generates plausible-sounding but completely fabricated answers — this is called hallucination. You cannot trust it on domain-specific content like networking textbooks.
>
> **Standard RAG** is better — it retrieves similar text using vector search. But it only finds chunks that are *semantically close* in embedding space. If I ask 'How does OSPF use Dijkstra's algorithm?', the system needs to understand a **relationship** between two separate concepts — not just similar words. Standard RAG misses this.
>
> **GraphMind** solves this by building a Knowledge Graph and traversing concept relationships using Breadth-First Search. It retrieves context from *connected* concepts — giving the LLM grounded, accurate information."

**What to SHOW:**
- Point to each column: Plain LLM → Standard RAG → GraphRAG.
- Emphasise the key insight quote at the bottom: *"How does OSPF use Dijkstra's algorithm?"*

---

## Slide 3 — Comparison Table
**⏱ 1:30 – 2:00 (30 sec)**

**What to SAY:**
> "This table summarises the comparison concisely. The critical differentiator is the last two rows — **Relationship Q&A** and **DSA Integration**.
> Plain LLM and Standard RAG have zero DSA integration. GraphMind uses Graph BFS and an Inverted Index — actual algorithms from your DSA coursework — directly in the retrieval pipeline. That's the core research contribution."

**What to SHOW:**
- Run your finger down the GraphRAG column: Full ✓✓ across the board.
- Highlight the bottom row — DSA integration.

---

## Slide 4 — System Architecture
**⏱ 2:00 – 3:15 (75 sec)**

**What to SAY:**
> "Here's the full 5-stage pipeline.
>
> **Stage 1 — Ingestion:** We feed in a PDF. `ingestion.py` uses pypdf to extract text, splits it into 512-character chunks with LangChain, runs spaCy Named Entity Recognition, and then extracts Subject–Predicate–Object triplets using dependency parsing.
>
> **Stage 2 — Indexing:** Two things happen in parallel. `graph_engine.py` builds a NetworkX Knowledge Graph from the triplets. `vector_engine.py` embeds every chunk into a 384-dimensional vector and builds a FAISS index for fast similarity search.
>
> **Stage 3 — Retrieval:** When you ask a question, `retriever.py` runs three parallel searches — vector, keyword, and graph BFS — and merges the results into the top 8 most relevant chunks.
>
> **Stage 4 — Answer:** Those 8 chunks are sent as context to LLaMA 3.2 running locally via Ollama. It answers using *only* that context — hallucination by design is prevented."

**What to SHOW:**
- Trace the pipeline left to right with your hand or a pointer.
- Point to the three retrieval boxes at the bottom: Vector, Keyword, Graph BFS.
- Point to the merge arrow and the "top 8 chunks" step.

---

## Slide 5 — Ingestion Pipeline
**⏱ 3:15 – 4:15 (60 sec)**

**What to SAY:**
> "Let me go deeper into `ingestion.py`. It has four stages.
>
> First, pypdf reads every page of the PDF. Second, LangChain's RecursiveCharacterTextSplitter breaks the text into 512-token windows with a 64-token overlap — the overlap ensures no sentence is cut off at a boundary.
>
> Third — Named Entity Recognition. spaCy's `en_core_web_lg` model identifies entities. But spaCy doesn't know networking terms out of the box, so I added a custom `EntityRuler` with patterns like `OSPF → PROTOCOL`, `Dijkstra → ALGORITHM`, `LSA → CONCEPT`. This is an important engineering decision.
>
> Fourth — Triplet extraction. spaCy's dependency parser identifies the subject, root verb, and object in each sentence and produces (S, P, O) triplets — the factual building blocks of the knowledge graph.
>
> Look at the sample triplets on the right: OSPF uses Dijkstra's algorithm, TCP uses three-way handshake, OSPF detects link failure. These are all facts the graph will encode."

**What to SHOW:**
- Point to steps 01–04 on the left.
- Point to the triplet table on the right. Read 2–3 examples aloud.

---

## Slide 6 — Knowledge Graph Construction
**⏱ 4:15 – 4:55 (40 sec)**

**What to SAY:**
> "`graph_engine.py` takes those triplets and builds a directed NetworkX graph. Every unique entity becomes a node — protocols, algorithms, concepts. Every triplet becomes a directed edge with the verb as the label. Additionally, entities that appear in the *same chunk* get co-occurrence edges, which captures implicit relationships the dependency parser may miss.
>
> After building the graph, the code computes **degree centrality** — how connected a node is — and **betweenness centrality** — is this node a bridge between clusters? These metrics identify hub concepts, the most important ideas in the document.
>
> For the computer_networks PDF, the graph produced **259 nodes and 641 edges**."

**What to SHOW:**
- Point to the definition box: Nodes = Entities, Edges = Relationships.
- Mention the output artifact: `data/graph.pkl`.

---

## Slide 7 — Vector Engine & Semantic Retrieval
**⏱ 4:55 – 5:45 (50 sec)**

**What to SAY:**
> "`vector_engine.py` handles the ML layer. It uses HuggingFace's `all-MiniLM-L6-v2` model to convert each text chunk into a 384-dimensional embedding vector. These vectors capture semantic meaning — two chunks about OSPF link failure will cluster together even if they use different words.
>
> FAISS then builds a flat L2 index over all these vectors. At query time, we embed the question and FAISS returns the top 5 most semantically similar chunks in milliseconds.
>
> This module also builds the **Inverted Index** for keyword search — a dictionary mapping every token to the chunks that contain it. At query time, tokens from the question are looked up in O(1), and chunks are scored by term frequency.
>
> The merge strategy gives vector results highest priority, followed by keyword, then graph BFS — deduplicated and capped at 8 chunks."

**What to SHOW:**
- Point to the query pipeline diagram: Query → embed() → FAISS.search() → top-5 chunks.
- Point to the merge pyramid: Vector → Keyword → Graph BFS → Cap at 8.

---

## Slide 8 — Retriever & LLM Integration
**⏱ 5:45 – 6:45 (60 sec)**

**What to SAY:**
> "`retriever.py` is the heart of the system. Let me walk through the 4 steps for a single query: *'How does OSPF handle a link failure?'*
>
> **Step A — Entity Extraction:** spaCy NER + regex finds 'OSPF' in the query.
>
> **Step B — Graph BFS:** We look up 'OSPF' in the knowledge graph and run a 2-hop Breadth-First Search. In 2 hops we visit 80 to 100 nodes — Dijkstra, LSA, Hello packets, Dead Interval — and collect every chunk ID attached to those nodes.
>
> **Step C — Merge:** We combine the chunk IDs from all three retrievers, deduplicate, and take the top 8.
>
> **Step D — LLM Answer:** The 8 chunks are formatted into a prompt: *'Use ONLY the context below. If the context does not contain enough information, say so.'* This instruction is what prevents hallucination. The answer comes back from LLaMA 3.2 running locally via Ollama — no API key, no cloud, fully private."

**What to SHOW:**
- Point to steps A → B → C → D in order.
- Read the prompt template quote at the bottom aloud — this is your key design decision.

---

## Slide 9 — neo4j_export.py & demo.py
**⏱ 6:45 – 7:15 (30 sec)**

**What to SAY:**
> "Two utility components. `neo4j_export.py` is an optional one-time script that pushes the full NetworkX graph into Neo4j Aura — the cloud graph database — so you can visually explore the knowledge graph in a browser. Every node gets its centrality scores as attributes.
>
> `demo.py` is the actual entry point. It loads all the data artifacts, shows a header with live stats — nodes, edges, chunks — and then loops, waiting for your questions. For each answer it prints a colour-coded trace: entities detected, BFS path taken, retrieval counts, and the final LLM answer."

**What to SHOW:**
- Left panel: neo4j_export.py — the Cypher query and browser visualisation.
- Right panel: demo.py — the terminal interface.

---

## Slides 10–13 — System in Action / Live Demo Results
**⏱ 7:15 – 8:30 (75 sec)**

**What to SAY:**
> "These slides show the system running on `computer_networks.pdf` — the actual document you provided as the test corpus. The graph has **259 nodes, 641 edges, and 25 chunks**.
>
> You can see the terminal output for queries like 'How does OSPF handle a link failure?' The system correctly identifies OSPF as the entity, runs BFS, retrieves the chunks about Dead Interval and Dijkstra re-computation, and the LLM answers with precision — citing that OSPF detects failure through the Dead Interval timer expiry and re-runs Dijkstra to recalculate the shortest path tree.
>
> Importantly, when asked a question whose answer is *not* in the document, the system says so — it doesn't fabricate. That refusal behaviour is by design."

**What to SHOW:**
- Show each screenshot briefly — don't read every line.
- Point to: entity detected, BFS hops, chunks used count, and the answer text.
- Emphasise any screenshot where the system correctly refuses out-of-context questions.

---

## Slide 14 — Thank You
**⏱ 8:30 – 9:00 (30 sec)**

**What to SAY:**
> "To summarise — GraphMind demonstrates that combining a Knowledge Graph, BFS traversal, and hybrid retrieval produces substantially better question-answering than plain vector RAG on relational queries. The system is fully offline, requires no API keys, and uses well-established DSA concepts directly in the production pipeline.
>
> The full source code is on GitHub at github.com/Harkeerat24/Btech-Mini-Project.
>
> Thank you. I'm happy to take questions."

**What to SHOW:**
- Point to the five library logos.
- Smile. You're done.

---

## Timing Summary

| Slide | Content | Time | Cumulative |
|-------|---------|------|------------|
| 1 | Title | 0:30 | 0:30 |
| 2 | Problem Statement | 1:00 | 1:30 |
| 3 | Comparison Table | 0:30 | 2:00 |
| 4 | System Architecture | 1:15 | 3:15 |
| 5 | Ingestion Pipeline | 1:00 | 4:15 |
| 6 | Knowledge Graph | 0:40 | 4:55 |
| 7 | Vector Engine | 0:50 | 5:45 |
| 8 | Retriever & LLM | 1:00 | 6:45 |
| 9 | neo4j / demo.py | 0:30 | 7:15 |
| 10–13 | Live Results | 1:15 | 8:30 |
| 14 | Thank You | 0:30 | **9:00** |

---

## Top 5 Questions You'll Likely Be Asked

**Q1: Why BFS and not DFS or Dijkstra for graph traversal?**
> BFS gives you all nodes within N hops uniformly — ideal for context gathering. DFS would go deep into one branch. Dijkstra finds shortest paths, but we don't need a single path — we need all related concepts within 2 hops. BFS is the right choice here.

**Q2: What is the time complexity of your retrieval pipeline?**
> FAISS L2 search is O(N·d) where N = chunks and d = 384. BFS is O(V + E) on the graph. Keyword lookup is O(Q) per query token. Together the full retrieval runs in under a second on this corpus size.

**Q3: Why all-MiniLM-L6-v2 and not a larger model?**
> It's small (22M parameters), fast, runs fully offline, and produces 384-dim vectors that balance quality and speed. For a B.Tech project running on a laptop without a GPU, it's the optimal choice.

**Q4: How does the system prevent hallucination?**
> The LLM prompt explicitly says: *"Use ONLY the context below. If the context does not contain enough information, say so."* The LLM has no freedom to make up facts outside the retrieved chunks. This is a design constraint, not a model property.

**Q5: What would you improve with more time?**
> Three things: (1) Replace BM25-style scoring with proper BM25 for better keyword ranking. (2) Add re-ranking of the top-8 chunks using a cross-encoder before the LLM call. (3) Build a web UI instead of the terminal interface.
