"""
retriever.py — GraphRAG System
=================================
Hybrid retrieval: FAISS vector search + 2-hop graph BFS.
Includes Ollama LLM call and Compare Mode (Standard RAG vs GraphRAG).

Usage:
  from retriever import GraphRAGRetriever
  r = GraphRAGRetriever()
  answer, trace = r.query("How does OSPF handle link failures?")
"""

import pickle
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import spacy
import networkx as nx
import ollama

from graph_engine import bfs_traverse, ENTITY_COLORS
from lexical_engine import load_index as load_lexical_index, search_index
from vector_engine import load_embedding_model, load_index, query_index

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retriever")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / "data"
CHUNKS_PATH    = DATA_DIR / "chunks.pkl"
GRAPH_PATH     = DATA_DIR / "graph.pkl"
FAISS_PATH     = DATA_DIR / "faiss_index.bin"
CHUNK_MAP_PATH = DATA_DIR / "chunk_map.pkl"
INVERTED_INDEX_PATH = DATA_DIR / "inverted_index.pkl"

# ─── Config ───────────────────────────────────────────────────────────────────
OLLAMA_MODEL  = "llama3.2"          # change to "phi3" if needed
TOP_K_VECTOR  = 5                    # FAISS top-k
TOP_K_KEYWORD = 5                    # inverted-index top-k
MAX_HOPS      = 2                    # graph BFS depth
MAX_CONTEXT_CHUNKS = 8               # final cap on merged context


# ─── Prompt Templates ─────────────────────────────────────────────────────────

GRAPHRAG_PROMPT_TEMPLATE = """You are an expert assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't have enough context to answer this."

Context (retrieved from a knowledge graph + vector store):
---
{context}
---

Question: {question}

Answer (be concise and technical):"""


STANDARD_RAG_PROMPT_TEMPLATE = """You are an expert assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't have enough context to answer this."

Context:
---
{context}
---

Question: {question}

Answer:"""


class GraphRAGRetriever:
    """
    Main retriever class. Loads all artifacts once and exposes a query() method.
    """

    def __init__(
        self,
        ollama_model: str = OLLAMA_MODEL,
        llm_provider: str = "ollama",
        top_k_vector: int = TOP_K_VECTOR,
        top_k_keyword: int = TOP_K_KEYWORD,
        max_hops: int = MAX_HOPS,
        verbose: bool = True,
    ):
        self.ollama_model = ollama_model
        self.llm_provider = llm_provider
        self.top_k_vector = top_k_vector
        self.top_k_keyword = top_k_keyword
        self.max_hops     = max_hops
        self.verbose      = verbose

        log.info("Initializing GraphRAG Retriever ...")
        self._load_artifacts()
        log.info("Retriever ready.")

    def _load_artifacts(self) -> None:
        # Load chunks into a lookup dict: chunk_id → text
        with open(CHUNKS_PATH, "rb") as f:
            raw_chunks = pickle.load(f)
        self.chunk_lookup: Dict[str, str] = {
            c["chunk_id"]: c["text"] for c in raw_chunks
        }
        log.info(f"  Loaded {len(self.chunk_lookup)} chunks")

        # Load graph
        with open(GRAPH_PATH, "rb") as f:
            self.G: nx.DiGraph = pickle.load(f)
        log.info(f"  Loaded graph: {self.G.number_of_nodes()} nodes, "
                 f"{self.G.number_of_edges()} edges")

        # Load FAISS index + embedding model
        self.faiss_index, self.int_to_cid = load_index()
        self.embed_model = load_embedding_model()

        if INVERTED_INDEX_PATH.exists():
            self.lexical_index = load_lexical_index()
            log.info(
                f"  Loaded inverted index: "
                f"{len(self.lexical_index.get('postings', {}))} terms"
            )
        else:
            self.lexical_index = None
            log.warning("Inverted index not found; keyword retrieval disabled.")

        # Load lightweight spaCy for query NER (use sm model for speed here)
        try:
            self.nlp_query = spacy.load("en_core_web_trf")
        except OSError:
            log.warning("en_core_web_trf not available; falling back to en_core_web_sm for query NER")
            try:
                self.nlp_query = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp_query = None
                log.warning("No spaCy model available; graph retrieval disabled.")

    # ─── Step A: Vector Retrieval ──────────────────────────────────────────────

    def _vector_retrieve(self, query: str) -> List[str]:
        """Return top-k chunk_ids from FAISS."""
        log.info(f"[VECTOR] Searching FAISS top-{self.top_k_vector} ...")
        results = query_index(
            query,
            self.embed_model,
            self.faiss_index,
            self.int_to_cid,
            top_k=self.top_k_vector,
            verbose=self.verbose,
        )
        cids = [r["chunk_id"] for r in results]
        log.info(f"[VECTOR] Retrieved chunk_ids: {cids}")
        return cids

    def _keyword_retrieve(self, query: str) -> List[str]:
        """Return top-k chunk_ids from the inverted index."""
        if self.lexical_index is None:
            return []
        log.info(f"[KEYWORD] Searching inverted index top-{self.top_k_keyword} ...")
        results = search_index(query, self.lexical_index, top_k=self.top_k_keyword)
        cids = [r["chunk_id"] for r in results]
        log.info(f"[KEYWORD] Retrieved chunk_ids: {cids}")
        return cids

    # ─── Step B: Entity Extraction + Graph BFS ────────────────────────────────

    def _extract_query_entities(self, query: str) -> List[str]:
        """Run spaCy NER on the query and return entity texts."""
        regex_candidates = re.findall(r"\b[A-Z][A-Za-z0-9_\-]{1,}\b", query)
        if self.nlp_query is None:
            return list(dict.fromkeys(regex_candidates))
        doc = self.nlp_query(query)
        entities = [ent.text for ent in doc.ents]
        try:
            noun_chunks = [
                chunk.text for chunk in doc.noun_chunks
                if len(chunk.text.split()) <= 3
            ]
        except ValueError:
            noun_chunks = []
        combined = list(dict.fromkeys(entities + noun_chunks + regex_candidates))
        log.info(f"[GRAPH]  Query entities/nouns: {combined}")
        return combined

    def _find_graph_nodes(self, entity_texts: List[str]) -> List[str]:
        """
        Match query entities to graph nodes using case-insensitive fuzzy matching.
        Returns list of matched node keys.
        """
        matched = []
        graph_nodes_lower = {n.lower(): n for n in self.G.nodes()}

        for entity in entity_texts:
            entity_lower = entity.lower().strip()
            # Exact match
            if entity_lower in graph_nodes_lower:
                matched.append(graph_nodes_lower[entity_lower])
                log.info(f"[GRAPH]  Exact match: '{entity}' → '{entity_lower}'")
                continue
            # Partial / substring match
            for node_key in graph_nodes_lower:
                if entity_lower in node_key or node_key in entity_lower:
                    matched.append(node_key)
                    log.info(f"[GRAPH]  Partial match: '{entity}' → '{node_key}'")
                    break

        matched = list(set(matched))
        log.info(f"[GRAPH]  Matched graph nodes: {matched}")
        return matched

    def _graph_retrieve(self, query: str) -> Tuple[Set[str], List[str], List[str]]:
        """
        Full graph retrieval:
          1. Extract entities from query
          2. Match to graph nodes
          3. BFS traverse up to max_hops
          4. Return (chunk_id_set, identified_entities, traversed_nodes)
        """
        log.info("[GRAPH]  Starting graph retrieval ...")
        entities = self._extract_query_entities(query)
        matched_nodes = self._find_graph_nodes(entities)

        if not matched_nodes:
            log.info("[GRAPH]  No graph nodes matched. Skipping graph traversal.")
            return set(), entities, []

        all_cids: Set[str] = set()
        all_traversed: List[str] = []

        for node in matched_nodes:
            log.info(f"[GRAPH]  ─── BFS from: '{node}' ───")
            traversal = bfs_traverse(
                self.G, node, max_hops=self.max_hops, verbose=self.verbose
            )
            for tnode, cids in traversal.items():
                all_cids.update(cids)
                if tnode not in all_traversed:
                    all_traversed.append(tnode)

        log.info(f"[GRAPH]  Total unique chunk_ids from graph: {len(all_cids)}")
        return all_cids, entities, all_traversed

    # ─── Step C: Merge Context ────────────────────────────────────────────────

    def _merge_context(
        self,
        vector_cids: List[str],
        graph_cids: Set[str],
        max_chunks: int = MAX_CONTEXT_CHUNKS,
    ) -> Tuple[str, List[str]]:
        """
        Merge vector + graph chunk_ids (vector results first = priority).
        Deduplicates and caps at max_chunks.
        Returns (context_string, ordered_chunk_ids).
        """
        merged_cids = list(dict.fromkeys(vector_cids + list(graph_cids)))
        merged_cids = [cid for cid in merged_cids if cid in self.chunk_lookup]
        merged_cids = merged_cids[:max_chunks]

        context_parts = []
        for i, cid in enumerate(merged_cids):
            text = self.chunk_lookup[cid]
            context_parts.append(f"[Source {i+1} | {cid}]\n{text}")

        context = "\n\n".join(context_parts)
        return context, merged_cids

    # ─── Step D: LLM Call ─────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider. Returns response text."""
        log.info(f"[LLM]    Calling provider: {self.llm_provider} ...")
        t0 = time.time()
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI

                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                response = OpenAI().chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                answer = response.choices[0].message.content.strip()
            else:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response["message"]["content"].strip()
        except Exception as e:
            log.error(f"[LLM]    LLM call failed: {e}")
            answer = f"[Error calling LLM: {e}]"
        elapsed = time.time() - t0
        log.info(f"[LLM]    Response received in {elapsed:.1f}s")
        return answer

    # ─── Public API ───────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        compare_mode: bool = False,
    ) -> Dict:
        """
        Main query entry point.

        Returns dict:
          {
            "graphrag_answer": str,
            "standard_rag_answer": str (only if compare_mode=True),
            "trace": {
              "identified_entities": list,
              "matched_graph_nodes": list,
              "traversed_nodes": list,
              "vector_chunk_ids": list,
              "graph_chunk_ids": list,
              "final_chunk_ids": list,
            }
          }
        """
        log.info("=" * 60)
        log.info(f"QUERY: {question}")
        log.info("=" * 60)

        # ── Step A: Vector ───────────────────────────────────────────────────
        vector_cids = self._vector_retrieve(question)
        keyword_cids = self._keyword_retrieve(question)

        # ── Step B: Graph ────────────────────────────────────────────────────
        graph_cids, identified_entities, traversed_nodes = self._graph_retrieve(question)

        # ── Step C: Merge ────────────────────────────────────────────────────
        context, final_cids = self._merge_context(
            list(dict.fromkeys(vector_cids + keyword_cids)),
            graph_cids,
        )
        log.info(f"[MERGE]  Final context: {len(final_cids)} chunks → {final_cids}")

        # ── Step D: GraphRAG LLM ─────────────────────────────────────────────
        graphrag_prompt = GRAPHRAG_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        graphrag_answer = self._call_llm(graphrag_prompt)

        result = {
            "graphrag_answer": graphrag_answer,
            "trace": {
                "identified_entities":  identified_entities,
                "matched_graph_nodes":  self._find_graph_nodes(identified_entities),
                "traversed_nodes":      traversed_nodes,
                "vector_chunk_ids":     vector_cids,
                "keyword_chunk_ids":    keyword_cids,
                "graph_chunk_ids":      list(graph_cids),
                "final_chunk_ids":      final_cids,
                "context":              context,
            },
        }

        # ── Compare Mode: Standard RAG ───────────────────────────────────────
        if compare_mode:
            log.info("[COMPARE] Running Standard RAG (vector only) ...")
            vector_only_context, vector_only_cids = self._merge_context(
                vector_cids, set(), max_chunks=5
            )
            standard_prompt = STANDARD_RAG_PROMPT_TEMPLATE.format(
                context=vector_only_context, question=question
            )
            standard_answer = self._call_llm(standard_prompt)
            result["standard_rag_answer"] = standard_answer
            result["trace"]["standard_rag_chunks"] = vector_only_cids
            log.info("[COMPARE] Done.")

        log.info("=" * 60)
        log.info("QUERY COMPLETE")
        log.info("=" * 60)
        return result


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GraphRAG Retriever CLI")
    parser.add_argument("--query",   required=True,  help="Question to ask")
    parser.add_argument("--compare", action="store_true", help="Enable compare mode")
    parser.add_argument("--model",   default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    retriever = GraphRAGRetriever(ollama_model=args.model, llm_provider=args.provider)
    result = retriever.query(args.query, compare_mode=args.compare)

    print("\n" + "=" * 60)
    print("GRAPHRAG ANSWER:")
    print("=" * 60)
    print(result["graphrag_answer"])

    if args.compare:
        print("\n" + "=" * 60)
        print("STANDARD RAG ANSWER:")
        print("=" * 60)
        print(result.get("standard_rag_answer", "N/A"))

    print("\n" + "=" * 60)
    print("RETRIEVAL TRACE:")
    print("=" * 60)
    trace = result["trace"]
    print(f"  Identified entities : {trace['identified_entities']}")
    print(f"  Matched graph nodes : {trace['matched_graph_nodes']}")
    print(f"  Traversed nodes     : {trace['traversed_nodes']}")
    print(f"  Vector chunk_ids    : {trace['vector_chunk_ids']}")
    print(f"  Graph chunk_ids     : {trace['graph_chunk_ids'][:8]}...")
    print(f"  Final chunk_ids     : {trace['final_chunk_ids']}")
