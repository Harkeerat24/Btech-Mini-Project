import os
import pickle
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx
import ollama
import spacy
from dotenv import load_dotenv

from graph_engine import bfs_traverse
from vector_engine import (
    load_embedding_model,
    load_index as load_faiss_index,
    load_lexical_index,
    query_index,
    search_index,
)

load_dotenv()

"""
retriever.py - GraphMind hybrid retrieval engine
================================================
Runs vector, keyword, and graph retrieval concurrently, ranks context with a
weighted hybrid score, and calls the configured generation provider.
"""

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retriever")

DATA_DIR = Path(__file__).parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
GRAPH_PATH = DATA_DIR / "graph.pkl"
INVERTED_INDEX_PATH = DATA_DIR / "inverted_index.pkl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2").strip()
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", os.getenv("RETRIEVAL_TOP_K", "5")))
TOP_K_KEYWORD = int(os.getenv("TOP_K_KEYWORD", "5"))
MAX_HOPS = int(os.getenv("MAX_HOPS", "2"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
SCORE_WEIGHTS = {
    "vector": float(os.getenv("SCORE_WEIGHT_VECTOR", "0.50")),
    "keyword": float(os.getenv("SCORE_WEIGHT_KEYWORD", "0.25")),
    "graph": float(os.getenv("SCORE_WEIGHT_GRAPH", "0.25")),
}

_SPACY_PRIORITY = [
    "en_core_web_trf",
    "en_core_web_lg",
    "en_core_web_md",
    "en_core_web_sm",
]


def _load_spacy():
    for name in _SPACY_PRIORITY:
        try:
            model = spacy.load(name)
            log.info("spaCy model loaded: %s", name)
            return model
        except OSError:
            continue
    log.warning(
        "No spaCy model found. NER and graph retrieval are reduced. "
        "Install one with: python -m spacy download en_core_web_sm"
    )
    return spacy.blank("en")


nlp = _load_spacy()

def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {cid: 1.0 if max_score > 0 else 0.0 for cid in scores}
    return {cid: (score - min_score) / (max_score - min_score) for cid, score in scores.items()}


GRAPHRAG_PROMPT_TEMPLATE = """You are an expert assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't have enough context to answer this."

Context (retrieved from a knowledge graph, vector store, and BM25 index):
---
{context}
---

Question: {question}

Answer concisely. Do not reference source numbers in your answer:"""


class GraphRAGRetriever:
    """Hybrid GraphMind retriever with concurrent retrieval paths."""

    def __init__(
        self,
        llm_provider: str = LLM_PROVIDER,
        llm_model: Optional[str] = None,
        top_k_vector: int = TOP_K_VECTOR,
        top_k_keyword: int = TOP_K_KEYWORD,
        max_hops: int = MAX_HOPS,
    ):
        self.llm_provider = (llm_provider or LLM_PROVIDER).strip().lower()
        self.llm_model = (llm_model or os.getenv("LLM_MODEL") or LLM_MODEL).strip()
        self.top_k_vector = top_k_vector
        self.top_k_keyword = top_k_keyword
        self.max_hops = max_hops
        self.executor = ThreadPoolExecutor(max_workers=3)

        self._load_artifacts()
        self._validate_llm_config()

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _load_artifacts(self) -> None:
        with open(CHUNKS_PATH, "rb") as f:
            chunk_list = pickle.load(f)
        self.chunk_lookup: Dict[str, str] = {
            chunk["chunk_id"]: chunk["text"] for chunk in chunk_list
        }

        with open(GRAPH_PATH, "rb") as f:
            self.G: nx.DiGraph = pickle.load(f)

        self.faiss_index, self.faiss_id_to_chunk_id = load_faiss_index()
        self.embed_model = load_embedding_model()
        self.lexical_index = load_lexical_index() if INVERTED_INDEX_PATH.exists() else None
        self.nlp_query = nlp
        self.graph_pagerank = (
            nx.pagerank(self.G) if self.G.number_of_nodes() else {}
        )

    def _validate_llm_config(self) -> None:
        if self.llm_provider not in {"ollama", "openai", "gemini"}:
            raise ValueError("Unsupported LLM_PROVIDER. Use: ollama, openai, or gemini.")
        if not self.llm_model:
            raise ValueError("LLM_MODEL must be set in .env.")
        if self.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY in .env.")
        if self.llm_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY in .env.")

    def _vector_retrieve(self, query: str) -> List[Dict]:
        results = query_index(
            query,
            self.embed_model,
            self.faiss_index,
            self.faiss_id_to_chunk_id,
            top_k=self.top_k_vector,
            verbose=False,
        )
        return results

    def _keyword_retrieve(self, query: str) -> List[Dict]:
        if self.lexical_index is None:
            return []
        return search_index(query, self.lexical_index, top_k=self.top_k_keyword)

    def _extract_query_entities(self, query: str) -> List[str]:
        regex_candidates = re.findall(r"\b[A-Z][A-Za-z0-9_\-]{1,}\b", query)
        doc = self.nlp_query(query) if self.nlp_query is not None else None
        entities = [ent.text for ent in doc.ents] if doc is not None else []
        try:
            noun_chunks = [
                chunk.text for chunk in doc.noun_chunks
                if len(chunk.text.split()) <= 3
            ] if doc is not None else []
        except ValueError:
            noun_chunks = []
        return list(dict.fromkeys(entities + noun_chunks + regex_candidates))

    def _find_graph_nodes(self, entity_texts: List[str]) -> List[str]:
        matched = []
        graph_nodes_lower = {node.lower(): node for node in self.G.nodes()}
        for entity in entity_texts:
            entity_lower = entity.lower().strip()
            if entity_lower in graph_nodes_lower:
                matched.append(graph_nodes_lower[entity_lower])
                continue
            for node_lower, node in graph_nodes_lower.items():
                if entity_lower in node_lower or node_lower in entity_lower:
                    matched.append(node)
                    break
        return list(dict.fromkeys(matched))

    def _graph_retrieve(self, query: str) -> Tuple[Dict[str, float], List[str], List[str], List[str]]:
        entities = self._extract_query_entities(query)
        matched_nodes = self._find_graph_nodes(entities)
        if not matched_nodes:
            return {}, entities, [], []

        graph_scores: Counter = Counter()
        traversed_node_keys: List[str] = []
        for node in matched_nodes:
            traversal = bfs_traverse(self.G, node, max_hops=self.max_hops, verbose=False)
            for tnode, chunk_ids in traversal.items():
                if tnode not in traversed_node_keys:
                    traversed_node_keys.append(tnode)
                node_score = (
                    1.0
                    + self.graph_pagerank.get(tnode, 0.0)
                    + self.G.nodes[tnode].get("degree_centrality", 0.0)
                )
                for cid in chunk_ids:
                    graph_scores[cid] += node_score
        return dict(graph_scores), entities, traversed_node_keys, matched_nodes

    def _rank_context(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        graph_scores: Dict[str, float],
        max_chunks: int = MAX_CONTEXT_CHUNKS,
    ) -> Tuple[str, List[str], Dict[str, Dict[str, float]]]:
        vector_scores = _normalize_scores({r["chunk_id"]: r["score"] for r in vector_results})
        keyword_scores = _normalize_scores({r["chunk_id"]: r["score"] for r in keyword_results})
        graph_norm = _normalize_scores(graph_scores)

        all_cids = set(vector_scores) | set(keyword_scores) | set(graph_norm)
        score_trace: Dict[str, Dict[str, float]] = {}
        ranked = []
        for cid in all_cids:
            if cid not in self.chunk_lookup:
                continue
            components = {
                "vector": vector_scores.get(cid, 0.0),
                "keyword": keyword_scores.get(cid, 0.0),
                "graph": graph_norm.get(cid, 0.0),
            }
            total = (
                SCORE_WEIGHTS["vector"] * components["vector"]
                + SCORE_WEIGHTS["keyword"] * components["keyword"]
                + SCORE_WEIGHTS["graph"] * components["graph"]
            )
            components["total"] = total
            score_trace[cid] = components
            ranked.append((cid, total))

        final_cids = [
            cid for cid, _ in sorted(ranked, key=lambda item: item[1], reverse=True)[:max_chunks]
        ]
        context_parts = [
            f"[Context {idx + 1} | {cid} | score={score_trace[cid]['total']:.3f}]\n"
            f"{self.chunk_lookup[cid]}"
            for idx, cid in enumerate(final_cids)
        ]
        return "\n\n".join(context_parts), final_cids, score_trace

    def _call_llm(self, prompt: str) -> str:
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                response = OpenAI().chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                answer = response.choices[0].message.content.strip()
            elif self.llm_provider == "gemini":
                from google import genai
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model=self.llm_model,
                    contents=prompt,
                )
                answer = (response.text or "").strip()
            else:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response["message"]["content"].strip()
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            answer = f"[Error calling LLM: {exc}]"
        return answer

    def query(self, question: str) -> Dict:
        f_vec = self.executor.submit(self._vector_retrieve, question)
        f_kw = self.executor.submit(self._keyword_retrieve, question)
        f_gr = self.executor.submit(self._graph_retrieve, question)

        vector_results = f_vec.result()
        keyword_results = f_kw.result()
        graph_scores, identified_entities, traversed_nodes, matched_nodes = f_gr.result()

        vector_cids = [r["chunk_id"] for r in vector_results]
        keyword_cids = [r["chunk_id"] for r in keyword_results]
        graph_cids = set(graph_scores)

        context, final_cids, score_trace = self._rank_context(
            vector_results,
            keyword_results,
            graph_scores,
        )
        prompt = GRAPHRAG_PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self._call_llm(prompt)

        return {
            "graphrag_answer": answer,
            "trace": {
                "identified_entities": identified_entities,
                "matched_graph_nodes": matched_nodes,
                "traversed_nodes": traversed_nodes,
                "vector_chunk_ids": vector_cids,
                "keyword_chunk_ids": keyword_cids,
                "graph_chunk_ids": list(graph_cids),
                "final_chunk_ids": final_cids,
                "scores": score_trace,
                "context": context,
            },
        }
