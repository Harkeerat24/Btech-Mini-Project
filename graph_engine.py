"""Build and traverse the GraphMind knowledge graph."""

import pickle
import logging
import argparse
import shutil
import re
from difflib import SequenceMatcher
from collections import deque
from pathlib import Path
from typing import List, Dict, Set, Optional

import networkx as nx

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graph_engine")

DATA_DIR = Path(__file__).parent / "data"
GRAPH_PATH = DATA_DIR / "graph.pkl"
TRIPLETS_PATH = DATA_DIR / "triplets.pkl"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"

ENTITY_COLORS = {
    "PERSON": "#F39C12",
    "ORG": "#1ABC9C",
    "GPE": "#E74C3C",
    "LOC": "#E74C3C",
    "PRODUCT": "#E67E22",
    "EVENT": "#D35400",
    "WORK_OF_ART": "#8E44AD",
    "LAW": "#3498DB",
    "LANGUAGE": "#2ECC71",
    "NORP": "#9B59B6",
    "FAC": "#16A085",
    "DEFAULT": "#95A5A6",
}


def _normalize(text: str) -> str:
    """Build and traverse the GraphMind knowledge graph."""
    return " ".join(text.strip().lower().split())


LEGAL_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|corp|corporation|co|company|ltd|limited|llc|plc)\b\.?",
    re.IGNORECASE,
)


def _resolution_key(text: str) -> str:
    text = LEGAL_SUFFIX_RE.sub("", text or "")
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    return " ".join(text.lower().split())


class EntityResolver:
    """Build and traverse the GraphMind knowledge graph."""

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self._canonical_by_signature: Dict[str, str] = {}
        self._signatures_by_bucket: Dict[str, List[str]] = {}

    @staticmethod
    def _bucket(signature: str) -> str:
        first_token = signature.split(" ", 1)[0] if signature else ""
        return first_token[:3]

    def resolve(self, text: str) -> str:
        candidate = _resolution_key(text)
        if not candidate:
            return _normalize(text)
        if candidate in self._canonical_by_signature:
            return self._canonical_by_signature[candidate]

        bucket = self._bucket(candidate)
        candidates = self._signatures_by_bucket.get(bucket, [])
        if not candidates:
            candidates = [
                signature for signature in self._canonical_by_signature
                if signature[:1] == candidate[:1]
            ]

        best_key = None
        best_score = 0.0
        for signature in candidates:
            canonical = self._canonical_by_signature[signature]
            if candidate in signature or signature in candidate:
                score = min(len(candidate), len(signature)) / max(len(candidate), len(signature))
            else:
                score = SequenceMatcher(None, candidate, signature).ratio()
            if score > best_score:
                best_score = score
                best_key = canonical

        if best_key and best_score >= self.threshold:
            self._canonical_by_signature[candidate] = best_key
            return best_key

        self._canonical_by_signature[candidate] = candidate
        self._signatures_by_bucket.setdefault(bucket, []).append(candidate)
        return candidate


def infer_entity_type(label: str, node_key: str) -> str:
    """Build and traverse the GraphMind knowledge graph."""
    spacy_to_type = {
        "PERSON":    "PERSON",
        "ORG":       "ORG",
        "GPE":       "GPE",
        "LOC":       "LOC",
        "PRODUCT":   "PRODUCT",
        "EVENT":     "EVENT",
        "WORK_OF_ART": "WORK_OF_ART",
        "LAW":       "LAW",
        "LANGUAGE":  "LANGUAGE",
        "NORP":      "NORP",
        "FAC":       "FAC",
    }
    return spacy_to_type.get(label, "DEFAULT")


def _merge_node_attrs(G: nx.DiGraph, key: str, label: str, entity_type: str, chunk_id: str) -> None:
    if key not in G:
        G.add_node(key, **{
            "label": label,
            "entity_type": entity_type,
            "chunk_ids": [],
            "color": ENTITY_COLORS.get(entity_type, ENTITY_COLORS["DEFAULT"]),
        })
    if chunk_id not in G.nodes[key]["chunk_ids"]:
        G.nodes[key]["chunk_ids"].append(chunk_id)
    if G.nodes[key].get("entity_type") == "DEFAULT" and entity_type != "DEFAULT":
        G.nodes[key]["entity_type"] = entity_type
        G.nodes[key]["color"] = ENTITY_COLORS.get(
            entity_type, ENTITY_COLORS["DEFAULT"])


def _add_or_update_edge(
    G: nx.DiGraph,
    src: str,
    dst: str,
    predicate: str,
    chunk_id: str,
    relation_type: str,
) -> None:
    if src == dst:
        return
    if G.has_edge(src, dst):
        G[src][dst]["weight"] += 1
        G[src][dst].setdefault("chunk_ids", [])
        if chunk_id not in G[src][dst]["chunk_ids"]:
            G[src][dst]["chunk_ids"].append(chunk_id)
        G[src][dst].setdefault("relation_types", [])
        if relation_type not in G[src][dst]["relation_types"]:
            G[src][dst]["relation_types"].append(relation_type)
        if predicate not in G[src][dst]["predicates"]:
            G[src][dst]["predicates"].append(predicate)
    else:
        G.add_edge(
            src,
            dst,
            predicate=predicate,
            predicates=[predicate],
            relation_types=[relation_type],
            chunk_ids=[chunk_id],
            weight=1,
        )


def add_entity_nodes_and_cooccurrence_edges(
    G: nx.DiGraph,
    chunks: List[Dict],
    resolver: Optional[EntityResolver] = None,
) -> nx.DiGraph:
    """Build and traverse the GraphMind knowledge graph."""
    for chunk in chunks:
        cid = chunk["chunk_id"]
        seen_keys = []
        for ent in chunk.get("entities", []):
            key = resolver.resolve(ent.get("text", "")) if resolver else _normalize(ent.get("text", ""))
            if not key or key in seen_keys:
                continue
            etype = infer_entity_type(ent.get("label", ""), key)
            _merge_node_attrs(G, key, ent.get("text", key), etype, cid)
            seen_keys.append(key)

        for i, src in enumerate(seen_keys):
            for dst in seen_keys[i + 1:]:
                _add_or_update_edge(
                    G, src, dst, "co_occurs_with", cid, "cooccurrence")
                _add_or_update_edge(
                    G, dst, src, "co_occurs_with", cid, "cooccurrence")
    return G


def _entity_type_lookup_resolved(chunks: List[Dict], resolver: EntityResolver) -> Dict[str, str]:
    lookup = {}
    for chunk in chunks:
        for ent in chunk.get("entities", []):
            key = resolver.resolve(ent.get("text", ""))
            if key:
                lookup[key] = infer_entity_type(ent.get("label", ""), key)
    return lookup


def build_graph(triplets: List[Dict], chunks: Optional[List[Dict]] = None) -> nx.DiGraph:
    """Build and traverse the GraphMind knowledge graph."""
    log.info("Building NetworkX DiGraph from NER entities and triplets ...")
    G = nx.DiGraph()
    chunks = chunks or []
    resolver = EntityResolver()
    entity_types = _entity_type_lookup_resolved(chunks, resolver)
    add_entity_nodes_and_cooccurrence_edges(G, chunks, resolver)

    for t in triplets:
        subj_key = resolver.resolve(t["subject"])
        obj_key = resolver.resolve(t["object"])
        pred = t["predicate"].strip()
        cid = t["chunk_id"]
        subj_type = entity_types.get(subj_key, infer_entity_type("", subj_key))
        _merge_node_attrs(G, subj_key, t["subject"], subj_type, cid)
        obj_type = entity_types.get(obj_key, infer_entity_type("", obj_key))
        _merge_node_attrs(G, obj_key, t["object"], obj_type, cid)
        _add_or_update_edge(G, subj_key, obj_key, pred,
                            cid, "extracted_triplet")

    log.info(
        f"  Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_centrality(G: nx.DiGraph) -> nx.DiGraph:
    """Build and traverse the GraphMind knowledge graph."""
    log.info("Computing centrality metrics ...")

    undirected = G.to_undirected()
    deg_cen = nx.degree_centrality(undirected)
    if G.number_of_nodes() > 500:
        log.info("  Large graph detected â€” using approximate betweenness (k=100)")
        btw_cen = nx.betweenness_centrality(undirected, k=100, normalized=True)
    else:
        btw_cen = nx.betweenness_centrality(undirected, normalized=True)

    for node in G.nodes():
        G.nodes[node]["degree_centrality"] = round(deg_cen.get(node, 0.0), 4)
        G.nodes[node]["betweenness_centrality"] = round(
            btw_cen.get(node, 0.0), 4)
        G.nodes[node]["in_degree"] = G.in_degree(node)
        G.nodes[node]["out_degree"] = G.out_degree(node)

    log.info("  Centrality attached to all nodes.")
    return G


def bfs_traverse(
    G: nx.DiGraph,
    start_node: str,
    max_hops: int = 2,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Build and traverse the GraphMind knowledge graph."""
    if start_node not in G:
        if verbose:
            log.warning(f"[BFS] Node '{start_node}' not found in graph.")
        return {}

    if verbose:
        log.info(
            f"[BFS] â”€â”€â”€ Starting traversal from: '{start_node}' (max_hops={max_hops}) â”€â”€â”€")

    collected: Dict[str, List[str]] = {}
    visited: Set[str] = {start_node}
    queue = deque([(start_node, 0)])
    cids = G.nodes[start_node].get("chunk_ids", [])
    collected[start_node] = cids
    if verbose:
        log.info(
            f"[BFS]   Root node '{start_node}' â†’ chunk_ids: {sorted(cids)}")

    while queue:
        current, hop = queue.popleft()
        if hop >= max_hops:
            continue
        neighbors = list(G.successors(current)) + list(G.predecessors(current))
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            if G.has_edge(current, neighbor):
                edge_data = G[current][neighbor]
                direction = f"{current} â†’[{edge_data['predicate']}]â†’ {neighbor}"
            else:
                edge_data = G[neighbor][current]
                direction = f"{neighbor} â†[{edge_data['predicate']}]â† {current}"

            neighbor_cids = G.nodes[neighbor].get("chunk_ids", [])
            collected[neighbor] = neighbor_cids

            if verbose:
                log.info(
                    f"[BFS]   Hop {hop + 1} | {direction} "
                    f"| chunk_ids: {sorted(neighbor_cids)}"
                )

            queue.append((neighbor, hop + 1))

    if verbose:
        total_cids = set().union(*(set(cids) for cids in collected.values())) if collected else set()
        log.info(f"[BFS] â”€â”€â”€ Done. Visited {len(collected)} nodes, "
                 f"collected {len(total_cids)} unique chunk_ids â”€â”€â”€")

    return collected


def get_all_chunk_ids_for_entities(
    G: nx.DiGraph,
    entity_keys: List[str],
    max_hops: int = 2,
    verbose: bool = True,
) -> Set[str]:
    """Build and traverse the GraphMind knowledge graph."""
    all_cids: Set[str] = set()
    for key in entity_keys:
        traversal = bfs_traverse(G, key, max_hops=max_hops, verbose=verbose)
        for cids in traversal.values():
            all_cids.update(cids)
    return all_cids


def save_graph(G: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    for _, attrs in G.nodes(data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, set):
                attrs[key] = sorted(value)
    for _, _, attrs in G.edges(data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, set):
                attrs[key] = sorted(value)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Graph saved -> {path}")


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def print_top_nodes(G: nx.DiGraph, n: int = 10) -> None:
    """Build and traverse the GraphMind knowledge graph."""
    nodes = sorted(
        G.nodes(data=True),
        key=lambda x: x[1].get("betweenness_centrality", 0),
        reverse=True,
    )
    term_width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    sep = "-" * term_width
    node_width = max(18, min(36, term_width - 40))
    print(f"\n{sep}")
    print(f"  TOP-{n} HUB NODES (by betweenness centrality)")
    print(sep)
    print(f"  {'Node':<{node_width}} {'Type':<10} {'Btw':>7} {'Deg':>7} Chunks")
    print(sep)
    for node, attrs in nodes[:n]:
        label = attrs.get('label', node)
        try:
            label_safe = label.encode(
                'cp1252', errors='replace').decode('cp1252')
        except Exception:
            label_safe = label
        if len(label_safe) > node_width:
            label_safe = label_safe[:node_width - 1] + "â€¦"
        print(
            f"  {label_safe:<{node_width}} "
            f"{attrs.get('entity_type', 'DEFAULT'):<10} "
            f"{attrs.get('betweenness_centrality', 0):>7.4f} "
            f"{attrs.get('degree_centrality', 0):>7.4f} "
            f"{len(attrs.get('chunk_ids', []))}"
        )
    print(f"{sep}\n")



def build(verbose: bool = True) -> nx.DiGraph:
    """Build and traverse the GraphMind knowledge graph."""
    log.info("=" * 60)
    log.info("GRAPH ENGINE â€” BUILD PIPELINE START")
    log.info("=" * 60)

    with open(TRIPLETS_PATH, "rb") as f:
        triplets = pickle.load(f)
    log.info(f"Loaded {len(triplets)} triplets from {TRIPLETS_PATH}")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    G = build_graph(triplets, chunks)
    G = compute_centrality(G)
    save_graph(G)

    log.info("=" * 60)
    log.info("GRAPH BUILD COMPLETE")
    log.info("=" * 60)
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphMind graph builder")
    parser.add_argument("--show", action="store_true",
                        help="Print top hub nodes after build")
    parser.add_argument("--bfs_demo", type=str, default=None,
                        help="Node key to run BFS demo from (e.g. 'ospf')")
    args = parser.parse_args()

    G = build()

    if args.show:
        print_top_nodes(G, n=10)

    if args.bfs_demo:
        print(f"\n--- BFS Demo from node: '{args.bfs_demo}' ---")
        result = bfs_traverse(G, args.bfs_demo.lower(),
                              max_hops=2, verbose=True)
        all_cids = set().union(*result.values()) if result else set()
        cid_text = ", ".join(sorted(all_cids)) if all_cids else "none"
        print(f"Total chunk_ids retrieved: {cid_text}")

