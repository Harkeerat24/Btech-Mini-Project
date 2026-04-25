"""
graph_engine.py — GraphRAG System
====================================
Builds a NetworkX DiGraph from extracted triplets.
Each node stores references to its source chunk_ids.

Usage:
  python graph_engine.py          # uses data/triplets.pkl
  python graph_engine.py --show   # prints top-10 nodes by centrality
"""

import pickle
import logging
import argparse
from collections import deque
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import networkx as nx

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graph_engine")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
GRAPH_PATH  = DATA_DIR / "graph.pkl"
TRIPLETS_PATH = DATA_DIR / "triplets.pkl"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"

# ─── Color Mapping (entity_type → hex color) ──────────────────────────────────
ENTITY_COLORS = {
    "PROTOCOL":  "#3498DB",   # Blue
    "ALGORITHM": "#2ECC71",   # Green
    "CONCEPT":   "#9B59B6",   # Purple
    "PERSON":    "#F39C12",   # Orange
    "ORG":       "#1ABC9C",   # Teal
    "PRODUCT":   "#E67E22",   # Dark Orange
    "GPE":       "#E74C3C",   # Red
    "LOC":       "#E74C3C",   # Red
    "EVENT":     "#D35400",   # Burnt Orange
    "WORK_OF_ART": "#8E44AD", # Violet
    "DEFAULT":   "#95A5A6",   # Gray
}


def _normalize(text: str) -> str:
    """Consistent node key: lowercase, stripped."""
    return " ".join(text.strip().lower().split())


def infer_entity_type(label: str, node_key: str) -> str:
    """
    Infer entity type from spaCy label or domain keyword heuristics.
    Falls back to 'DEFAULT'.
    """
    spacy_to_type = {
        "PERSON":    "PERSON",
        "ORG":       "ORG",
        "GPE":       "GPE",
        "LOC":       "LOC",
        "PRODUCT":   "PRODUCT",
        "EVENT":     "EVENT",
        "WORK_OF_ART": "WORK_OF_ART",
        "PROTOCOL":  "PROTOCOL",
        "ALGORITHM": "ALGORITHM",
        "CONCEPT":   "CONCEPT",
    }
    if label in spacy_to_type:
        return spacy_to_type[label]
    # Heuristic fallback based on known keywords
    nk = node_key.lower()
    if any(w in nk for w in ["ospf", "bgp", "tcp", "udp", "http", "dns", "dhcp", "mpls", "rip"]):
        return "PROTOCOL"
    if any(w in nk for w in ["dijkstra", "bellman", "floyd", "algorithm", "sort"]):
        return "ALGORITHM"
    return "DEFAULT"


def _merge_node_attrs(G: nx.DiGraph, key: str, label: str, entity_type: str, chunk_id: str) -> None:
    if key not in G:
        G.add_node(key, **{
            "label": label,
            "entity_type": entity_type,
            "chunk_ids": set(),
            "color": ENTITY_COLORS.get(entity_type, ENTITY_COLORS["DEFAULT"]),
        })
    G.nodes[key]["chunk_ids"].add(chunk_id)
    if G.nodes[key].get("entity_type") == "DEFAULT" and entity_type != "DEFAULT":
        G.nodes[key]["entity_type"] = entity_type
        G.nodes[key]["color"] = ENTITY_COLORS.get(entity_type, ENTITY_COLORS["DEFAULT"])


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
        G[src][dst].setdefault("chunk_ids", set()).add(chunk_id)
        G[src][dst].setdefault("relation_types", set()).add(relation_type)
        if predicate not in G[src][dst]["predicates"]:
            G[src][dst]["predicates"].append(predicate)
    else:
        G.add_edge(
            src,
            dst,
            predicate=predicate,
            predicates=[predicate],
            relation_types={relation_type},
            chunk_ids={chunk_id},
            weight=1,
        )


def add_entity_nodes_and_cooccurrence_edges(G: nx.DiGraph, chunks: List[Dict]) -> nx.DiGraph:
    """
    Add all NER entities as nodes and connect entities that co-occur in a
    chunk. This keeps the KG useful even when dependency parsing misses a
    subject-verb-object relation.
    """
    for chunk in chunks:
        cid = chunk["chunk_id"]
        seen_keys = []
        for ent in chunk.get("entities", []):
            key = _normalize(ent.get("text", ""))
            if not key or key in seen_keys:
                continue
            etype = infer_entity_type(ent.get("label", ""), key)
            _merge_node_attrs(G, key, ent.get("text", key), etype, cid)
            seen_keys.append(key)

        for i, src in enumerate(seen_keys):
            for dst in seen_keys[i + 1:]:
                _add_or_update_edge(G, src, dst, "co_occurs_with", cid, "cooccurrence")
                _add_or_update_edge(G, dst, src, "co_occurs_with", cid, "cooccurrence")
    return G


def _entity_type_lookup(chunks: List[Dict]) -> Dict[str, str]:
    lookup = {}
    for chunk in chunks:
        for ent in chunk.get("entities", []):
            key = _normalize(ent.get("text", ""))
            if key:
                lookup[key] = infer_entity_type(ent.get("label", ""), key)
    return lookup


def build_graph(triplets: List[Dict], chunks: Optional[List[Dict]] = None) -> nx.DiGraph:
    """
    Build a directed graph from S-P-O triplets.

    Node attributes:
      - label      : display text (original case)
      - entity_type: one of ENTITY_COLORS keys
      - chunk_ids  : set of chunk_ids this entity appears in
      - color      : hex color

    Edge attributes:
      - predicate  : relationship verb (lemmatized)
      - weight     : count of times this edge was seen
    """
    log.info("Building NetworkX DiGraph from NER entities and triplets ...")
    G = nx.DiGraph()
    chunks = chunks or []
    entity_types = _entity_type_lookup(chunks)
    add_entity_nodes_and_cooccurrence_edges(G, chunks)

    for t in triplets:
        subj_key = _normalize(t["subject"])
        obj_key  = _normalize(t["object"])
        pred     = t["predicate"].strip()
        cid      = t["chunk_id"]

        # ── Add / update SUBJECT node ───────────────────────────
        subj_type = entity_types.get(subj_key, infer_entity_type("", subj_key))
        _merge_node_attrs(G, subj_key, t["subject"], subj_type, cid)

        # ── Add / update OBJECT node ────────────────────────────
        obj_type = entity_types.get(obj_key, infer_entity_type("", obj_key))
        _merge_node_attrs(G, obj_key, t["object"], obj_type, cid)

        # ── Add / update EDGE ───────────────────────────────────
        _add_or_update_edge(G, subj_key, obj_key, pred, cid, "extracted_triplet")

    log.info(f"  Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_centrality(G: nx.DiGraph) -> nx.DiGraph:
    """
    Compute and attach centrality metrics to every node.
      - degree_centrality
      - betweenness_centrality (approximated for large graphs)
      - in_degree / out_degree
    """
    log.info("Computing centrality metrics ...")

    undirected = G.to_undirected()
    deg_cen = nx.degree_centrality(undirected)

    # Use approximation if graph is large (> 500 nodes)
    if G.number_of_nodes() > 500:
        log.info("  Large graph detected — using approximate betweenness (k=100)")
        btw_cen = nx.betweenness_centrality(undirected, k=100, normalized=True)
    else:
        btw_cen = nx.betweenness_centrality(undirected, normalized=True)

    for node in G.nodes():
        G.nodes[node]["degree_centrality"]      = round(deg_cen.get(node, 0.0), 4)
        G.nodes[node]["betweenness_centrality"] = round(btw_cen.get(node, 0.0), 4)
        G.nodes[node]["in_degree"]              = G.in_degree(node)
        G.nodes[node]["out_degree"]             = G.out_degree(node)

    log.info("  Centrality attached to all nodes.")
    return G


def bfs_traverse(
    G: nx.DiGraph,
    start_node: str,
    max_hops: int = 2,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    BFS traversal from a start_node up to max_hops.

    Returns:
      { node_key: set_of_chunk_ids }  for all discovered nodes.

    Console output shows every BFS step (for viva demo):
      [BFS] Hop 1 | OSPF → hello_packets (uses)
      [BFS] Hop 1 | OSPF → dijkstra (use)
      [BFS] Hop 2 | hello_packets → dead_interval (requires)
    """
    if start_node not in G:
        if verbose:
            log.warning(f"[BFS] Node '{start_node}' not found in graph.")
        return {}

    if verbose:
        log.info(f"[BFS] ─── Starting traversal from: '{start_node}' (max_hops={max_hops}) ───")

    collected: Dict[str, Set[str]] = {}
    visited: Set[str] = {start_node}
    # Queue entries: (node_key, current_hop)
    queue = deque([(start_node, 0)])

    # Include the start node itself
    cids = G.nodes[start_node].get("chunk_ids", set())
    collected[start_node] = cids
    if verbose:
        log.info(f"[BFS]   Root node '{start_node}' → chunk_ids: {sorted(cids)}")

    while queue:
        current, hop = queue.popleft()
        if hop >= max_hops:
            continue

        # Explore both outgoing and incoming edges (undirected BFS on DiGraph)
        neighbors = list(G.successors(current)) + list(G.predecessors(current))
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)

            # Get the edge label (try both directions)
            if G.has_edge(current, neighbor):
                edge_data = G[current][neighbor]
                direction = f"{current} →[{edge_data['predicate']}]→ {neighbor}"
            else:
                edge_data = G[neighbor][current]
                direction = f"{neighbor} ←[{edge_data['predicate']}]← {current}"

            neighbor_cids = G.nodes[neighbor].get("chunk_ids", set())
            collected[neighbor] = neighbor_cids

            if verbose:
                log.info(
                    f"[BFS]   Hop {hop + 1} | {direction} "
                    f"| chunk_ids: {sorted(neighbor_cids)}"
                )

            queue.append((neighbor, hop + 1))

    if verbose:
        total_cids = set().union(*collected.values()) if collected else set()
        log.info(f"[BFS] ─── Done. Visited {len(collected)} nodes, "
                 f"collected {len(total_cids)} unique chunk_ids ───")

    return collected


def get_all_chunk_ids_for_entities(
    G: nx.DiGraph,
    entity_keys: List[str],
    max_hops: int = 2,
    verbose: bool = True,
) -> Set[str]:
    """
    For a list of entity keys, run BFS from each and return the
    union of all collected chunk_ids.
    """
    all_cids: Set[str] = set()
    for key in entity_keys:
        traversal = bfs_traverse(G, key, max_hops=max_hops, verbose=verbose)
        for cids in traversal.values():
            all_cids.update(cids)
    return all_cids


def save_graph(G: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Graph saved -> {path}")


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def print_top_nodes(G: nx.DiGraph, n: int = 10) -> None:
    """Print top-n nodes ranked by betweenness centrality (hub nodes)."""
    nodes = sorted(
        G.nodes(data=True),
        key=lambda x: x[1].get("betweenness_centrality", 0),
        reverse=True,
    )
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  TOP-{n} HUB NODES (by betweenness centrality)")
    print(sep)
    print(f"  {'Node':<30} {'Type':<12} {'Btw':>6}  {'Deg':>6}  Chunks")
    print(sep)
    for node, attrs in nodes[:n]:
        label = attrs.get('label', node)
        # Encode safely for Windows terminals
        try:
            label_safe = label.encode('cp1252', errors='replace').decode('cp1252')
        except Exception:
            label_safe = label
        print(
            f"  {label_safe:<30} "
            f"{attrs.get('entity_type', 'DEFAULT'):<12} "
            f"{attrs.get('betweenness_centrality', 0):>6.4f}  "
            f"{attrs.get('degree_centrality', 0):>6.4f}  "
            f"{len(attrs.get('chunk_ids', []))}"
        )
    print(f"{sep}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def build(verbose: bool = True) -> nx.DiGraph:
    """Full graph build pipeline: load triplets → build → centrality → save."""
    log.info("=" * 60)
    log.info("GRAPH ENGINE — BUILD PIPELINE START")
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
    parser = argparse.ArgumentParser(description="GraphRAG — Graph Builder")
    parser.add_argument("--show", action="store_true", help="Print top hub nodes after build")
    parser.add_argument("--bfs_demo", type=str, default=None,
                        help="Node key to run BFS demo from (e.g. 'ospf')")
    args = parser.parse_args()

    G = build()

    if args.show:
        print_top_nodes(G, n=10)

    if args.bfs_demo:
        print(f"\n--- BFS Demo from node: '{args.bfs_demo}' ---")
        result = bfs_traverse(G, args.bfs_demo.lower(), max_hops=2, verbose=True)
        all_cids = set().union(*result.values()) if result else set()
        print(f"Total chunk_ids retrieved: {sorted(all_cids)}")
