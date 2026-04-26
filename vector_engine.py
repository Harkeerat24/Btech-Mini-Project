import textwrap
import shutil
import pickle
import logging
import inspect
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()

"""
vector_engine.py — GraphRAG System
=====================================
Builds a FAISS index from chunk embeddings using SentenceTransformers.

Usage:
  python vector_engine.py          # uses data/chunks.pkl
  python vector_engine.py --query "How does OSPF handle link failure?"
"""


# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vector_engine")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
FAISS_PATH = DATA_DIR / "faiss_index.bin"
CHUNK_MAP_PATH = DATA_DIR / "chunk_map.pkl"

# Embedding model — fast, offline, 384-dim
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    log.info(f"Loading SentenceTransformer: {EMBED_MODEL_NAME} ...")
    init_kwargs = {}
    if "show_progress_bar" in inspect.signature(SentenceTransformer.__init__).parameters:
        init_kwargs["show_progress_bar"] = False
    model = SentenceTransformer(EMBED_MODEL_NAME, **init_kwargs)
    try:
        dim = model.get_embedding_dimension()
    except AttributeError:
        dim = model.get_sentence_embedding_dimension()
    log.info(f"  Embedding dim: {dim}")
    return model


def embed_chunks(
    model: SentenceTransformer,
    chunks: List[Dict],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Embed all chunk texts.

    Returns:
      embeddings  — np.ndarray of shape (N, dim), dtype float32
      int_to_cid  — {faiss_int_id: chunk_id}
    """
    log.info(f"Embedding {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    int_to_cid = {i: c["chunk_id"] for i, c in enumerate(chunks)}

    # Batch encode — SentenceTransformers handles batching internally
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # normalize for cosine-equivalent L2 search
    ).astype("float32")

    log.info(f"  Embeddings shape: {embeddings.shape}")
    return embeddings, int_to_cid


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a flat L2 index (exact search, no training required).
    With normalized embeddings, L2 == cosine similarity ranking.
    """
    log.info("Building FAISS IndexFlatL2 ...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    log.info(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatL2, int_to_cid: Dict[int, str]) -> None:
    faiss.write_index(index, str(FAISS_PATH))
    log.info(f"  FAISS index saved → {FAISS_PATH}")
    with open(CHUNK_MAP_PATH, "wb") as f:
        pickle.dump(int_to_cid, f)
    log.info(f"  Chunk map saved   → {CHUNK_MAP_PATH}")


def load_index() -> Tuple[faiss.IndexFlatL2, Dict[int, str]]:
    index = faiss.read_index(str(FAISS_PATH))
    with open(CHUNK_MAP_PATH, "rb") as f:
        int_to_cid = pickle.load(f)
    log.info(f"  FAISS index loaded: {index.ntotal} vectors")
    return index, int_to_cid


def query_index(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    int_to_cid: Dict[int, str],
    top_k: int = 5,
    verbose: bool = True,
) -> List[Dict]:
    """
    Embed a query and return top-k results.

    Returns list of:
      { "chunk_id": str, "score": float, "rank": int }
    """
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")

    distances, indices = index.search(q_emb, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        cid = int_to_cid[int(idx)]
        results.append({
            "chunk_id": cid,
            "score":    float(dist),
            "rank":     rank + 1,
        })
        if verbose:
            log.info(
                f"[FAISS] Rank {rank+1}: chunk_id={cid}  L2_dist={dist:.4f}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def build() -> Tuple[faiss.IndexFlatL2, Dict[int, str], SentenceTransformer]:
    """Full vector index build pipeline."""
    log.info("=" * 60)
    log.info("VECTOR ENGINE — BUILD PIPELINE START")
    log.info("=" * 60)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    model = load_embedding_model()
    embeddings, int_to_cid = embed_chunks(model, chunks)
    index = build_faiss_index(embeddings)
    save_index(index, int_to_cid)

    log.info("=" * 60)
    log.info("VECTOR ENGINE BUILD COMPLETE")
    log.info("=" * 60)
    return index, int_to_cid, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraphRAG — Vector Index Builder")
    parser.add_argument("--query", type=str, default=None,
                        help="Optional test query to run after building")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    index, int_to_cid, model = build()

    if args.query:
        term_width = min(shutil.get_terminal_size((100, 20)).columns, 120)
        print(f"\n--- Test Query: '{args.query}' ---")
        results = query_index(args.query, model, index,
                              int_to_cid, top_k=args.top_k)
        for r in results:
            line = f"Rank {r['rank']}: {r['chunk_id']} (score={r['score']:.4f})"
            wrapped = textwrap.wrap(line, width=max(30, term_width - 4))
            for i, part in enumerate(wrapped):
                print(f"  {part}" if i == 0 else f"    {part}")
