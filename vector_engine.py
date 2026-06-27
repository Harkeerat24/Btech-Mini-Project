import textwrap
import shutil
import pickle
import logging
import inspect
import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

"""Build and query the FAISS and BM25 indexes."""


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vector_engine")

DATA_DIR = Path(__file__).parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
FAISS_PATH = DATA_DIR / "faiss_index.bin"
CHUNK_MAP_PATH = DATA_DIR / "chunk_map.pkl"
INVERTED_INDEX_PATH = DATA_DIR / "inverted_index.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}")
STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "been", "before", "being", "between", "both", "can", "does", "for",
    "from", "had", "has", "have", "how", "into", "its", "more", "not",
    "of", "off", "on", "only", "or", "other", "over", "same", "should",
    "such", "than", "that", "the", "their", "then", "there", "these",
    "they", "this", "those", "through", "to", "under", "uses", "was",
    "were", "what", "when", "where", "which", "while", "who", "why",
    "with", "within", "would", "you", "your",
}


def tokenize(text: str) -> List[str]:
    return [
        match.group(0).lower()
        for match in TOKEN_RE.finditer(text or "")
        if match.group(0).lower() not in STOPWORDS
    ]


def build_inverted_index(chunks: List[Dict]) -> Dict:
    postings: Dict[str, Dict[str, int]] = defaultdict(dict)
    chunk_lengths: Dict[str, int] = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        tokens = tokenize(chunk.get("text", ""))
        counts = Counter(tokens)
        chunk_lengths[cid] = max(len(tokens), 1)
        for token, tf in counts.items():
            postings[token][cid] = tf

    doc_count = len(chunks)
    idf = {
        token: math.log((doc_count + 1) / (len(token_postings) + 1)) + 1.0
        for token, token_postings in postings.items()
    }
    return {
        "postings": dict(postings),
        "idf": idf,
        "chunk_lengths": chunk_lengths,
        "doc_count": doc_count,
    }


def save_lexical_index(index: Dict) -> None:
    with open(INVERTED_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    log.info(f"  Inverted index saved -> {INVERTED_INDEX_PATH}")


def load_lexical_index(path: Path = INVERTED_INDEX_PATH) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def search_index(query: str, index: Dict, top_k: int = 5) -> List[Dict]:
    query_terms = tokenize(query)
    if not query_terms:
        return []

    postings = index.get("postings", {})
    idf = index.get("idf", {})
    chunk_lengths = index.get("chunk_lengths", {})
    if not chunk_lengths:
        log.warning("Lexical index has no chunk lengths; keyword retrieval skipped.")
        return []

    scores: Counter = Counter()
    doc_count = max(index.get("doc_count", 1), 1)
    avgdl = sum(chunk_lengths.values()) / len(chunk_lengths)
    k1 = 1.5
    b = 0.75

    for term in query_terms:
        df = len(postings.get(term, {}))
        term_idf = idf.get(term, math.log((doc_count + 1) / (df + 1)) + 1.0)
        for cid, tf in postings.get(term, {}).items():
            dl = max(chunk_lengths.get(cid, 1), 1)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            scores[cid] += term_idf * ((tf * (k1 + 1)) / max(denom, 1e-9))

    return [
        {"chunk_id": cid, "score": float(score), "rank": rank + 1}
        for rank, (cid, score) in enumerate(scores.most_common(top_k))
    ]


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
    """Build and query the FAISS and BM25 indexes."""
    log.info(f"Embedding {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    int_to_cid = {i: c["chunk_id"] for i, c in enumerate(chunks)}
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    log.info(f"  Embeddings shape: {embeddings.shape}")
    return embeddings, int_to_cid


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build and query the FAISS and BM25 indexes."""
    log.info("Building FAISS IndexFlatIP ...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatIP, int_to_cid: Dict[int, str]) -> None:
    faiss.write_index(index, str(FAISS_PATH))
    log.info(f"  FAISS index saved â†’ {FAISS_PATH}")
    with open(CHUNK_MAP_PATH, "wb") as f:
        pickle.dump(int_to_cid, f)
    log.info(f"  Chunk map saved   â†’ {CHUNK_MAP_PATH}")


def load_index() -> Tuple[faiss.IndexFlatIP, Dict[int, str]]:
    index = faiss.read_index(str(FAISS_PATH))
    with open(CHUNK_MAP_PATH, "rb") as f:
        int_to_cid = pickle.load(f)
    log.info(f"  FAISS index loaded: {index.ntotal} vectors")
    return index, int_to_cid


def query_index(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    int_to_cid: Dict[int, str],
    top_k: int = 5,
    verbose: bool = True,
) -> List[Dict]:
    """Build and query the FAISS and BM25 indexes."""
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")

    similarities, indices = index.search(q_emb, min(top_k, index.ntotal))

    results = []
    for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx == -1:
            continue
        cid = int_to_cid[int(idx)]
        results.append({
            "chunk_id": cid,
            "score":    float(similarity),
            "rank":     rank + 1,
        })
        if verbose:
            log.info(
                f"[FAISS] Rank {rank+1}: chunk_id={cid}  cosine={similarity:.4f}")

    return results



def build() -> Tuple[faiss.IndexFlatIP, Dict[int, str], SentenceTransformer]:
    """Build and query the FAISS and BM25 indexes."""
    log.info("=" * 60)
    log.info("VECTOR ENGINE â€” BUILD PIPELINE START")
    log.info("=" * 60)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    model = load_embedding_model()
    embeddings, int_to_cid = embed_chunks(model, chunks)
    index = build_faiss_index(embeddings)
    save_index(index, int_to_cid)
    save_lexical_index(build_inverted_index(chunks))

    log.info("=" * 60)
    log.info("VECTOR ENGINE BUILD COMPLETE")
    log.info("=" * 60)
    return index, int_to_cid, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraphMind vector index builder")
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

