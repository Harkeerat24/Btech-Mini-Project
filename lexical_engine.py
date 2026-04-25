"""
lexical_engine.py - inverted-index retrieval for GraphRAG.

Builds a lightweight token -> chunk posting list so query-time retrieval can
combine exact keyword matches with FAISS vectors and graph traversal.
"""

import argparse
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lexical_engine")

DATA_DIR = Path(__file__).parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
INVERTED_INDEX_PATH = DATA_DIR / "inverted_index.pkl"

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


def save_index(index: Dict, path: Path = INVERTED_INDEX_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)
    log.info(f"Inverted index saved -> {path}")


def load_index(path: Path = INVERTED_INDEX_PATH) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def search_index(query: str, index: Dict, top_k: int = 5) -> List[Dict]:
    query_terms = tokenize(query)
    if not query_terms:
        return []

    scores = Counter()
    postings = index.get("postings", {})
    idf = index.get("idf", {})
    chunk_lengths = index.get("chunk_lengths", {})

    for term in query_terms:
        for cid, tf in postings.get(term, {}).items():
            norm_tf = tf / max(chunk_lengths.get(cid, 1), 1)
            scores[cid] += norm_tf * idf.get(term, 1.0)

    ranked = scores.most_common(top_k)
    return [
        {"chunk_id": cid, "score": float(score), "rank": rank + 1}
        for rank, (cid, score) in enumerate(ranked)
    ]


def build() -> Dict:
    log.info("=" * 60)
    log.info("LEXICAL ENGINE - BUILD PIPELINE START")
    log.info("=" * 60)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    index = build_inverted_index(chunks)
    save_index(index)

    log.info(
        f"LEXICAL ENGINE BUILD COMPLETE | chunks={index['doc_count']} "
        f"| terms={len(index['postings'])}"
    )
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG inverted-index builder")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    lexical_index = build()
    if args.query:
        print(f"\n--- Lexical Query: '{args.query}' ---")
        for result in search_index(args.query, lexical_index, top_k=args.top_k):
            print(
                f"  Rank {result['rank']}: {result['chunk_id']} "
                f"(score={result['score']:.4f})"
            )
