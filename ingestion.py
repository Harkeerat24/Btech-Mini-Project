"""
ingestion.py — GraphRAG System
================================
Pipeline: PDF -> Text Chunks -> NER (spaCy) -> Triplets (dep parse)

Usage:
  python ingestion.py --pdf path/to/document.pdf
  python ingestion.py --pdf path/to/document.pdf --chunk_size 512 --chunk_overlap 64
"""

import textwrap
import shutil
from pypdf import PdfReader
import spacy
from typing import List, Dict, Tuple
from pathlib import Path
import logging
import argparse
import pickle
import re
import os
from dotenv import load_dotenv
load_dotenv()


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ingestion")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
TRIPLETS_PATH = DATA_DIR / "triplets.pkl"

SPACY_MODEL_CANDIDATES = (
    "en_core_web_trf", "en_core_web_md", "en_core_web_sm", "en_core_web_lg")

CUSTOM_ENTITY_PATTERNS = [
    {"label": "PROTOCOL",  "pattern": "OSPF"},
    {"label": "PROTOCOL",  "pattern": "BGP"},
    {"label": "PROTOCOL",  "pattern": "TCP"},
    {"label": "PROTOCOL",  "pattern": "UDP"},
    {"label": "PROTOCOL",  "pattern": "HTTP"},
    {"label": "PROTOCOL",  "pattern": "HTTPS"},
    {"label": "PROTOCOL",  "pattern": "DNS"},
    {"label": "PROTOCOL",  "pattern": "DHCP"},
    {"label": "PROTOCOL",  "pattern": "MPLS"},
    {"label": "PROTOCOL",  "pattern": "RIP"},
    {"label": "PROTOCOL",  "pattern": "EIGRP"},
    {"label": "PROTOCOL",  "pattern": "IS-IS"},
    {"label": "PROTOCOL",  "pattern": "ARP"},
    {"label": "PROTOCOL",  "pattern": "ICMP"},
    {"label": "PROTOCOL",  "pattern": "IPSec"},
    {"label": "PROTOCOL",  "pattern": "OpenFlow"},
    {"label": "ALGORITHM", "pattern": "Dijkstra"},
    {"label": "ALGORITHM", "pattern": [
        {"LOWER": "dijkstra"}, {"LOWER": "algorithm", "OP": "?"}]},
    {"label": "ALGORITHM", "pattern": "Bellman-Ford"},
    {"label": "ALGORITHM", "pattern": "DUAL"},
    {"label": "ALGORITHM", "pattern": "PageRank"},
    {"label": "CONCEPT",   "pattern": "LSA"},
    {"label": "CONCEPT",   "pattern": "SPF"},
    {"label": "CONCEPT",   "pattern": "NLRI"},
    {"label": "CONCEPT",   "pattern": "VLSM"},
    {"label": "CONCEPT",   "pattern": "CIDR"},
    {"label": "CONCEPT",   "pattern": [{"LOWER": "link"}, {"LOWER": "state"}]},
    {"label": "CONCEPT",   "pattern": [
        {"LOWER": "routing"}, {"LOWER": "table"}]},
    {"label": "CONCEPT",   "pattern": [
        {"LOWER": "hello"}, {"LOWER": "packet"}]},
    {"label": "CONCEPT",   "pattern": [
        {"LOWER": "dead"}, {"LOWER": "interval"}]},
    {"label": "CONCEPT",   "pattern": [
        {"LOWER": "spanning"}, {"LOWER": "tree"}]},
    {"label": "CONCEPT",   "pattern": [
        {"LOWER": "autonomous"}, {"LOWER": "system"}]},
    {"label": "CONCEPT",   "pattern": "SDN"},
    {"label": "CONCEPT",   "pattern": "NFV"},
    {"label": "CONCEPT",   "pattern": "VPN"},
    {"label": "CONCEPT",   "pattern": "DNSSEC"},
]


def load_spacy_model():
    for model_name in SPACY_MODEL_CANDIDATES:
        try:
            nlp = spacy.load(model_name)
            log.info(f"Loaded spaCy model: {model_name}")
            break
        except OSError:
            log.debug(f"spaCy model unavailable: {model_name}")
    else:
        nlp = None

    if nlp is None:
        log.warning(
            "No spaCy model found. Run: python -m spacy download en_core_web_md")
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    ruler_kwargs = {"name": "domain_ruler"}
    if "ner" in nlp.pipe_names:
        ruler_kwargs["before"] = "ner"
    ruler = nlp.add_pipe("entity_ruler", **ruler_kwargs)
    ruler.add_patterns(CUSTOM_ENTITY_PATTERNS)
    log.info(f"  Pipeline: {nlp.pipe_names}")
    log.info(f"  Custom patterns: {len(CUSTOM_ENTITY_PATTERNS)}")
    return nlp


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    log.info(f"Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    log.info(f"  Extracted {len(pages)} pages")
    return pages


def chunk_pages(pages: List[Dict], chunk_size=512, chunk_overlap=64) -> List[Dict]:
    log.info(f"Chunking: size={chunk_size}, overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    counter = 0
    for p in pages:
        for raw in splitter.split_text(p["text"]):
            chunks.append({"chunk_id": f"chunk_{counter:04d}",
                          "text": raw.strip(), "page": p["page"]})
            counter += 1
    log.info(f"  Total chunks: {len(chunks)}")
    return chunks


def run_ner(nlp, chunks: List[Dict]) -> List[Dict]:
    log.info(f"Running NER on {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    docs = list(nlp.pipe(texts, batch_size=8))
    total_ents = 0
    enriched = []
    for chunk, doc in zip(chunks, docs):
        seen = set()
        ents = []
        for e in doc.ents:
            key = (_normalize(e.text), e.label_)
            if key in seen:
                continue
            seen.add(key)
            ents.append({
                "text": e.text.strip(),
                "label": e.label_,
                "start": e.start_char,
                "end": e.end_char,
            })
        c = dict(chunk)
        c["entities"] = ents
        c["entity_texts"] = [e["text"] for e in ents]
        c["doc"] = doc
        enriched.append(c)
        total_ents += len(ents)
    log.info(f"  Total entities found: {total_ents}")
    return enriched


def _normalize(text: str) -> str:
    return text.strip().lower()


def _get_full_span(token) -> str:
    parts = sorted(
        [t for t in token.subtree if t.dep_ in {
            "compound", "amod", "nummod"} or t == token],
        key=lambda t: t.i,
    )
    return " ".join(t.text for t in parts)


def extract_triplets_from_doc(doc, chunk_id: str) -> List[Dict]:
    triplets = []
    if not doc.has_annotation("DEP"):
        return triplets
    for sent in doc.sents:
        root = next((t for t in sent if t.dep_ ==
                    "ROOT" and t.pos_ in {"VERB", "AUX"}), None)
        if root is None:
            continue
        subjects = [t for t in sent if t.dep_ in {
            "nsubj", "nsubjpass", "csubj"} and t.head == root]
        objects = [t for t in sent if t.dep_ in {
            "dobj", "pobj", "attr", "oprd", "dative"} and (t.head == root or t.head.head == root)]
        for subj in subjects:
            for obj in objects:
                s = _get_full_span(subj)
                o = _get_full_span(obj)
                if not s or not o or s.lower() == o.lower():
                    continue
                triplets.append(
                    {"subject": s, "predicate": root.lemma_, "object": o, "chunk_id": chunk_id})
    return triplets


def extract_all_triplets(enriched_chunks: List[Dict]) -> List[Dict]:
    log.info("Extracting triplets via dependency parsing ...")
    all_triplets, seen = [], set()
    for chunk in enriched_chunks:
        doc = chunk.get("doc")
        if doc is None:
            continue
        for t in extract_triplets_from_doc(doc, chunk["chunk_id"]):
            key = (_normalize(t["subject"]), _normalize(
                t["predicate"]), _normalize(t["object"]))
            if key not in seen:
                seen.add(key)
                all_triplets.append(t)
    log.info(f"  Unique triplets extracted: {len(all_triplets)}")
    return all_triplets


def ingest(pdf_path: str, chunk_size=512, chunk_overlap=64) -> Tuple[List, List]:
    log.info("=" * 60)
    log.info("GRAPHRAG INGESTION PIPELINE — START")
    log.info("=" * 60)

    nlp = load_spacy_model()
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    enriched = run_ner(nlp, chunks)
    triplets = extract_all_triplets(enriched)

    # Save chunks (without spaCy Doc objects)
    clean = [{k: v for k, v in c.items() if k != "doc"} for c in enriched]
    with open(CHUNKS_PATH,   "wb") as f:
        pickle.dump(clean,    f)
    with open(TRIPLETS_PATH, "wb") as f:
        pickle.dump(triplets, f)
    log.info(f"  Chunks saved   -> {CHUNKS_PATH}")
    log.info(f"  Triplets saved -> {TRIPLETS_PATH}")

    log.info("=" * 60)
    log.info(
        f"INGESTION COMPLETE  |  Chunks: {len(enriched)}  |  Triplets: {len(triplets)}")
    log.info("=" * 60)
    return enriched, triplets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG PDF Ingestion")
    parser.add_argument("--pdf",           required=True)
    parser.add_argument("--chunk_size",    type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        log.error(f"PDF not found: {args.pdf}")
        raise SystemExit(1)

    chunks, triplets = ingest(args.pdf, args.chunk_size, args.chunk_overlap)
    term_width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    preview_width = max(30, term_width - 24)

    print("\n--- Sample Chunks (first 3) ---")
    for c in chunks[:3]:
        preview = textwrap.shorten(
            c["text"], width=preview_width, placeholder="...")
        print(f"  [{c['chunk_id']}] page={c['page']} | {preview}")

    print("\n--- Sample Triplets (first 8) ---")
    for t in triplets[:8]:
        line = f"({t['subject']}) --[{t['predicate']}]--> ({t['object']}) [src: {t['chunk_id']}]"
        wrapped = textwrap.wrap(line, width=max(30, term_width - 4))
        for i, part in enumerate(wrapped):
            print(f"  {part}" if i == 0 else f"    {part}")
