import argparse
import logging
import pickle
import re
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

"""Ingest local knowledge-base files into chunks and graph facts."""

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingestion")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
TRIPLETS_PATH = DATA_DIR / "triplets.pkl"

_SPACY_PRIORITY = [
    "en_core_web_trf",
    "en_core_web_lg",
    "en_core_web_md",
    "en_core_web_sm",
]

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _load_spacy():
    for name in _SPACY_PRIORITY:
        try:
            model = spacy.load(name)
            log.info("spaCy model loaded: %s", name)
            return model
        except OSError:
            continue
    log.warning(
        "No spaCy model found. NER and triplet extraction are reduced. "
        "Install one with: python -m spacy download en_core_web_sm"
    )
    return spacy.blank("en")


nlp = _load_spacy()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _with_source(path: Path, sections: List[Dict]) -> List[Dict]:
    return [
        {"source_ref": f"{path.name}:{section['source_ref']}", "text": section["text"]}
        for section in sections
    ]


def _read_pdf(path: Path) -> List[Dict]:
    reader = PdfReader(str(path))
    sections = []
    for index, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        if text:
            sections.append({"source_ref": f"page:{index + 1}", "text": text})
    return sections


def _read_plaintext(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = []
    current = []
    section = 1
    for line in text.splitlines():
        stripped = line.strip()
        starts_section = path.suffix.lower() == ".md" and stripped.startswith("#")
        if starts_section and current:
            block_text = _clean_text(" ".join(current))
            if block_text:
                blocks.append({"source_ref": f"section:{section}", "text": block_text})
                section += 1
            current = [stripped]
        else:
            current.append(stripped)

    block_text = _clean_text(" ".join(current))
    if block_text:
        blocks.append({"source_ref": f"section:{section}", "text": block_text})
    return blocks


def read_source(path: str) -> List[Dict]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _with_source(source, _read_pdf(source))
    if suffix in {".txt", ".md"}:
        return _with_source(source, _read_plaintext(source))
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported source format '{suffix}'. Supported formats: {supported}")


def discover_sources(path: str) -> List[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path not found: {source}")

    files = sorted(
        item for item in source.iterdir()
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No knowledge-base files found in {source}. Add one of: {supported}"
        )
    return files


def read_sources(path: str) -> List[Dict]:
    sections = []
    for source in discover_sources(path):
        sections.extend(read_source(str(source)))
    return sections


def _sentence_units(text: str) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_RE.split(text or "") if s.strip()]
    return sentences or ([text.strip()] if text and text.strip() else [])


def _window_sentences(sentences: List[str], target_chars: int, overlap_chars: int) -> List[str]:
    chunks = []
    start = 0
    target_chars = max(target_chars, 128)
    overlap_chars = max(0, min(overlap_chars, target_chars // 2))

    while start < len(sentences):
        end = start
        current_len = 0
        while end < len(sentences):
            next_len = len(sentences[end]) + (1 if current_len else 0)
            if current_len and current_len + next_len > target_chars:
                break
            current_len += next_len
            end += 1

        if end == start:
            end += 1
        window = " ".join(sentences[start:end]).strip()
        if window:
            chunks.append(window)
        if end >= len(sentences):
            break

        rewind = 0
        kept_chars = 0
        for idx in range(end - 1, start - 1, -1):
            kept_chars += len(sentences[idx]) + (1 if kept_chars else 0)
            if kept_chars > overlap_chars:
                break
            rewind += 1
        start = max(start + 1, end - max(rewind, 1))

    return chunks


def chunk_sections(sections: List[Dict], chunk_size=512, chunk_overlap=64) -> List[Dict]:
    chunks = []
    counter = 0
    for section in sections:
        for raw in _window_sentences(_sentence_units(section["text"]), chunk_size, chunk_overlap):
            chunks.append({
                "chunk_id": f"chunk_{counter:04d}",
                "text": raw.strip(),
                "source_ref": section["source_ref"],
            })
            counter += 1
    return chunks


def _normalize(text: str) -> str:
    return text.strip().lower()


def run_ner(nlp_model, chunks: List[Dict]) -> List[Dict]:
    texts = [c["text"] for c in chunks]
    docs = list(nlp_model.pipe(texts, batch_size=8))
    enriched = []
    for chunk, doc in zip(chunks, docs):
        seen = set()
        ents = []
        for ent in doc.ents:
            key = (_normalize(ent.text), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            ents.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        enriched_chunk = dict(chunk)
        enriched_chunk["entities"] = ents
        enriched_chunk["entity_texts"] = [ent["text"] for ent in ents]
        enriched_chunk["doc"] = doc
        enriched.append(enriched_chunk)
    return enriched


def _get_full_span(token) -> str:
    parts = sorted(
        [t for t in token.subtree if t.dep_ in {"compound", "amod", "nummod"} or t == token],
        key=lambda t: t.i,
    )
    return " ".join(t.text for t in parts)


def extract_triplets_from_doc(doc, chunk_id: str) -> List[Dict]:
    triplets = []
    if not doc.has_annotation("DEP"):
        return triplets
    for sent in doc.sents:
        root = next(
            (token for token in sent if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}),
            None,
        )
        if root is None:
            continue
        subjects = [
            token for token in sent
            if token.dep_ in {"nsubj", "nsubjpass", "csubj"} and token.head == root
        ]
        objects = [
            token for token in sent
            if token.dep_ in {"dobj", "pobj", "attr", "oprd", "dative"}
            and (token.head == root or token.head.head == root)
        ]
        for subject in subjects:
            for obj in objects:
                subj_text = _get_full_span(subject)
                obj_text = _get_full_span(obj)
                if not subj_text or not obj_text or subj_text.lower() == obj_text.lower():
                    continue
                triplets.append({
                    "subject": subj_text,
                    "predicate": root.lemma_,
                    "object": obj_text,
                    "chunk_id": chunk_id,
                })
    return triplets


def extract_all_triplets(enriched_chunks: List[Dict]) -> List[Dict]:
    all_triplets, seen = [], set()
    for chunk in enriched_chunks:
        doc = chunk.get("doc")
        if doc is None:
            continue
        for triplet in extract_triplets_from_doc(doc, chunk["chunk_id"]):
            key = (
                _normalize(triplet["subject"]),
                _normalize(triplet["predicate"]),
                _normalize(triplet["object"]),
            )
            if key not in seen:
                seen.add(key)
                all_triplets.append(triplet)
    return all_triplets


def ingest(source_path: str, chunk_size=512, chunk_overlap=64) -> Tuple[List, List]:
    sections = read_sources(source_path)
    chunks = chunk_sections(sections, chunk_size, chunk_overlap)
    enriched = run_ner(nlp, chunks)
    triplets = extract_all_triplets(enriched)

    clean_chunks = [{k: v for k, v in chunk.items() if k != "doc"} for chunk in enriched]
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(clean_chunks, f)
    with open(TRIPLETS_PATH, "wb") as f:
        pickle.dump(triplets, f)

    log.warning(
        "Ingestion complete | chunks=%s | triplets=%s | source=%s",
        len(clean_chunks),
        len(triplets),
        source_path,
    )
    return enriched, triplets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphMind document ingestion")
    parser.add_argument(
        "--input",
        default=str(KNOWLEDGE_BASE_DIR),
        help="Path to a .pdf/.txt/.md file or a folder containing those files",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    args = parser.parse_args()

    try:
        chunks, triplets = ingest(args.input, args.chunk_size, args.chunk_overlap)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        raise SystemExit(1)
    term_width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    preview_width = max(30, term_width - 24)

    print("\n--- Sample Chunks (first 3) ---")
    for chunk in chunks[:3]:
        preview = textwrap.shorten(chunk["text"], width=preview_width, placeholder="...")
        print(f"  [{chunk['chunk_id']}] source={chunk['source_ref']} | {preview}")

    print("\n--- Sample Triplets (first 8) ---")
    for triplet in triplets[:8]:
        line = (
            f"({triplet['subject']}) --[{triplet['predicate']}]--> "
            f"({triplet['object']}) [src: {triplet['chunk_id']}]"
        )
        for idx, part in enumerate(textwrap.wrap(line, width=max(30, term_width - 4))):
            print(f"  {part}" if idx == 0 else f"    {part}")

