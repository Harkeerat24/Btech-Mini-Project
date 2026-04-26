import sys
import os
import shutil
import textwrap
import pickle
from pathlib import Path
from retriever import GraphRAGRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class Fore:
        CYAN = YELLOW = MAGENTA = WHITE = RED = GREEN = ""

    class Style:
        BRIGHT = DIM = RESET_ALL = ""

OLLAMA_MODEL = "llama3.2"
LLM_PROVIDER = "ollama"
TOP_K_VECTOR = 5
TOP_K_KEYWORD = 5
MAX_HOPS = 2
TERM_W = min(shutil.get_terminal_size(fallback=(120, 40)).columns, 140)
LINE = "─" * (TERM_W - 2)

DATA_DIR = Path(__file__).parent / "data"
required = ["chunks.pkl", "graph.pkl", "faiss_index.bin", "chunk_map.pkl"]
missing = [f for f in required if not (DATA_DIR / f).exists()]
if missing:
    print("\n  [!] Index not found. Run ingestion first:")
    print("      python ingestion.py --pdf sample_data/computer_networks.pdf")
    print("      python graph_engine.py")
    print("      python vector_engine.py")
    sys.exit(1)

with open(DATA_DIR / "graph.pkl", "rb") as f:
    G = pickle.load(f)
with open(DATA_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
n_chunks = len(chunks)

_inner = TERM_W - 4
_border = "═" * _inner
_t1 = "GraphRAG System  ·  B.Tech Minor".center(_inner)
_t2 = "spaCy · NetworkX · FAISS · HuggingFace · Ollama".center(_inner)
print(Fore.MAGENTA + Style.BRIGHT + f"  ╔{_border}╗")
print(Fore.MAGENTA + Style.BRIGHT + f"  ║{_t1}║")
print(Fore.MAGENTA + Style.BRIGHT + f"  ║{_t2}║")
print(Fore.MAGENTA + Style.BRIGHT + f"  ╚{_border}╝" + Style.RESET_ALL)
print(Fore.CYAN + f"  Index: " + Fore.WHITE +
      f"{n_nodes} nodes  ·  {n_edges} edges  ·  {n_chunks} chunks")
print(Style.DIM + "  Press Enter on empty input to exit.")
print(Style.DIM + f"  {LINE}")

retriever = GraphRAGRetriever(
    ollama_model=OLLAMA_MODEL,
    llm_provider=LLM_PROVIDER,
    verbose=False,
)
retriever.top_k_vector = TOP_K_VECTOR
retriever.top_k_keyword = TOP_K_KEYWORD
retriever.max_hops = MAX_HOPS

try:
    while True:
        query = input("\n" + Fore.CYAN + "  Query  " +
                      Fore.WHITE + "› ").strip()
        if query == "":
            print(Style.DIM + "\n  Goodbye.\n")
            break

        print(Style.DIM + f"  {LINE}")
        result = retriever.query(query, compare_mode=False)
        trace = result["trace"]
        answer = result["graphrag_answer"]

        entities_list = trace.get("identified_entities", [])
        traversed = trace.get("traversed_nodes", [])
        ent_str = ", ".join(entities_list[:10]) or "—"
        ent_lines = textwrap.wrap(ent_str, width=TERM_W - 18) or ["—"]
        gpath = " → ".join(traversed[:8]) + \
            (" …" if len(traversed) > 8 else "")
        gpath_lines = textwrap.wrap(
            gpath, width=TERM_W - 18) or ["— no nodes matched"]
        n_vec = len(trace.get("vector_chunk_ids", []))
        n_graph = len(trace.get("graph_chunk_ids", []))
        n_final = len(trace.get("final_chunk_ids", []))
        n_hops = MAX_HOPS
        bfs_line = f"{n_hops} hops · {len(traversed)} nodes traversed"
        bfs_wrapped = textwrap.wrap(bfs_line, width=TERM_W - 18)
        retrieval_line = f"{n_vec} vector  +  {n_graph} graph  →  {n_final} merged chunks"
        retrieval_wrapped = textwrap.wrap(retrieval_line, width=TERM_W - 18)

        print(f"  {Fore.CYAN}{'Entities':<12}{Style.RESET_ALL} {ent_lines[0]}")
        for extra in ent_lines[1:]:
            print(f"  {'':12}  {extra}")
        print(f"  {Fore.CYAN}{'Graph':<12}{Style.RESET_ALL} {gpath_lines[0]}")
        for extra in gpath_lines[1:]:
            print(f"  {'':12}  {extra}")
        print(f"  {Fore.CYAN}{'BFS':<12}{Style.RESET_ALL} " +
              (bfs_wrapped[0] if bfs_wrapped else "—"))
        for extra_line in bfs_wrapped[1:]:
            print(f"  {'':12}  {extra_line}")
        print(f"  {Fore.CYAN}{'Retrieval':<12}{Style.RESET_ALL} " +
              (retrieval_wrapped[0] if retrieval_wrapped else "—"))
        for extra_line in retrieval_wrapped[1:]:
            print(f"  {'':12}  {extra_line}")
        print(Style.DIM + f"  {LINE}")
        print(Fore.CYAN + "  Answer")
        for line in textwrap.wrap(answer, width=TERM_W - 14):
            print(Fore.YELLOW + f"  {line}")
        print(Style.DIM + f"  {LINE}")
except KeyboardInterrupt:
    print(Style.DIM + "\n  Interrupted. Goodbye.\n")
