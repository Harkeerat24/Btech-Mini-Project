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
TERM_WIDTH = min(shutil.get_terminal_size().columns, 120)
LINE = "─" * (TERM_WIDTH - 2)
LABEL_WIDTH = 10
VALUE_WIDTH = max(24, TERM_WIDTH - (LABEL_WIDTH + 3))

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

inner = TERM_WIDTH - 4  # 2 for ╔╗, 2 for spaces
title1 = "GraphRAG System  ·  B.Tech Minor"
title2 = "spaCy · NetworkX · FAISS · HuggingFace · Ollama"
line1 = title1.center(inner)
line2 = title2.center(inner)
border = "═" * inner
print(Fore.MAGENTA + Style.BRIGHT + f"  ╔{border}╗")
print(Fore.MAGENTA + Style.BRIGHT + f"  ║{line1}║")
print(Fore.MAGENTA + Style.BRIGHT + f"  ║{line2}║")
print(Fore.MAGENTA + Style.BRIGHT + f"  ╚{border}╝")
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

        entities_str = ", ".join(trace.get("identified_entities", [])[:10])
        entities_wrapped = textwrap.wrap(entities_str, width=VALUE_WIDTH)
        traversed = trace.get("traversed_nodes", [])
        graph_path = " → ".join(
            traversed[:5]) + (" …" if len(traversed) > 5 else "")
        graph_wrapped = textwrap.wrap(graph_path, width=VALUE_WIDTH)
        n_vec = len(trace.get("vector_chunk_ids", []))
        n_graph = len(trace.get("graph_chunk_ids", []))
        n_final = len(trace.get("final_chunk_ids", []))
        n_hops = MAX_HOPS
        bfs_line = f"{n_hops} hops · {len(traversed)} nodes traversed"
        bfs_wrapped = textwrap.wrap(bfs_line, width=VALUE_WIDTH)
        retrieval_line = f"{n_vec} vector  +  {n_graph} graph  →  {n_final} merged chunks"
        retrieval_wrapped = textwrap.wrap(retrieval_line, width=VALUE_WIDTH)

        print(f"  {Fore.CYAN}{'Entities':<{LABEL_WIDTH}}{Style.RESET_ALL} " +
              (entities_wrapped[0] if entities_wrapped else "—"))
        for extra_line in entities_wrapped[1:]:
            print(f"  {'':<{LABEL_WIDTH}} {extra_line}")
        print(f"  {Fore.CYAN}{'Graph':<{LABEL_WIDTH}}{Style.RESET_ALL} " +
              (graph_wrapped[0] if graph_wrapped else "— no nodes matched"))
        for extra_line in graph_wrapped[1:]:
            print(f"  {'':<{LABEL_WIDTH}} {extra_line}")
        print(f"  {Fore.CYAN}{'BFS':<{LABEL_WIDTH}}{Style.RESET_ALL} " +
              (bfs_wrapped[0] if bfs_wrapped else "—"))
        for extra_line in bfs_wrapped[1:]:
            print(f"  {'':<{LABEL_WIDTH}} {extra_line}")
        print(f"  {Fore.CYAN}{'Retrieval':<{LABEL_WIDTH}}{Style.RESET_ALL} " +
              (retrieval_wrapped[0] if retrieval_wrapped else "—"))
        for extra_line in retrieval_wrapped[1:]:
            print(f"  {'':<{LABEL_WIDTH}} {extra_line}")
        print(Style.DIM + f"  {LINE}")
        print(Fore.CYAN + "  Answer")
        for line in textwrap.wrap(answer, width=TERM_WIDTH - 4):
            print(Fore.YELLOW + f"  {line}")
        print(Style.DIM + f"  {LINE}")
except KeyboardInterrupt:
    print(Style.DIM + "\n  Interrupted. Goodbye.\n")
