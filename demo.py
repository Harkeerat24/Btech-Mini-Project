import sys, os, textwrap, pickle
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

OLLAMA_MODEL  = "llama3.2"
LLM_PROVIDER  = "ollama"
TOP_K_VECTOR  = 5
TOP_K_KEYWORD = 5
MAX_HOPS      = 2
LINE          = "─" * 54
WIDE          = "═" * 54

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

print(Fore.MAGENTA + Style.BRIGHT + "  ╔══════════════════════════════════════════════════════╗")
print(Fore.MAGENTA + Style.BRIGHT + "  ║           GraphRAG System  ·  B.Tech Minor           ║")
print(Fore.MAGENTA + Style.BRIGHT + "  ║     spaCy · NetworkX · FAISS · HuggingFace · Ollama  ║")
print(Fore.MAGENTA + Style.BRIGHT + "  ╚══════════════════════════════════════════════════════╝")
print(Fore.CYAN + f"    Index   " + Fore.WHITE + f"{n_nodes} nodes  ·  {n_edges} edges  ·  {n_chunks} chunks")
print(Style.DIM + "    Press Enter on empty input to exit.")
print(Style.DIM + f"  {LINE}")

retriever = GraphRAGRetriever(
    ollama_model=OLLAMA_MODEL,
    llm_provider=LLM_PROVIDER,
    verbose=False,
)
retriever.top_k_vector  = TOP_K_VECTOR
retriever.top_k_keyword = TOP_K_KEYWORD
retriever.max_hops      = MAX_HOPS

try:
    while True:
        query = input("\n" + Fore.CYAN + "  Query  " + Fore.WHITE + "› ").strip()
        if query == "":
            print(Style.DIM + "\n  Goodbye.\n")
            break

        print(Style.DIM + f"  {LINE}")
        result = retriever.query(query, compare_mode=False)
        trace = result["trace"]
        answer = result["graphrag_answer"]

        entities = ", ".join(trace.get("identified_entities", [])[:6]) or "—"
        traversed = trace.get("traversed_nodes", [])
        graph_path = " → ".join(traversed[:5]) + (" …" if len(traversed) > 5 else "")
        n_vec = len(trace.get("vector_chunk_ids", []))
        n_graph = len(trace.get("graph_chunk_ids", []))
        n_final = len(trace.get("final_chunk_ids", []))
        n_hops = MAX_HOPS

        print(Fore.CYAN + "  Entities  " + Fore.WHITE + entities)
        print(Fore.CYAN + "  Graph     " + Fore.WHITE + (graph_path or "— no nodes matched"))
        print(Fore.CYAN + "  BFS       " + Fore.WHITE + f"{n_hops} hops · {len(traversed)} nodes traversed")
        print(Fore.CYAN + "  Retrieval " + Fore.WHITE + f"{n_vec} vector  +  {n_graph} graph  →  {n_final} merged chunks")
        print(Style.DIM + f"  {LINE}")
        print(Fore.CYAN + "  Answer")
        for line in textwrap.wrap(answer, width=52):
            print(Fore.YELLOW + f"  {line}")
        print(Style.DIM + f"  {LINE}")
except KeyboardInterrupt:
    print(Style.DIM + "\n  Interrupted. Goodbye.\n")
