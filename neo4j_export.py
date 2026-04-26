# ─────────────────────────────────────────────────────
# neo4j_export.py — One-time Neo4j export for PPT screenshot
#
# SETUP (do this once):
#   1. In Neo4j Aura, create/start your DB instance
#   2. Set environment variables:
#        set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687
#        set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:443
#        set NEO4J_USERNAME=9b183348
#        set NEO4J_DATABASE=9b183348
#        set NEO4J_PASSWORD=<your-password>
#   3. Run: python neo4j_export.py
#   4. Open Neo4j Browser from Aura Console
#   5. Run this Cypher query to visualise:
#        MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150
#   6. Arrange the graph, take a full-resolution screenshot
#   7. Put the screenshot on your PPT slide (Knowledge Graph slide)
#   You do NOT need Neo4j running during the presentation.
# ─────────────────────────────────────────────────────

import os
import pickle
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
except ImportError:
    print("[!] Install the Neo4j driver: pip install neo4j")
    sys.exit(1)

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _neo4j_uri_candidates(uri: str) -> list[str]:
    """Prefer Bolt over Neo4j routing, with Aura port fallbacks."""
    if not uri:
        return []

    parsed = urlparse(uri)
    host = parsed.hostname
    if not host:
        return [uri]

    scheme = "bolt+s"
    ports: list[int] = []
    if parsed.port is not None:
        ports.append(parsed.port)
    for port in (7687, 443):
        if port not in ports:
            ports.append(port)

    return [f"{scheme}://{host}:{port}" for port in ports]


def _is_auth_error(error: Exception) -> bool:
    text = str(error)
    return "Unauthorized" in text or "AuthError" in text or "authentication failure" in text.lower()


NEO4J_URI = _env("NEO4J_URI")
NEO4J_USER = _env("NEO4J_USER") or _env("NEO4J_USERNAME") or "neo4j"
NEO4J_DATABASE = _env("NEO4J_DATABASE") or "neo4j"
NEO4J_PASSWORD = _env("NEO4J_PASSWORD") or _env("NEO4J_PASS")
URI_CANDIDATES = _neo4j_uri_candidates(NEO4J_URI)

if not NEO4J_URI:
    print("[!] Missing NEO4J_URI environment variable.")
    print("    Example: set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687")
    print("    Fallback: set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:443")
    sys.exit(1)

if not NEO4J_PASSWORD:
    print("[!] Missing NEO4J_PASSWORD environment variable.")
    print("    Example (Aura):")
    print("    set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687")
    print("    set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:443")
    print("    set NEO4J_USERNAME=9b183348")
    print("    set NEO4J_DATABASE=9b183348")
    print("    set NEO4J_PASSWORD=<your-password>")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / "data"
with open(DATA_DIR / "graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"  Connecting to Neo4j at {NEO4J_URI} ...")
driver = None
last_error = None
active_user = NEO4J_USER
user_candidates = [NEO4J_USER]
if NEO4J_USER != "neo4j":
    user_candidates.append("neo4j")

for uri in URI_CANDIDATES:
    for user in user_candidates:
        trial_driver = None
        try:
            print(f"  Trying URI: {uri}")
            trial_driver = GraphDatabase.driver(
                uri, auth=(user, NEO4J_PASSWORD))
            trial_driver.verify_connectivity()
            driver = trial_driver
            active_user = user
            NEO4J_URI = uri
            break
        except Exception as e:
            last_error = e
            try:
                trial_driver.close()
            except Exception:
                pass
            if _is_auth_error(e):
                driver = None
                break
    if driver is not None:
        break
    if last_error is not None and _is_auth_error(last_error):
        break

if driver is None:
    print(f"[!] Neo4j connection failed: {last_error}")
    if "Unauthorized" in str(last_error):
        print("    Authentication failed. Reset/copy your Aura password and re-set NEO4J_PASSWORD.")
        print("    Aura Console -> Database -> ... -> Reset password")
    print("    Check NEO4J_URI / NEO4J_USER / NEO4J_USERNAME / NEO4J_DATABASE / NEO4J_PASSWORD and Aura instance status.")
    sys.exit(1)

if active_user != NEO4J_USER:
    print(
        f"  [!] NEO4J_USER/NEO4J_USERNAME='{NEO4J_USER}' failed; using '{active_user}'")

db_candidates = [NEO4J_DATABASE]
if NEO4J_DATABASE != "neo4j":
    db_candidates.append("neo4j")
db_candidates.append(None)

session = None
active_database = NEO4J_DATABASE
last_db_error = None
for db_name in db_candidates:
    try:
        if db_name is None:
            candidate_session = driver.session()
        else:
            candidate_session = driver.session(database=db_name)
        candidate_session.run("RETURN 1").consume()
        session = candidate_session
        active_database = db_name
        break
    except Exception as e:
        last_db_error = e
        try:
            candidate_session.close()
        except Exception:
            pass

if session is None:
    print(f"[!] Neo4j database error: {last_db_error}")
    print("    Check NEO4J_DATABASE. If unsure, leave it unset and let Neo4j use the default database.")
    sys.exit(1)

if active_database is None:
    print("  [!] Using Neo4j default database from user home DB setting")
elif active_database != NEO4J_DATABASE:
    print(
        f"  [!] NEO4J_DATABASE='{NEO4J_DATABASE}' not found; using '{active_database}'")

with session:
    session.run("MATCH (n) DETACH DELETE n")
    print("  Cleared existing graph.")

    nodes_data = [
        {"id": nk,
         "label": d.get("label", nk),
         "etype": d.get("entity_type", "DEFAULT"),
         "btw": round(d.get("betweenness_centrality", 0.0), 4),
         "deg": round(d.get("degree_centrality", 0.0), 4)}
        for nk, d in G.nodes(data=True)
    ]
    session.run("""
        UNWIND $rows AS row
        CREATE (:Entity {
          id: row.id, label: row.label,
          entity_type: row.etype,
          betweenness: row.btw,
          degree: row.deg
        })
    """, rows=nodes_data)
    print(f"  Exported {len(nodes_data)} nodes.")

    edges_data = [
        {"src": s, "dst": t,
         "pred": d.get("predicate", "relates_to"),
         "w": d.get("weight", 1)}
        for s, t, d in G.edges(data=True)
        if s in G.nodes() and t in G.nodes()
    ]
    session.run("""
        UNWIND $rows AS row
        MATCH (a:Entity {id: row.src}), (b:Entity {id: row.dst})
        CREATE (a)-[:RELATES_TO {predicate: row.pred, weight: row.w}]->(b)
    """, rows=edges_data)
    print(f"  Exported {len(edges_data)} edges.")

driver.close()

print(f"""──────────────────────────────────────────────────────
  Export complete.
    Open Neo4j Browser (Aura) from your Aura Console.
    Run this query to see your full knowledge graph:

    MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150

  Arrange the layout → take screenshot → use in PPT.
──────────────────────────────────────────────────────""")
