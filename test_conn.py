import os
import sys

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

URI = os.getenv("NEO4J_URI", "").strip()
USER = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")).strip()
PASSWORD = os.getenv("NEO4J_PASSWORD", os.getenv("NEO4J_PASS", "")).strip()


def test_connection():
    if not URI or not PASSWORD:
        print("[!] Set NEO4J_URI and NEO4J_PASSWORD before running this script.")
        sys.exit(1)

    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 'Connection Successful!' as message")
            print(f"[ok] {result.single()['message']}")
        driver.close()
    except Exception as e:
        print(f"[!] FAILED: {e}")


if __name__ == "__main__":
    test_connection()
