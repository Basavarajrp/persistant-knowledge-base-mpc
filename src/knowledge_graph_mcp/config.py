"""
config.py — Centralised configuration
======================================
All environment variables and tunable constants live here.
Imported by db/client.py and db/embeddings.py — never scattered across modules.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/pkg/ → root)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_root, ".env"))

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80MB, CPU-friendly, great semantic quality.
# Cached at ~/.cache/huggingface/ after first download.
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ── Thresholds ────────────────────────────────────────────────────────────────
# Cosine similarity above which a new fact is treated as a duplicate (0–1).
# 0.95 = strict. Lower to 0.90 for more aggressive dedup.
DUPLICATE_THRESHOLD = 0.95

# Cosine similarity above which two facts in the same profile+category are
# linked with a [:RELATED_TO] edge on write. Must be below DUPLICATE_THRESHOLD
# so duplicates are caught before this step runs.
# 0.70 = related but distinct ideas. Raise to 0.80 for tighter clusters.
RELATED_TO_THRESHOLD = 0.70

# Cosine similarity above which an incoming fact (with no explicit profile_id)
# is assigned to an existing profile instead of creating a new one.
# 0.75 = loose match. Raise to 0.85 if you want stricter profile assignment.
PROFILE_MATCH_THRESHOLD = 0.75

# Max [:RELATED_TO] neighbours returned per matched node during retrieval.
MAX_RELATED_FACTS = 3
