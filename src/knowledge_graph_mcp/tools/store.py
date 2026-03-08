"""
tools/store.py — store_fact tool
==================================

Stores a single atomic fact into the Neo4j knowledge graph.

Per Anthropic's MCP best practices:
  - Tool description is precise so Claude knows exactly when to call it
  - inputSchema uses JSON Schema with descriptions on every property
  - Returns structured JSON so Claude can parse and present results clearly
  - One fact per call (not batched) — keeps each operation atomic and retryable
"""

import asyncio
from mcp.types import Tool
from knowledge_graph_mcp.db import client, embeddings

# ── Tool definition ───────────────────────────────────────────────────────────
# This object is returned by server.list_tools() so Claude knows:
#   - When to call this tool (description)
#   - What arguments to pass (inputSchema)
#   - Which arguments are required vs optional

DEFINITION = Tool(
    name="store_fact",
    description=(
        "Store a single atomic fact into the persistent knowledge graph under "
        "a profile (project namespace) and category. "
        "Call this ONCE PER FACT — never combine multiple facts into one call. "
        "Automatically skips duplicates (cosine similarity > 0.95). "
        "Automatically links the new fact to semantically related facts in the "
        "same profile+category via [:RELATED_TO] edges. "
        "profile_id is optional — if omitted, the fact is matched to the closest "
        "existing profile by embedding similarity, or a new profile is auto-created "
        "if no profile matches well enough. "
        "Use list_profiles first to confirm the profile name before storing."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": (
                    "A single, self-contained atomic fact. One idea only. "
                    "Use declarative statements with concrete names and values. "
                    "Good: 'JWT access token expiry is 15 minutes'. "
                    "Bad: 'JWT expiry is 15 mins and refresh tokens are in httpOnly cookies' (two facts)."
                ),
            },
            "profile_id": {
                "type": "string",
                "description": (
                    "Profile namespace for this fact. Use snake_case. "
                    "Examples: 'project_1', 'octa', 'personal', 'research'. "
                    "Optional — if omitted, the best matching existing profile is used "
                    "automatically, or a new one is created if no match is found. "
                    "Profile is created automatically if it does not exist yet."
                ),
            },
            "category": {
                "type": "string",
                "description": (
                    "Knowledge category tag. "
                    "Examples: 'auth', 'database', 'api', 'ui', 'infra', "
                    "'business_logic', 'debug', 'integrations', 'meetings', 'research'. "
                    "Create a new tag if none of the examples fit."
                ),
            },
        },
        "required": ["fact", "category"],
    },
)


# ── Tool handler ──────────────────────────────────────────────────────────────

async def handle(args: dict) -> dict:
    """
    Execute the store_fact tool.

    Flow:
        1. Validate inputs
        2. Embed the fact text (run in thread pool — CPU-bound operation)
        3. Resolve profile_id:
               a. If profile_id provided → use it directly
               b. If omitted → call find_best_profile() to match an existing one
               c. If no match → auto-generate a new profile_id from category name
        4. Duplicate check via vector index (scoped to resolved profile)
        5. Write Profile + Category + Fact nodes to Neo4j
        6. Wire [:RELATED_TO] edges to similar facts in same profile+category
        7. Return structured result

    Args:
        args: Dict from Claude containing fact, category, and optional profile_id.

    Returns:
        Dict with status 'saved' or 'skipped', plus metadata including
        profile_resolved (how the profile was determined) and related_linked
        (how many RELATED_TO edges were created).
    """
    fact       = args.get("fact", "").strip()
    profile_id = args.get("profile_id", "").strip()   # optional — may be empty
    category   = args.get("category", "").strip()

    # Input validation
    if not fact:
        return {"status": "error", "message": "'fact' is required and cannot be empty."}
    if not category:
        return {"status": "error", "message": "'category' is required and cannot be empty."}

    # Embed in thread pool — sentence-transformers encode() is synchronous and
    # CPU-bound; running it directly in the async handler would block the event
    # loop and freeze all other tool calls during inference
    loop      = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, embeddings.embed_text, fact)

    # ── Profile resolution ────────────────────────────────────────────────────
    # Three outcomes:
    #   "explicit"   — caller passed a profile_id, use it as-is
    #   "matched"    — no profile_id given, but a close existing profile found
    #   "new"        — no profile_id given and no match; auto-create one
    if profile_id:
        profile_resolved = "explicit"
    else:
        matched = client.find_best_profile(embedding)
        if matched:
            profile_id       = matched
            profile_resolved = "matched"
        else:
            # Auto-generate: "{category}_{date}" e.g. "auth_2024-01-15"
            from datetime import date
            profile_id       = f"{category}_{date.today().isoformat()}"
            profile_resolved = "new"

    # ── Duplicate check ───────────────────────────────────────────────────────
    # Scoped to the resolved profile — different profiles can store the same fact
    duplicate = client.find_duplicate(embedding, profile_id)
    if duplicate:
        return {
            "status":           "skipped",
            "reason":           "duplicate",
            "profile":          profile_id,
            "profile_resolved": profile_resolved,
            "incoming_fact":    fact,
            "existing_match":   duplicate,
            "message":          f"Already stored: \"{duplicate}\"",
        }

    # ── Write to graph: Profile → Category → Fact + RELATED_TO edges ─────────
    client.upsert_profile(profile_id)
    client.upsert_category(profile_id, category)
    fact_id, related_linked = client.write_fact(fact, embedding, profile_id, category)

    return {
        "status":           "saved",
        "fact_id":          fact_id,
        "profile":          profile_id,
        "profile_resolved": profile_resolved,   # explicit | matched | new
        "category":         category,
        "fact":             fact,
        "related_linked":   related_linked,      # how many RELATED_TO edges were created
    }
