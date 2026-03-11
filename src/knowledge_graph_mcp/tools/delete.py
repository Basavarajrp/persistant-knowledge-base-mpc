"""
tools/delete.py — list_facts, preview_delete, delete_nodes tools
=================================================================

Three tools for safe, user-confirmed deletion in the knowledge graph.

Graph hierarchy targeted here: Profile → Category → Fact

Intended call order (enforced by the /knowledge-graph-delete prompt):
    1. list_facts       — browse individual fact IDs (only needed for selective deletes)
    2. preview_delete   — dry-run: shows full impact + cascade + cross-edges
    3. delete_nodes     — executes ONLY after user confirms the preview

Why three tools instead of one:
    - preview_delete is always safe to call (read-only) — Claude can call it proactively
      whenever it detects a contradicting or outdated fact.
    - delete_nodes is destructive — separating it forces an explicit second call,
      making it obvious in the tool trace that deletion was user-confirmed.
    - list_facts is optional browse — skip it when deleting a whole category/profile.

Cascade behaviour (shown in preview, executed in delete_nodes):
    Deleting facts → empty Category auto-removed → empty Profile auto-removed.
    Users see this in the preview template before confirming.
"""

from mcp.types import Tool
from knowledge_graph_mcp.db import client


# ─────────────────────────────────────────────────────────────────────────────
# list_facts
# ─────────────────────────────────────────────────────────────────────────────

LIST_FACTS_DEFINITION = Tool(
    name="list_facts",
    description=(
        "List all facts stored in a specific profile+category with their IDs. "
        "Use this when the user wants to selectively delete individual facts "
        "(not a whole category). Shows each fact's index, ID, and text so the "
        "user can pick which ones to target. "
        "For deleting a whole category or whole profile, skip this and call "
        "preview_delete directly with scope='category' or scope='profile'."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "profile_id": {
                "type":        "string",
                "description": "Profile the category belongs to.",
            },
            "category": {
                "type":        "string",
                "description": "Category to list facts from.",
            },
        },
        "required": ["profile_id", "category"],
    },
)


async def handle_list_facts(args: dict) -> dict:
    """
    Return all facts in a category as a numbered list with IDs.

    Facts are numbered so the user can reference them by index in the
    delete confirmation (e.g. "delete 1, 3, 5").

    Args:
        args: Dict with profile_id and category.

    Returns:
        Numbered list of {index, id, text} or empty status if none found.
    """
    profile_id = args.get("profile_id", "").strip()
    category   = args.get("category", "").strip()

    if not profile_id:
        return {"status": "error", "message": "'profile_id' is required."}
    if not category:
        return {"status": "error", "message": "'category' is required."}

    facts = client.list_facts(profile_id, category)

    if not facts:
        return {
            "status":   "empty",
            "profile":  profile_id,
            "category": category,
            "message":  f"No facts found in '{profile_id}/{category}'.",
            "facts":    [],
        }

    # Index starts at 1 so user can say "delete 1, 3" naturally
    numbered = [
        {"index": i + 1, "id": f["id"], "text": f["text"], "created_at": f["created_at"]}
        for i, f in enumerate(facts)
    ]

    return {
        "status":   "ok",
        "profile":  profile_id,
        "category": category,
        "count":    len(numbered),
        "facts":    numbered,
        "hint": (
            "To delete specific facts: call preview_delete with "
            "scope='facts' and the fact_ids you want removed. "
            "To delete all facts in this category: call preview_delete with scope='category'."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# preview_delete
# ─────────────────────────────────────────────────────────────────────────────

PREVIEW_DELETE_DEFINITION = Tool(
    name="preview_delete",
    description=(
        "Show the full impact of a deletion WITHOUT making any changes. "
        "Always call this BEFORE delete_nodes — never delete without previewing first. "
        "Returns a formatted text block showing: facts to be deleted, "
        "cross-category RELATED_TO edges to be unlinked, and cascade cleanup "
        "(empty categories and profiles that will be auto-removed). "
        "Present the preview_text field to the user exactly as returned, then "
        "wait for explicit confirmation before calling delete_nodes. "
        "Scopes: 'profile' = everything in a profile, "
        "'category' = all facts in one category, "
        "'facts' = specific fact UUIDs (use list_facts to get them)."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["profile", "category", "facts"],
                "description": (
                    "'profile'  — entire profile + all its categories + all facts. "
                    "'category' — all facts inside one category. "
                    "'facts'    — specific facts by UUID (from list_facts)."
                ),
            },
            "profile_id": {
                "type":        "string",
                "description": "Profile to target. Required for all scopes.",
            },
            "category": {
                "type":        "string",
                "description": "Category name. Required when scope='category'.",
            },
            "fact_ids": {
                "type":        "array",
                "items":       {"type": "string"},
                "description": "Fact UUIDs to preview-delete. Required when scope='facts'.",
            },
        },
        "required": ["scope", "profile_id"],
    },
)


async def handle_preview_delete(args: dict) -> dict:
    """
    Compute and return the full deletion impact — no graph changes made.

    Builds a formatted preview_text block showing three sections:
        1. Facts to be deleted (capped at 20 in display, full count always shown)
        2. Cross-category RELATED_TO edges to be unlinked
        3. Cascade auto-cleanup: empty categories and profile (if any)

    The /knowledge-graph-delete prompt presents this preview_text to the user
    and waits for confirmation before delete_nodes is called.

    Args:
        args: Dict with scope, profile_id, and optional category/fact_ids.

    Returns:
        Dict with preview_text (formatted string for user) + structured data
        (facts_count, edges_count, cascade, fact_ids_scope) for delete_nodes.
    """
    scope      = args.get("scope", "").strip()
    profile_id = args.get("profile_id", "").strip()
    category   = args.get("category", "").strip() or None
    fact_ids   = args.get("fact_ids") or []

    # Input validation
    if not scope:
        return {"status": "error", "message": "'scope' is required (profile | category | facts)."}
    if not profile_id:
        return {"status": "error", "message": "'profile_id' is required."}
    if scope == "category" and not category:
        return {"status": "error", "message": "'category' is required when scope='category'."}
    if scope == "facts" and not fact_ids:
        return {"status": "error", "message": "'fact_ids' list required when scope='facts'."}

    data = client.preview_delete_scope(scope, profile_id, category, fact_ids)

    if "error" in data:
        return {"status": "error", "message": data["error"]}

    facts       = data["facts_in_scope"]
    cross_edges = data["cross_edges"]
    cascade     = data["cascade"]

    if not facts:
        return {
            "status":  "nothing_to_delete",
            "scope":   scope,
            "profile": profile_id,
            "message": "No facts found for the given scope. Nothing to delete.",
        }

    # ── Build the formatted preview block shown to the user ───────────────────
    lines = []

    # Header — describe what scope is being deleted
    scope_label = {
        "profile":  f"profile '{profile_id}'",
        "category": f"category '{category}' in '{profile_id}'",
        "facts":    f"{len(fact_ids)} selected fact(s) in '{profile_id}'",
    }[scope]

    lines.append(f"DELETE PREVIEW — {scope_label}")
    lines.append("═" * 54)

    # Section 1 — facts being deleted
    lines.append(f"\nFACTS TO DELETE ({len(facts)})")
    for i, f in enumerate(facts[:20], 1):   # show max 20 to keep output readable
        lines.append(f"  [{i}] {f['text']}")
    if len(facts) > 20:
        lines.append(f"  ... and {len(facts) - 20} more")

    # Section 2 — cross-category RELATED_TO edges that will be unlinked
    if cross_edges:
        lines.append(f"\nEDGES TO UNLINK ({len(cross_edges)}) — connections to other categories")
        for e in cross_edges:
            # Truncate long fact text for readability
            from_text = e["from_text"][:55] + "…" if len(e["from_text"]) > 55 else e["from_text"]
            to_text   = e["to_text"][:55]   + "…" if len(e["to_text"])   > 55 else e["to_text"]
            lines.append(f"  · [{e['from_category']}] \"{from_text}\"")
            lines.append(f"    → [{e['to_category']}]  \"{to_text}\"")
    else:
        lines.append("\nEDGES TO UNLINK — none (no cross-category connections)")

    # Section 3 — cascade auto-cleanup (empty nodes after deletion)
    lines.append("\n⚠️  AUTO-CLEANUP — nodes that become empty after deletion")
    if cascade["categories"]:
        for cat in cascade["categories"]:
            lines.append(f"  · Category '{cat}' → 0 facts remaining → will be deleted")
    else:
        lines.append("  · No categories become empty ✓")

    if cascade["profile"]:
        lines.append(f"  · Profile '{profile_id}' → 0 categories remaining → will be deleted")
    else:
        lines.append(f"  · Profile '{profile_id}' stays ✓")

    # Section 4 — total impact summary + confirmation prompt
    total_cats    = len(cascade["categories"])
    total_profile = 1 if cascade["profile"] else 0
    lines.append("\n" + "─" * 54)
    lines.append(
        f"TOTAL IMPACT: {len(facts)} fact(s)  ·  {len(cross_edges)} edge(s)  ·  "
        f"{total_cats} category(s)  ·  {total_profile} profile(s)"
    )
    lines.append("\nReply:")
    lines.append('  "yes" or "delete all"  → removes everything shown above')
    if scope == "facts":
        lines.append('  "delete 1,3,5"         → remove only those numbered facts')
    lines.append('  "cancel"               → nothing changes')

    return {
        "status":         "preview_ready",
        "scope":          scope,
        "profile_id":     profile_id,
        "category":       category,
        "fact_ids_scope": data["scoped_ids"],   # full UUID list for delete_nodes
        "facts_count":    len(facts),
        "edges_count":    len(cross_edges),
        "cascade":        cascade,
        "preview_text":   "\n".join(lines),     # show this to the user
    }


# ─────────────────────────────────────────────────────────────────────────────
# delete_nodes
# ─────────────────────────────────────────────────────────────────────────────

DELETE_NODES_DEFINITION = Tool(
    name="delete_nodes",
    description=(
        "Permanently delete facts and cascade-clean empty nodes from the knowledge graph. "
        "ALWAYS call preview_delete first and get explicit user confirmation before calling this. "
        "Handles three scopes: 'profile' (everything), 'category' (all facts in one category), "
        "'facts' (specific fact UUIDs). "
        "Automatically removes empty categories and profiles after deletion. "
        "This action is irreversible — there is no undo."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["profile", "category", "facts"],
                "description": "Must match the scope used in the preceding preview_delete call.",
            },
            "profile_id": {
                "type":        "string",
                "description": "Profile to delete from. Required for all scopes.",
            },
            "category": {
                "type":        "string",
                "description": "Category name. Required when scope='category'.",
            },
            "fact_ids": {
                "type":        "array",
                "items":       {"type": "string"},
                "description": (
                    "Fact UUIDs to delete. Required when scope='facts'. "
                    "Use the fact_ids_scope list from the preceding preview_delete result."
                ),
            },
        },
        "required": ["scope", "profile_id"],
    },
)


async def handle_delete_nodes(args: dict) -> dict:
    """
    Execute the deletion the user confirmed in the preview step.

    Passes through to db.client.delete_by_scope which runs everything
    atomically in a single Neo4j write transaction:
        1. RELATED_TO edges removed
        2. Fact nodes deleted
        3. Empty Category nodes cascade-deleted
        4. Empty Profile node cascade-deleted

    Args:
        args: Dict with scope, profile_id, and optional category/fact_ids.

    Returns:
        Structured confirmation with deleted counts + one-line summary.
    """
    scope      = args.get("scope", "").strip()
    profile_id = args.get("profile_id", "").strip()
    category   = args.get("category", "").strip() or None
    fact_ids   = args.get("fact_ids") or []

    # Validation — mirrors preview_delete to catch any mismatch
    if not scope:
        return {"status": "error", "message": "'scope' is required (profile | category | facts)."}
    if not profile_id:
        return {"status": "error", "message": "'profile_id' is required."}
    if scope == "category" and not category:
        return {"status": "error", "message": "'category' required when scope='category'."}
    if scope == "facts" and not fact_ids:
        return {"status": "error", "message": "'fact_ids' required when scope='facts'."}

    result = client.delete_by_scope(scope, profile_id, category, fact_ids)

    if result.get("status") == "nothing_to_delete":
        return {
            "status":  "nothing_to_delete",
            "message": "No matching facts found — nothing was deleted.",
        }

    if "error" in result:
        return {"status": "error", "message": result["error"]}

    # Build a concise one-line summary of what was removed
    parts = [f"{result['facts_deleted']} fact(s) deleted"]
    if result["edges_removed"]:
        parts.append(f"{result['edges_removed']} edge(s) unlinked")
    if result["categories_deleted"]:
        parts.append(f"{result['categories_deleted']} empty category(s) removed")
    if result["profile_deleted"]:
        parts.append(f"profile '{profile_id}' removed (was empty)")

    return {
        "status":             "deleted",
        "scope":              scope,
        "profile_id":         profile_id,
        "category":           category,
        "facts_deleted":      result["facts_deleted"],
        "edges_removed":      result["edges_removed"],
        "categories_deleted": result["categories_deleted"],
        "profile_deleted":    result["profile_deleted"],
        "summary":            " · ".join(parts),    # one-liner for Claude to present
    }
