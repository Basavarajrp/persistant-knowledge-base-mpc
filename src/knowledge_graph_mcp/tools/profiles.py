"""
tools/profiles.py — list_profiles and list_categories tools
=============================================================

Discovery tools. Claude calls these at the start of every session to
understand what namespaces and knowledge categories already exist before
deciding where to store or search.

Following Anthropic MCP best practices:
  - Separate tool per distinct operation (not one overloaded "list" tool)
  - Descriptions explain WHEN to call, not just WHAT it does
  - Returns complete structured data so Claude can present it without
    needing a follow-up tool call
"""

from mcp.types import Tool
from knowledge_graph_mcp.db import client

# ── list_profiles ─────────────────────────────────────────────────────────────

LIST_PROFILES_DEFINITION = Tool(
    name="list_profiles",
    description=(
        "List all profile namespaces in the knowledge graph with fact counts. "
        "Call this at the START of every store or query session to see which "
        "profiles exist. Prevents duplicate profile names (e.g. 'Project1' vs 'project_1'). "
        "Returns profile IDs, fact counts, and creation dates."
    ),
    inputSchema={
        "type":       "object",
        "properties": {},
        "required":   [],
    },
)


async def handle_list_profiles(args: dict) -> dict:
    """
    Return all Profile nodes with fact counts sorted descending.

    Args:
        args: Empty dict (no arguments needed).

    Returns:
        Dict with 'profiles' list or 'empty' status if none exist yet.
    """
    profiles = client.list_profiles()

    if not profiles:
        return {
            "status":   "empty",
            "message":  "Knowledge graph is empty. No profiles stored yet.",
            "profiles": [],
        }

    return {
        "status":   "ok",
        "profiles": profiles,
    }


# ── list_categories ───────────────────────────────────────────────────────────

LIST_CATEGORIES_DEFINITION = Tool(
    name="list_categories",
    description=(
        "List all knowledge categories inside a specific profile with fact counts. "
        "Use this to understand how knowledge is organised within a profile before "
        "running a targeted query or to give the user an overview of what is stored. "
        "Example output: auth(12 facts), database(8 facts), api(15 facts)."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "profile_id": {
                "type":        "string",
                "description": "Profile ID to list categories for.",
            },
        },
        "required": ["profile_id"],
    },
)


async def handle_list_categories(args: dict) -> dict:
    """
    Return all Category nodes in a profile with fact counts.

    Args:
        args: Dict containing 'profile_id'.

    Returns:
        Dict with 'categories' list or 'empty' status.
    """
    profile_id = args.get("profile_id", "").strip()

    if not profile_id:
        return {"status": "error", "message": "'profile_id' is required."}

    categories = client.list_categories(profile_id)

    if not categories:
        return {
            "status":     "empty",
            "profile":    profile_id,
            "message":    f"No categories in profile '{profile_id}' yet.",
            "categories": [],
        }

    return {
        "status":     "ok",
        "profile":    profile_id,
        "categories": categories,
    }
