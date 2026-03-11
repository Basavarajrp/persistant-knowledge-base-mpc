"""
server.py — MCP Server entry point and tool/prompt router
==========================================================

Owns the MCP Server instance. Responsibilities:
    1. list_tools()    — tells Claude which tools exist + their schemas
    2. call_tool()     — routes every tool call to the correct handler
    3. list_prompts()  — tells Claude Desktop which '/' prompts exist
    4. get_prompt()    — returns the prompt template by name
    5. Server startup  — bootstraps Neo4j + embedding model on init

Architecture (per Anthropic MCP guide):
    Each tool and prompt lives in its own module under tools/.
    server.py only collects DEFINITION/handle exports — no business logic here.
    Adding a new tool  = add module + register in _TOOL_HANDLERS + list_tools()
    Adding a new prompt = add module + register in _PROMPT_HANDLERS + list_prompts()

Knowledge graph hierarchy exposed via these tools:
    Profile → Category → Fact
    (list_profiles → list_categories → list_facts → preview_delete → delete_nodes)

Transport:
    STDIO — Claude Desktop launches this process and communicates via stdin/stdout.
    No HTTP port needed. Process lifecycle is managed by Claude Desktop.
"""

import asyncio
import json
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import GetPromptResult, Prompt, TextContent, Tool

from knowledge_graph_mcp.db import client as db
from knowledge_graph_mcp.db import embeddings
from knowledge_graph_mcp.tools import delete, profiles, prompts, query, store

# ── MCP Server instance ───────────────────────────────────────────────────────
# "knowledge-graph" is the display name shown in Claude Desktop's server list
# and in tool call traces.
server = Server("knowledge-graph")


# ── Tool registry ─────────────────────────────────────────────────────────────
# Maps tool name → async handler. Adding a tool: add module, register here,
# add its DEFINITION to list_tools().
#
# Tool groups:
#   Discovery  — list_profiles, list_categories, list_facts
#   Core       — store_fact, query_knowledge
#   Deletion   — preview_delete, delete_nodes  (always call preview first)

_TOOL_HANDLERS: dict[str, callable] = {
    # Discovery — understand what's in the graph before acting
    "list_profiles":    profiles.handle_list_profiles,
    "list_categories":  profiles.handle_list_categories,
    "list_facts":       delete.handle_list_facts,

    # Core — store and retrieve knowledge
    "store_fact":       store.handle,
    "query_knowledge":  query.handle,

    # Deletion — preview → user confirms → execute
    "preview_delete":   delete.handle_preview_delete,
    "delete_nodes":     delete.handle_delete_nodes,
}


# ── Prompt registry ───────────────────────────────────────────────────────────
# Maps prompt name → async handler. Prompts are triggered via '/' in Claude Desktop.
# They inject a conversation template — unlike tools, the LLM doesn't auto-call them.
_PROMPT_HANDLERS: dict[str, callable] = {
    "knowledge-graph-delete": prompts.handle_delete_prompt,
}


# ── list_tools handler ────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Return all tool definitions to Claude.

    Claude reads these on connection to know:
      - When to call each tool (description)
      - What arguments each tool expects (inputSchema)
      - Which arguments are required vs optional

    Ordered: discovery → core → deletion (mirrors natural usage flow).
    """
    return [
        # Discovery
        profiles.LIST_PROFILES_DEFINITION,
        profiles.LIST_CATEGORIES_DEFINITION,
        delete.LIST_FACTS_DEFINITION,

        # Core
        store.DEFINITION,
        query.DEFINITION,

        # Deletion (preview always before delete)
        delete.PREVIEW_DELETE_DEFINITION,
        delete.DELETE_NODES_DEFINITION,
    ]


# ── call_tool handler ─────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Route every tool call from Claude to the correct handler.

    Per Anthropic MCP best practices:
      - All output is JSON so Claude can parse and format results cleanly
      - Errors are structured JSON (not raw exceptions) so Claude can
        surface meaningful messages to the user
      - A catch-all prevents server crashes from unhandled exceptions

    Args:
        name:      Tool name sent by Claude (e.g. "store_fact")
        arguments: Dict of arguments matching the tool's inputSchema

    Returns:
        list[TextContent]: Single-item list with JSON result string.
                           MCP requires this specific return type.
    """
    handler = _TOOL_HANDLERS.get(name)

    if handler is None:
        result = {
            "status":  "error",
            "message": f"Unknown tool '{name}'. Available: {list(_TOOL_HANDLERS.keys())}",
        }
    else:
        try:
            result = await handler(arguments)
        except Exception as exc:
            # Catch-all: return structured error so Claude can inform the user.
            # Never let an unhandled exception crash the server process.
            result = {
                "status":  "error",
                "code":    type(exc).__name__,
                "message": str(exc),
                "hint": (
                    "Check that Neo4j is running (`docker compose up -d`) "
                    "and .env credentials match docker-compose.yml."
                ),
            }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ── list_prompts handler ──────────────────────────────────────────────────────

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """
    Return all prompt definitions to Claude Desktop.

    Claude Desktop shows these as '/' commands in the chat input.
    Each prompt is a guided conversation template — the user triggers it
    explicitly, unlike tools which the LLM calls automatically.
    """
    return [
        prompts.DELETE_PROMPT,
    ]


# ── get_prompt handler ────────────────────────────────────────────────────────

@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    """
    Return the prompt template for a given prompt name.

    Called by Claude Desktop when the user selects a '/' prompt.
    The returned messages are injected into the conversation context.

    Args:
        name:      Prompt name (e.g. "knowledge-graph-delete")
        arguments: Optional pre-fill args (none of our prompts use these yet)

    Returns:
        GetPromptResult with the conversation template messages.
    """
    handler = _PROMPT_HANDLERS.get(name)

    if handler is None:
        # Raise — MCP spec expects an error here, not a TextContent response
        raise ValueError(
            f"Unknown prompt '{name}'. Available: {list(_PROMPT_HANDLERS.keys())}"
        )

    return await handler(arguments or {})


# ── Server startup ────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Bootstrap the server and start the STDIO transport loop.

    Startup sequence:
        1. Connect to Neo4j — fail fast with a clear error if it's not running
        2. Pre-load the embedding model — first call downloads it if not cached
        3. Initialise Neo4j schema (constraints + vector index) — idempotent
        4. Start the MCP STDIO server loop — blocks until Claude disconnects

    Startup errors are written to stderr (stdout is reserved for MCP protocol).
    Claude Desktop shows the server as "failed to connect" — check logs to diagnose
    (usually: Neo4j not running, or wrong .env credentials).
    """
    try:
        db.get_driver()               # verify Neo4j connectivity
        embeddings.embed_text("warmup")   # pre-load model, warm up cache
        db.initialise()               # create constraints + vector index (idempotent)
    except Exception as exc:
        print(
            json.dumps({
                "startup_error": str(exc),
                "hint": "Run: docker compose up -d  then restart Claude Desktop.",
            }),
            file=sys.stderr,
        )
        raise

    # STDIO server loop — runs until Claude Desktop closes the connection
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
