"""
server.py — MCP Server entry point and tool router
====================================================

This module owns the MCP Server instance. It:
  1. Registers list_tools() — tells Claude which tools exist + their schemas
  2. Registers call_tool() — routes every tool call to the correct handler
  3. Runs the STDIO server loop (Claude Desktop transport)
  4. Bootstraps Neo4j + embedding model on startup

Architecture principle (per Anthropic MCP guide):
  Each tool is its own module (tools/store.py, tools/query.py, tools/profiles.py).
  server.py collects their DEFINITION and handle() exports — it never contains
  business logic itself. Adding a new tool = add a new module + register here.

Transport:
  STDIO — Claude Desktop launches this process and communicates via stdin/stdout.
  No HTTP port needed. The process lifecycle is managed by Claude Desktop.
"""

import asyncio
import json
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from knowledge_graph_mcp.db import client as db
from knowledge_graph_mcp.db import embeddings
from knowledge_graph_mcp.tools import store, query, profiles

# ── MCP Server instance ───────────────────────────────────────────────────────
# "knowledge-graph" is the display name Claude Desktop shows in its
# connected servers list and in tool call traces.
server = Server("knowledge-graph")


# ── Tool registry ─────────────────────────────────────────────────────────────
# Maps tool name → async handler function.
# Adding a new tool: import its module, add its DEFINITION to list_tools(),
# and add its handle() to this dict.
_TOOL_HANDLERS: dict[str, callable] = {
    "store_fact":       store.handle,
    "query_knowledge":  query.handle,
    "list_profiles":    profiles.handle_list_profiles,
    "list_categories":  profiles.handle_list_categories,
}


# ── list_tools handler ────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Return all tool definitions to Claude.

    Claude reads these once on connection to understand:
      - Which tools are available (name)
      - When to use each tool (description)
      - What arguments each tool expects (inputSchema)

    Per Anthropic best practices:
      - Each description explains WHEN to call the tool, not just WHAT it does
      - inputSchema has a description on every property
      - Required vs optional fields are clearly separated
    """
    return [
        store.DEFINITION,
        query.DEFINITION,
        profiles.LIST_PROFILES_DEFINITION,
        profiles.LIST_CATEGORIES_DEFINITION,
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
      - A catch-all wraps every handler to prevent server crashes

    Args:
        name:      Tool name sent by Claude (e.g. "store_fact")
        arguments: Dict of arguments from Claude matching the tool's inputSchema

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
            # Catch-all: surface structured error so Claude can inform the user.
            # Never let an unhandled exception crash the server process.
            result = {
                "status":  "error",
                "code":    type(exc).__name__,
                "message": str(exc),
                "hint":    (
                    "Check that Neo4j is running (`docker compose up -d`) "
                    "and .env credentials match docker-compose.yml."
                ),
            }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


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
    Claude Desktop will show the server as "failed to connect" and the user
    can check the logs to diagnose (typically: Neo4j not running).
    """
    try:
        db.get_driver()           # Verify Neo4j connectivity
        embeddings.embed_text("warmup")  # Pre-load model, cache it
        db.initialise()           # Create constraints + vector index
    except Exception as exc:
        print(
            json.dumps({
                "startup_error": str(exc),
                "hint": "Run: docker compose up -d  then restart Claude Desktop.",
            }),
            file=sys.stderr,
        )
        raise

    # STDIO server loop — runs until Claude Desktop closes the connection.
    # All communication happens via stdin/stdout using MCP JSON-RPC protocol.
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
