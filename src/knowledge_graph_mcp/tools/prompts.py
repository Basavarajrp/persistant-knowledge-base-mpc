"""
tools/prompts.py — MCP Prompt definitions
==========================================

MCP Prompts are server-defined conversation templates triggered by the user
via '/' commands in Claude Desktop. Unlike tools (LLM auto-called) and
resources (user @-tagged), a prompt injects a structured instruction block
into the active conversation.

Why a prompt for deletion:
    Deletion is irreversible. The prompt enforces a strict step order —
    preview always runs before delete — and requires explicit user confirmation
    at each step. Without this template, Claude could theoretically skip the
    preview and call delete_nodes directly.

MCP Prompt registration (in server.py):
    list_prompts()  → returns [DELETE_PROMPT, ...]
    get_prompt()    → routes name → handler → returns GetPromptResult

Current prompts:
    /knowledge-graph-delete  — guided delete workflow: discover → preview → confirm → execute
"""

from mcp.types import GetPromptResult, Prompt, PromptMessage, TextContent


# ─────────────────────────────────────────────────────────────────────────────
# /knowledge-graph-delete
# ─────────────────────────────────────────────────────────────────────────────

DELETE_PROMPT = Prompt(
    name="knowledge-graph-delete",
    description=(
        "Guided deletion workflow for the knowledge graph. "
        "Identifies what to delete (profile / category / specific facts), "
        "shows a full preview with cascade auto-cleanup, "
        "and waits for explicit confirmation before touching anything."
    ),
    arguments=[],   # no pre-fill args — the prompt asks the user interactively
)


async def handle_delete_prompt(args: dict) -> GetPromptResult:
    """
    Return the deletion workflow instruction injected into the conversation.

    The instruction enforces a 5-step safety order:
        STEP 1 — Identify: ask what to delete, help user find exact names
        STEP 2 — Preview:  call preview_delete, show output verbatim to user
        STEP 3 — Confirm:  wait for explicit "yes" / "delete 1,3" / "cancel"
        STEP 4 — Execute:  call delete_nodes only if user confirmed
        STEP 5 — Report:   show the summary line from delete_nodes result

    Claude must NOT skip or reorder these steps. The confirmation gate at
    STEP 3 is the key safety boundary — vague or non-confirmatory replies
    must not proceed to STEP 4.

    Args:
        args: Empty dict (this prompt takes no arguments).

    Returns:
        GetPromptResult with the workflow as a user-role message.
    """
    instruction = """\
You are now running the knowledge-graph deletion workflow.
Follow these 5 steps in order — do not skip or reorder them.

──────────────────────────────────────────────────────────────
STEP 1 — IDENTIFY THE TARGET
──────────────────────────────────────────────────────────────
Ask the user what they want to delete if they haven't already said:
  "What would you like to remove — a whole profile, a whole category, or specific facts?"

Help them confirm exact names:
  - To list profiles: call list_profiles
  - To list categories in a profile: call list_categories
  - To list individual facts (for selective deletion): call list_facts

Do not proceed to STEP 2 until you have: scope, profile_id, and (if needed) category or fact_ids.

──────────────────────────────────────────────────────────────
STEP 2 — PREVIEW (mandatory — never skip this step)
──────────────────────────────────────────────────────────────
Call preview_delete with the identified scope, profile_id, category, and/or fact_ids.

Present the preview_text from the result to the user EXACTLY as returned.
Do not paraphrase, summarise, or omit any section of the preview.

The preview shows:
  · Facts to be deleted (numbered list)
  · Cross-category edges that will be unlinked
  · ⚠️ AUTO-CLEANUP: empty categories and profiles that will be removed

──────────────────────────────────────────────────────────────
STEP 3 — WAIT FOR EXPLICIT CONFIRMATION
──────────────────────────────────────────────────────────────
After showing the preview, ask:
  "Shall I go ahead with this deletion?"

Valid confirmations:
  "yes" or "delete all"   → proceed with full deletion as previewed
  "delete 1, 3, 5"        → for scope='facts' only — delete only those numbered items
                             (map numbers back to fact_ids from the preview result)
  "cancel" or "no"        → stop immediately, delete nothing, tell user nothing was changed

Do NOT interpret vague, ambiguous, or unclear replies as confirmation.
When in doubt, ask again.

──────────────────────────────────────────────────────────────
STEP 4 — EXECUTE (only after confirmed)
──────────────────────────────────────────────────────────────
Call delete_nodes with:
  - The same scope / profile_id / category as the preview
  - If user said "delete all": use fact_ids_scope from the preview_delete result
  - If user selected numbers (e.g. "delete 1,3"): map those indices to the
    corresponding fact_ids from the preview's facts list, pass only those

──────────────────────────────────────────────────────────────
STEP 5 — CONFIRM COMPLETION
──────────────────────────────────────────────────────────────
Show the summary field from the delete_nodes result as a single confirmation line.
Example: "Done — 12 facts deleted · 3 edges unlinked · category 'auth' removed"

If the user cancelled at STEP 3, confirm:
"Cancelled — nothing was deleted."\
"""

    return GetPromptResult(
        description="Guided knowledge-graph deletion workflow",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=instruction),
            )
        ],
    )
