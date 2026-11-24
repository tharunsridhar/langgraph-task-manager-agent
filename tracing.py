"""Opik tracing, opt-in.

Tracing is on only when Opik is configured (an `OPIK_API_KEY`, a self-hosted
`OPIK_URL_OVERRIDE`, or a `~/.opik.config` created by `opik configure`) and the `opik`
package is installed. Otherwise the app runs exactly as before, with no network calls.

The LangGraph integration (`OpikTracer`) records the agent node, each LLM call, each tool
call and the guardrail nodes as nested spans under one trace per request.
"""
import os
from pathlib import Path

PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "task-manager-agent")


def tracing_enabled() -> bool:
    if os.getenv("OPIK_TRACING", "").lower() in ("0", "false", "off"):
        return False
    configured = (
        os.getenv("OPIK_API_KEY")
        or os.getenv("OPIK_URL_OVERRIDE")
        or (Path.home() / ".opik.config").exists()
    )
    return bool(configured)


def get_callbacks(thread_id: str | None = None) -> list:
    """Callbacks to put in the LangGraph run config. Empty when tracing is off or
    the opik package is missing; a tracing problem must never break a chat request."""
    if not tracing_enabled():
        return []
    try:
        from opik.integrations.langchain import OpikTracer

        return [OpikTracer(project_name=PROJECT_NAME, thread_id=thread_id,
                           tags=["task-manager-agent"])]
    except Exception:
        return []
