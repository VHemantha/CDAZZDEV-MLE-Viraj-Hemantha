"""Task 3A (single autonomous agent) and 3C short-term memory follow-ups."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Run a single LangGraph research agent, extract
# a validated report, and answer follow-ups from checkpointed context', Date: 2026-10-07

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from .agent import build_agent, run_agent
from .llm import make_chat, structured_invoke
from .prompts import REPORT_SYSTEM, RESEARCH_AGENT_SYSTEM, RESEARCH_QUERY
from .schemas import ResearchReport
from .tracing import TRACER


def evidence_text(messages: list[AnyMessage], max_chars_per_obs: int = 1000) -> str:
    """Compact transcript of tool observations + the agent's final analysis."""
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            parts.append(f"[{m.name} result] {str(m.content)[:max_chars_per_obs]}")
    finals = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls and m.content]
    if finals:
        parts.append(f"[agent final analysis] {finals[-1].content}")
    return "\n\n".join(parts)


@dataclass
class SingleAgentSession:
    graph: object
    thread_id: str
    messages: list[AnyMessage]
    report: ResearchReport | None


def run_single_agent(ticker: str) -> SingleAgentSession:
    """3A: the agent chooses its own tool sequence; the report is then extracted
    and validated against ResearchReport."""
    checkpointer = InMemorySaver()           # short-term memory for follow-ups (3C)
    graph = build_agent("research_agent", RESEARCH_AGENT_SYSTEM, checkpointer)
    thread_id = f"single-{uuid.uuid4().hex[:6]}"
    TRACER.say(f"=== Research agent | thread {thread_id} ===")
    messages = run_agent(graph, "research_agent", RESEARCH_QUERY.substitute(ticker=ticker), thread_id)
    token = TRACER.set_agent("research_agent")
    try:
        report = structured_invoke(make_chat("research_agent", json_mode=True), REPORT_SYSTEM,
                                   f"Ticker: {ticker}\n\nEvidence:\n{evidence_text(messages)}",
                                   ResearchReport, "Write the structured research report.")
    except ValueError as exc:
        TRACER.say(f"  report extraction failed: {exc}")
        report = None
    finally:
        TRACER.reset_agent(token)
    return SingleAgentSession(graph, thread_id, messages, report)


def follow_up(session: SingleAgentSession, question: str) -> tuple[str, int]:
    """Ask a follow-up on the SAME thread. Returns (answer, tools called during it)."""
    before = len(session.messages)
    msgs = run_agent(session.graph, "research_agent", question, session.thread_id)
    new = msgs[before:]
    n_tools = sum(isinstance(m, ToolMessage) for m in new)
    answer = next((str(m.content) for m in reversed(new) if isinstance(m, AIMessage) and m.content), "")
    session.messages = msgs
    return answer, n_tools
