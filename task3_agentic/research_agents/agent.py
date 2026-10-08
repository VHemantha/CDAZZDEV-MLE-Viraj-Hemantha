"""A ReAct agent as an explicit LangGraph StateGraph.

    START -> agent --(tool_calls?)--> tools -> agent -> ... -> END

Built by hand (rather than a prebuilt helper) so that three production concerns
are visible and testable:
  1. Tool access control is enforced at EXECUTION time, not just by which tools
     are bound to the model -- a hallucinated call to a forbidden tool is denied
     and traced, never executed.
  2. Every step is echoed as a readable message trace (thought -> call -> observation).
  3. A step budget stops runaway tool loops; tool errors become observations.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Hand-built LangGraph ReAct agent with
# execution-time tool allowlist, step budget, checkpointer memory and message trace',
# Date: 2026-10-07

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from openai import BadRequestError
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from langgraph.graph.message import add_messages

from . import config
from .llm import TRANSIENT_ERRORS, make_chat
from .tools import ALL_TOOLS, ROLE_TOOLS
from .tracing import TRACER


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def _preview(text: str, n: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def compact_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Context-budget management for small free-tier input limits (e.g. 7K tokens/min):
    the most recent KEEP_FULL_OBSERVATIONS tool results stay intact, older ones are cut
    to a short prefix. The agent already reasoned over them, and the full results
    remain in the graph state and the trace -- only the prompt is compacted."""
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    old = set(tool_idx[:-config.KEEP_FULL_OBSERVATIONS]) if len(tool_idx) > config.KEEP_FULL_OBSERVATIONS else set()
    out = []
    for i, m in enumerate(messages):
        if i in old and len(str(m.content)) > config.COMPACTED_OBSERVATION_CHARS:
            m = ToolMessage(content=str(m.content)[: config.COMPACTED_OBSERVATION_CHARS] + "…(compacted)",
                            tool_call_id=m.tool_call_id, name=m.name)
        out.append(m)
    return out


class ToolExecutor:
    """Runs tool calls for ONE role. Calls outside the role's allowlist are denied."""

    def __init__(self, role: str):
        self.role = role
        self.allowed = set(ROLE_TOOLS[role])

    def __call__(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for call in getattr(last, "tool_calls", []) or []:
            name, args = call["name"], call.get("args", {}) or {}
            if name not in self.allowed:
                content = json.dumps({"error": f"PermissionError: tool '{name}' is not available to "
                                               f"{self.role}", "allowed_tools": sorted(self.allowed)})
                TRACER.tool_call(name, args, content, 0.0, "denied")
            elif name not in ALL_TOOLS:
                content = json.dumps({"error": f"unknown tool '{name}'"})
                TRACER.tool_call(name, args, content, 0.0, "error")
            else:
                content = ALL_TOOLS[name].invoke(args)   # traced inside; never raises
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
        return {"messages": results}


def build_agent(role: str, system_prompt: str, checkpointer: BaseCheckpointSaver | None = None):
    """Compile a ReAct graph for ``role`` with only that role's tools bound."""
    tools = [ALL_TOOLS[t] for t in ROLE_TOOLS[role]]
    llm = make_chat(role, tools=tools)
    executor = ToolExecutor(role)

    plain = make_chat(role)   # unbound model: used when tools must not be called

    def invoke_resilient(msgs: list) -> AIMessage:
        """gpt-oss occasionally emits a malformed tool name (e.g. 'get_news<|channel|>commentary')
        which the provider rejects with 400 tool_use_failed. Retry with a corrective nudge;
        if it persists, answer without tools instead of crashing the run."""
        for attempt in range(1 + config.MALFORMED_TOOL_CALL_RETRIES):
            try:
                return llm.invoke(msgs)
            except BadRequestError as exc:
                if "tool_use_failed" not in str(exc):
                    raise
                TRACER.event("malformed_tool_call", str(exc)[:300], attempt=attempt + 1)
                TRACER.say(f"  [{role}] provider rejected a malformed tool call (attempt {attempt + 1}); retrying")
                msgs = [*msgs, SystemMessage(
                    f"Your last tool call was malformed. Call exactly one of {ROLE_TOOLS[role]} "
                    "using its exact name, with no extra text in the name.")]
        msgs = [*msgs, SystemMessage("Tool calling is unavailable; answer with the evidence you have.")]
        return plain.invoke(msgs)

    def agent_node(state: AgentState) -> dict:
        steps = state.get("steps", 0) + 1
        msgs = [SystemMessage(system_prompt), *compact_history(state["messages"])]
        if steps >= config.MAX_AGENT_STEPS:
            msgs.append(SystemMessage("Step budget reached: do not call more tools; answer now."))
            reply = plain.invoke(msgs)
        else:
            reply = invoke_resilient(msgs)
        # Prefer visible content; fall back to the model's reasoning summary (gpt-oss).
        reasoning = reply.additional_kwargs.get("reasoning", "")
        thought = _preview(reply.content or reasoning, 300)
        calls = ", ".join(f"{c['name']}({json.dumps(c.get('args', {}))[:80]})" for c in reply.tool_calls or [])
        model = (reply.response_metadata or {}).get("model_name", "?")
        if calls:
            TRACER.say(f"  [{role}] step {steps} THINK ({model}): {thought or '(decides to call tools)'}")
            TRACER.say(f"  [{role}] step {steps} ACT  : {calls}")
        else:
            TRACER.say(f"  [{role}] step {steps} ANSWER ({model}): {thought}")
        return {"messages": [reply], "steps": steps}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node, retry_policy=RetryPolicy(
        max_attempts=config.NODE_MAX_ATTEMPTS, initial_interval=10.0, backoff_factor=2.0, retry_on=TRANSIENT_ERRORS))
    g.add_node("tools", executor)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


def run_agent(graph, role: str, user_message: str, thread_id: str | None = None) -> list[AnyMessage]:
    """Invoke with the tracer's agent context set; returns the full message list."""
    token = TRACER.set_agent(role)
    try:
        cfg: dict[str, Any] = {"recursion_limit": 4 * config.MAX_AGENT_STEPS + 5}
        if thread_id:
            cfg["configurable"] = {"thread_id": thread_id}
        # steps resets per user turn so follow-ups get a fresh budget
        out = graph.invoke({"messages": [("user", user_message)], "steps": 0}, cfg)
        return out["messages"]
    finally:
        TRACER.reset_agent(token)


def tool_observations(messages: list[AnyMessage]) -> list[tuple[str, str]]:
    return [(m.name, str(m.content)) for m in messages if isinstance(m, ToolMessage)]
