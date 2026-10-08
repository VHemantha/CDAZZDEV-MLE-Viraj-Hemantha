"""Task 3B: two specialised agents coordinated by a LangGraph pipeline, with a
critique loop, plus Task 3C persistent caching of the final brief.

    cache_check --hit--> END
         | miss
    analyst_brief -> writer_research -> writer_critique -> analyst_clarify -> writer_final -> save -> END
       (Agent A)        (Agent B)        (B -> A request)     (A -> B answer)     (Agent B)

Data crosses agent boundaries ONLY as Pydantic models (DataBrief,
ClarificationRequest, ClarificationResponse) -- never as free text.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph orchestrator for a data-analyst and a
# research-writer agent with Pydantic handoffs, one critique loop and a JSON cache',
# Date: 2026-10-07

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from . import config, memory
from .agent import build_agent, run_agent
from .llm import TRANSIENT_ERRORS, make_chat, structured_invoke
from .prompts import (ANALYST_CLARIFY_TASK, ANALYST_SYSTEM, ANALYST_TASK, CLARIFICATION_INSTRUCTION,
                      CLARIFICATION_SYSTEM, CLARIFY_RESPONSE_INSTRUCTION, DATA_BRIEF_INSTRUCTION,
                      FINAL_REPORT_SYSTEM, RESEARCH_QUERY, WRITER_SYSTEM, WRITER_TASK)
from .research import evidence_text
from .schemas import (BriefNarrative, ClarificationRequest, ClarificationResponse, ClarificationSummary,
                      DataBrief, FinalReport, PriceSnapshot, SentimentSnapshot, VolatilitySnapshot)
from .tracing import TRACER


class PipelineState(TypedDict, total=False):
    ticker: str
    query: str
    use_cache: bool
    cache_hit: bool
    cache_file: str
    data_brief: DataBrief
    writer_evidence: str
    clarification_request: ClarificationRequest
    clarification_response: ClarificationResponse
    final_report: FinalReport


def _banner(text: str) -> None:
    TRACER.say("\n" + "═" * 100 + f"\n{text}\n" + "═" * 100)


def _handoff(kind: str, sender: str, receiver: str, model) -> None:
    """Print the full typed payload and log it to the trace file."""
    payload = model.model_dump(mode="json")
    TRACER.say(f"\n  ⇢ HANDOFF {kind}: {sender} → {receiver} ({type(model).__name__})")
    TRACER.say("  " + json.dumps(payload, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    TRACER.event("handoff", payload, kind=kind, sender=sender, receiver=receiver,
                 schema=type(model).__name__)


def _as_agent(role: str, fn, *args, **kwargs):
    token = TRACER.set_agent(role)
    try:
        return fn(*args, **kwargs)
    finally:
        TRACER.reset_agent(token)


def _latest_tool_json(messages, tool: str) -> dict | None:
    """Most recent successful JSON result of ``tool`` in an agent's messages."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == tool:
            try:
                data = json.loads(str(m.content))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "error" not in data:
                return data
    return None


def snapshots_from_tools(messages) -> dict[str, Any]:
    """Numbers handed between agents are copied from tool output, never re-typed
    by an LLM -- so a handoff cannot contain a hallucinated figure."""
    out: dict[str, Any] = {}
    if (p := _latest_tool_json(messages, "get_price_data")):
        ind = p.get("indicators", {})
        out["price"] = PriceSnapshot(last_close=p["last_close"], period_return_pct=p.get("period_return_pct"),
                                     pct_vs_sma_50=ind.get("pct_vs_sma_50"), pct_vs_sma_200=ind.get("pct_vs_sma_200"),
                                     rsi_14=ind.get("rsi_14"), macd_hist=ind.get("macd_hist"),
                                     max_drawdown_pct=p.get("max_drawdown_pct"))
    if (v := _latest_tool_json(messages, "calculate_volatility")):
        out["volatility"] = VolatilitySnapshot(window_days=v["window_days"], annualised_vol_pct=v["annualised_vol_pct"],
                                               one_year_vol_pct=v.get("one_year_vol_pct"),
                                               var_95_1d_pct=v.get("var_95_1d_pct"), beta_vs_spy=v.get("beta_vs_spy"))
    if (s := _latest_tool_json(messages, "llm_sentiment")):
        out["sentiment"] = SentimentSnapshot(score=s["score"], label=s["label"], n_headlines=s["n_headlines"])
    return out


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def cache_check(state: PipelineState) -> PipelineState:
    if not state.get("use_cache", True):
        return {"cache_hit": False}
    hit = memory.load_cached(state["ticker"])
    if hit is None:
        TRACER.say(f"[memory] no cached brief for {state['ticker']} today -> running agents")
        return {"cache_hit": False}
    payload, path = hit
    TRACER.say(f"[memory] CACHE HIT {path.name} -> loading brief, skipping all tools")
    TRACER.event("cache_hit", {"file": path.name}, ticker=state["ticker"])
    return {"cache_hit": True, "cache_file": str(path),
            "final_report": FinalReport.model_validate(payload["final_report"])}


def analyst_brief(state: PipelineState) -> PipelineState:
    _banner("AGENT A · Data Analyst · tools: get_price_data, calculate_volatility, llm_sentiment")
    graph = build_agent("data_analyst", ANALYST_SYSTEM)
    msgs = run_agent(graph, "data_analyst", ANALYST_TASK.substitute(ticker=state["ticker"]))
    snaps = snapshots_from_tools(msgs)
    if "price" not in snaps or "volatility" not in snaps:
        raise RuntimeError("Agent A could not obtain price and volatility data; cannot build DataBrief")
    narrative = _as_agent("data_analyst", structured_invoke, make_chat("data_analyst", json_mode=True),
                          ANALYST_SYSTEM, f"Ticker {state['ticker']}.\nTool results:\n{evidence_text(msgs)}",
                          BriefNarrative, DATA_BRIEF_INSTRUCTION)
    brief = DataBrief(ticker=state["ticker"], as_of=str(datetime.now(timezone.utc).date()),
                      **snaps, **narrative.model_dump())
    _handoff("data_brief", "Agent A", "Agent B", brief)
    return {"data_brief": brief}


def writer_research(state: PipelineState) -> PipelineState:
    _banner("AGENT B · Research Writer · tools: web_search, get_news")
    graph = build_agent("research_writer", WRITER_SYSTEM)
    task = WRITER_TASK.substitute(query=state["query"], brief=state["data_brief"].model_dump_json(indent=1))
    msgs = run_agent(graph, "research_writer", task)
    return {"writer_evidence": evidence_text(msgs)}


def writer_critique(state: PipelineState) -> PipelineState:
    _banner("CRITIQUE LOOP · Agent B reviews the brief and asks Agent A for one clarification")
    req = _as_agent("research_writer", structured_invoke, make_chat("research_writer", json_mode=True),
                    CLARIFICATION_SYSTEM,
                    f"DataBrief:\n{state['data_brief'].model_dump_json()}\n\nYour research:\n"
                    f"{state['writer_evidence']}", ClarificationRequest, CLARIFICATION_INSTRUCTION)
    _handoff("clarification_request", "Agent B", "Agent A", req)
    return {"clarification_request": req}


def analyst_clarify(state: PipelineState) -> PipelineState:
    _banner("AGENT A · answering Agent B's clarification with its own tools")
    graph = build_agent("data_analyst", ANALYST_SYSTEM)
    req = state["clarification_request"]
    msgs = run_agent(graph, "data_analyst", ANALYST_CLARIFY_TASK.substitute(
        request=req.model_dump_json(indent=1)) + f"\nTicker: {state['ticker']}")
    summary = _as_agent("data_analyst", structured_invoke, make_chat("data_analyst", json_mode=True),
                        ANALYST_SYSTEM, f"Request:\n{req.model_dump_json()}\n\nTool results:\n{evidence_text(msgs)}",
                        ClarificationSummary, CLARIFY_RESPONSE_INSTRUCTION)
    resp = ClarificationResponse(answer_summary=summary.answer_summary, **snapshots_from_tools(msgs))
    _handoff("clarification_response", "Agent A", "Agent B", resp)
    return {"clarification_response": resp}


def writer_final(state: PipelineState) -> PipelineState:
    _banner("AGENT B · writing the final report (incorporating Agent A's answer)")
    context = (f"Research question: {state['query']}\n\nAgent A DataBrief:\n"
               f"{state['data_brief'].model_dump_json()}\n\nAgent B research evidence:\n"
               f"{state['writer_evidence']}\n\nClarification asked:\n"
               f"{state['clarification_request'].model_dump_json()}\n\nAgent A answer:\n"
               f"{state['clarification_response'].model_dump_json()}")
    report = _as_agent("research_writer", structured_invoke, make_chat("research_writer", json_mode=True),
                       FINAL_REPORT_SYSTEM, context, FinalReport, "Write the final report.")
    TRACER.event("final_report", report.model_dump(mode="json"), ticker=state["ticker"])
    return {"final_report": report}


def save(state: PipelineState) -> PipelineState:
    payload: dict[str, Any] = {
        "ticker": state["ticker"], "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": TRACER.session_id,
        "final_report": state["final_report"].model_dump(mode="json"),
        "data_brief": state["data_brief"].model_dump(mode="json"),
        "clarification_request": state["clarification_request"].model_dump(mode="json"),
        "clarification_response": state["clarification_response"].model_dump(mode="json"),
    }
    path = memory.save_brief(state["ticker"], payload)
    TRACER.say(f"\n[memory] brief saved -> {path.name}")
    return {"cache_file": str(path)}


def build_pipeline():
    g = StateGraph(PipelineState)
    for name, fn in [("cache_check", cache_check), ("analyst_brief", analyst_brief),
                     ("writer_research", writer_research), ("writer_critique", writer_critique),
                     ("analyst_clarify", analyst_clarify), ("writer_final", writer_final), ("save", save)]:
        # Node-level retry for transient provider/network failures: the node re-runs
        # from its input state, so a blip mid-pipeline doesn't lose the earlier agents' work.
        g.add_node(name, fn, retry_policy=RetryPolicy(max_attempts=config.NODE_MAX_ATTEMPTS,
                                                      initial_interval=10.0, backoff_factor=2.0,
                                                      retry_on=TRANSIENT_ERRORS))
    g.add_edge(START, "cache_check")
    g.add_conditional_edges("cache_check", lambda s: END if s.get("cache_hit") else "analyst_brief",
                            {END: END, "analyst_brief": "analyst_brief"})
    g.add_edge("analyst_brief", "writer_research")
    g.add_edge("writer_research", "writer_critique")
    g.add_edge("writer_critique", "analyst_clarify")
    g.add_edge("analyst_clarify", "writer_final")
    g.add_edge("writer_final", "save")
    g.add_edge("save", END)
    return g.compile()


def run_pipeline(ticker: str, use_cache: bool = True) -> PipelineState:
    TRACER.new_session()
    return build_pipeline().invoke({"ticker": ticker.upper(), "use_cache": use_cache,
                                    "query": RESEARCH_QUERY.substitute(ticker=ticker.upper())})
