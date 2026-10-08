"""Generates notebooks/Task3_Multi_Agent_Research.ipynb (execute with nbconvert)."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate the Task 3 walkthrough notebook',
# Date: 2026-10-07

from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName"
COLAB = ("https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName/"
         "blob/main/task3_agentic/notebooks/Task3_Multi_Agent_Research.ipynb")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(f"""# Task 3: Multi-Agent Financial Research System
**CDAZZDEV Senior MLE Assessment · Agentic Workflows** · framework: **LangGraph** · LLMs: Groq free tier (`gpt-oss-120b` / `gpt-oss-20b`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

| Part | What this notebook shows | Code |
|---|---|---|
| 3A | Five traced tools; one autonomous ReAct agent; a visible observe→replan cycle (including recovery from a tool failure); a three-section report | `tools.py`, `agent.py`, `research.py` |
| 3B | Agent A (Data Analyst) and Agent B (Research Writer) with tool allowlists enforced at execution time; Pydantic handoffs; a critique loop | `multi_agent.py`, `schemas.py` |
| 3C | Short-term memory (a follow-up answered from context with 0 tool calls); a persistent JSON cache keyed by ticker and date; `agent_trace.jsonl` | `memory.py`, `tracing.py` |
| Bonus | A Streamlit dashboard that reads the trace | `dashboard/app.py` |

**Design choice:** the ReAct loop is an explicit LangGraph `StateGraph` (`agent ⇄ tools`), not a prebuilt helper. This lets the tool executor deny calls outside an agent's role, apply a step budget, and turn every tool error into an observation the agent can react to."""),
    code(f"""import os, sys, subprocess, json, time
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not Path("/content/repo").exists():
        subprocess.run(["git", "clone", "-q", "{REPO}", "/content/repo"], check=True)
    os.chdir("/content/repo/task3_agentic")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    from google.colab import userdata
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
else:
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)
    from dotenv import load_dotenv
    load_dotenv(Path.cwd().parent / ".env", override=True)
sys.path.insert(0, str(Path.cwd()))
print("cwd:", Path.cwd(), "| GROQ_API_KEY", "set" if os.getenv("GROQ_API_KEY") else "MISSING")"""),
    code("""import logging
import pandas as pd
from IPython.display import Markdown, display
from research_agents import config, tools as T
from research_agents.tracing import TRACER
logging.getLogger("httpx").setLevel(logging.WARNING)
pd.set_option("display.max_colwidth", 120)

# Fresh trace for this notebook run (the previous one is archived, not deleted).
if config.TRACE_PATH.exists():
    config.TRACE_PATH.rename(config.TRACE_PATH.with_name(f"agent_trace_prev_{int(time.time())}.jsonl"))
TICKER = "MSFT"
print("models:", config.MODELS)"""),

    md("""---
# 3A: Tool-using research agent
## 3A.1 The five tools: each is callable and returns a typed result
Every tool is wrapped by `@traced`, which handles timing, JSONL logging, fault injection, and a last-resort exception guard. A failure comes back as `{"error", "hint"}` or an empty list and never as an exception, so the agent can *observe* the failure and re-plan."""),
    code("""TRACER.set_agent("tool_smoke_test")
checks = {
    "get_price_data": T.get_price_data(ticker=TICKER, period="1y"),
    "get_news": T.get_news(ticker=TICKER, n=5),
    "calculate_volatility": T.calculate_volatility(ticker=TICKER, window=30),
    "web_search": T.web_search(query=f"{TICKER} analyst outlook risks", max_results=3),
}
checks["llm_sentiment"] = T.llm_sentiment(headlines=[n["headline"] for n in checks["get_news"]])
pd.DataFrame([{"tool": k, "return type": type(v).__name__,
               "items/keys": len(v), "preview": json.dumps(v, default=str)[:150]} for k, v in checks.items()])"""),
    code("""print("Failure paths return structured errors instead of raising:")
print(" bad ticker  ->", T.get_price_data(ticker="NOTATICKERZZ", period="1y"))
print(" bad window  ->", T.calculate_volatility(ticker=TICKER, window=2))
print(" no headlines->", T.llm_sentiment(headlines=[]))"""),

    md("""## 3A.2 Autonomous run with an injected tool failure
The agent gets the brief's query and all five tools; **no tool order is coded anywhere**. To show the "tool fails → try an alternative" requirement on a real run, we **inject one simulated outage of `get_news`** (`set_fault_injection`, clearly labelled in the trace).

How to read the trace: `THINK` is the model's own reasoning summary returned by the provider, `ACT` is the tool call it chose, and `✓/✗` is the observation. The observe→replan cycle is the `✗ get_news` line followed by a THINK that chooses an alternative."""),
    code("""from research_agents.research import run_single_agent
from research_agents.render import report_markdown
T.set_fault_injection({"get_news": 1})
TRACER.new_session()
session = run_single_agent(TICKER)
T.set_fault_injection({})"""),
    code("""# Extract the observe -> replan evidence from the run's messages
from langchain_core.messages import AIMessage, ToolMessage
seq = []
for m in session.messages:
    if isinstance(m, AIMessage) and m.tool_calls:
        seq.append(("DECIDE", ", ".join(c["name"] for c in m.tool_calls), m.additional_kwargs.get("reasoning", "")[:140]))
    elif isinstance(m, ToolMessage):
        status = "ERROR" if '"error"' in str(m.content)[:40] else "ok"
        seq.append(("OBSERVE", m.name, f"{status}: {str(m.content)[:110]}"))
display(pd.DataFrame(seq, columns=["phase", "tool", "detail"]))
order = [c["name"] for m in session.messages if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
print("Tool order chosen by the agent:", " -> ".join(order))"""),
    code("""assert session.report is not None, "report extraction failed"
display(Markdown(report_markdown(session.report)))"""),

    md("""---
# 3B: Multi-agent coordination
| Agent | Tools (enforced) | Output |
|---|---|---|
| A · Data Analyst | `get_price_data`, `calculate_volatility`, `llm_sentiment` | `DataBrief` (Pydantic) |
| B · Research Writer | `web_search`, `get_news` | `FinalReport` (Pydantic) |

**Why neither agent can do it alone:** A has no news source, so it can't score sentiment until B supplies headlines. B has no price tools, so every number in its report must come from A. The **critique loop** connects them: B sends a `ClarificationRequest` (with the headlines B collected), A answers with a `ClarificationResponse` computed with A's tools, and B incorporates it.

**Handoff integrity:** the numeric fields of `DataBrief` and `ClarificationResponse` are copied from tool JSON (`snapshots_from_tools`). The LLM writes only the narrative fields, so no figure passed between agents is re-typed by a model.

## 3B.1 Tool restriction is enforced at execution time, not only at binding time"""),
    code("""from langchain_core.messages import AIMessage as _AI
from research_agents.agent import ToolExecutor
TRACER.set_agent("data_analyst")
forged = _AI(content="", tool_calls=[{"name": "web_search", "args": {"query": "MSFT"}, "id": "forged-1"}])
denied = ToolExecutor("data_analyst")({"messages": [forged], "steps": 1})["messages"][0]
print("Agent A attempting web_search ->", denied.content)
TRACER.set_agent("research_writer")
forged = _AI(content="", tool_calls=[{"name": "get_price_data", "args": {"ticker": "MSFT"}, "id": "forged-2"}])
print("Agent B attempting get_price_data ->", ToolExecutor("research_writer")({"messages": [forged], "steps": 1})["messages"][0].content)"""),
    md("## 3B.2 End-to-end pipeline: query → final report with no manual steps (full message trace)"),
    code("""from research_agents.multi_agent import run_pipeline
t0 = time.time()
state = run_pipeline(TICKER, use_cache=False)     # force a fresh run; this run writes the cache
print(f"\\nPipeline finished in {time.time() - t0:.0f} s, no manual intervention.")"""),
    code("""final = state["final_report"]
req, resp = state["clarification_request"], state["clarification_response"]
print("Critique loop executed:", req is not None and resp is not None)
print("  B asked :", req.question)
print("  A answer:", resp.answer_summary)
display(Markdown(report_markdown(final)))"""),

    md("""---
# 3C: Memory and observability
## 3C.1 Short-term memory
The single agent's graph uses a LangGraph `InMemorySaver` checkpointer keyed by `thread_id`, so earlier tool observations stay in context. We ask a follow-up and count tool calls made *during the follow-up*."""),
    code("""from research_agents.research import follow_up
TRACER.new_session()
n_trace_before = sum(1 for _ in config.TRACE_PATH.open(encoding="utf-8"))
answer, n_tools = follow_up(session, "Follow-up: what 30-day annualised volatility and beta did you find, "
                                     "and is current volatility above or below its 1-year level?")
n_trace_after = sum(1 for _ in config.TRACE_PATH.open(encoding="utf-8"))
display(Markdown(answer))
print(f"Tool calls during follow-up: {n_tools} | new trace lines: {n_trace_after - n_trace_before}")
assert n_tools == 0, 'follow-up should be answered from context'"""),
    md("## 3C.2 Persistent memory: the second run loads the cached brief instead of re-running tools"),
    code("""from research_agents import memory
print("Cached file:", memory.cache_path(TICKER).name, "exists:", memory.cache_path(TICKER).exists())
n_before = sum(1 for _ in config.TRACE_PATH.open(encoding="utf-8"))
t0 = time.time()
state2 = run_pipeline(TICKER)          # use_cache=True (default)
n_after = sum(1 for _ in config.TRACE_PATH.open(encoding="utf-8"))
new_records = [json.loads(l) for l in config.TRACE_PATH.read_text(encoding="utf-8").splitlines()[n_before:]]
print(f"cache_hit={state2['cache_hit']} in {time.time() - t0:.2f} s; tool calls in second run: "
      f"{sum(r['event'] == 'tool_call' for r in new_records)}")
assert state2["cache_hit"] and state2["final_report"] == final"""),
    md("## 3C.3 `agent_trace.jsonl`: one record per tool call (tool, inputs, output ≤ 200 chars, duration)"),
    code("""trace = pd.DataFrame([json.loads(l) for l in config.TRACE_PATH.read_text(encoding="utf-8").splitlines()])
calls = trace[trace.event == "tool_call"]
print(f"{len(trace)} records ({len(calls)} tool calls) in {config.TRACE_PATH.relative_to(Path.cwd())}")
print("max output length:", calls["output"].str.len().max(), "chars")
display(calls[["ts", "agent", "tool", "inputs", "status", "duration_ms", "output"]].tail(12))
display(calls.groupby(["agent", "tool"]).agg(calls=("tool", "size"), mean_ms=("duration_ms", "mean"),
                                            errors=("status", lambda s: int((s != "ok").sum()))).round(0))"""),
    md("""---
## Bonus: observability dashboard
`streamlit run task3_agentic/dashboard/app.py` opens a viewer for `agent_trace.jsonl` with KPIs, latency per tool, a filterable tool-call timeline, and expandable handoff payloads. A screenshot is in the task README."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "Task3_Multi_Agent_Research.ipynb"
nbf.write(nb, out)
print("wrote", out)
