# Task 3: Agentic Workflows, Multi-Agent Financial Research System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName/blob/main/task3_agentic/notebooks/Task3_Multi_Agent_Research.ipynb)

**Notebook (executed outputs):** [`notebooks/Task3_Multi_Agent_Research.ipynb`](notebooks/Task3_Multi_Agent_Research.ipynb) · **Trace:** [`logs/agent_trace.jsonl`](logs/agent_trace.jsonl) · **Cached brief:** [`cache/`](cache/)

Framework: **LangGraph** (explicit `StateGraph`s). LLMs: Groq free tier. Each role prefers `openai/gpt-oss-120b` and automatically fails over to `openai/gpt-oss-20b` on rate-limit or daily-quota errors. The trace records which model answered each step.

## Architecture

```
 3A  single agent (all 5 tools)            3B  pipeline (LangGraph StateGraph)
 ┌────────┐  tool_calls  ┌────────────┐     cache_check ─hit──────────────────────────────► END
 │ agent  │─────────────►│ ToolExec   │        │miss
 │ (LLM)  │◄─────────────│ allowlist  │     analyst_brief ──DataBrief──► writer_research
 └────────┘ observations │ + tracing  │       (Agent A)                    (Agent B)
     │ no tool_calls     └────────────┘                                        │
     ▼                                       analyst_clarify ◄─ClarificationRequest─ writer_critique
 ResearchReport (Pydantic, validated)           │ ClarificationResponse
                                             writer_final ──FinalReport──► save (JSON cache) ──► END
```

| File | Responsibility |
|---|---|
| `tools.py` | The 5 tools. Each is `@traced` (JSONL record + timing), never raises (`{"error", "hint"}` comes back as an observation), returns compact JSON, and supports fault injection for demos. Indicators and news reuse the unit-tested Task 1 package. |
| `agent.py` | The ReAct loop as a `StateGraph`. **The tool allowlist is enforced at execution time** (a forged call to a forbidden tool is denied and traced). Also: a step budget, recovery from malformed tool calls (a gpt-oss quirk), and a reasoning trace. |
| `llm.py` | `ReasoningChatOpenAI` keeps the provider's reasoning summary for the trace. Model fallback chain. JSON + Pydantic structured output with repair turns. |
| `schemas.py` | `ResearchReport`, `DataBrief`, `ClarificationRequest/Response`, `FinalReport`. A validator requires each evidence item to cite its source tool and a figure or quote. |
| `multi_agent.py` | The 3B pipeline. **Handoff figures are copied from tool JSON, never re-typed by an LLM.** |
| `research.py` | The 3A run and short-term-memory follow-ups (`InMemorySaver` checkpointer per `thread_id`). |
| `memory.py` | Persistent cache `cache/{TICKER}_{YYYY-MM-DD}.json`. |
| `tracing.py` | `logs/agent_trace.jsonl`: `ts, session_id, event, agent, tool, inputs, output (≤200 chars), duration_ms, status`, plus handoff and cache-hit events. |
| `dashboard/app.py` | Streamlit trace viewer (bonus). |

## How each requirement is met

| Requirement | Evidence in the notebook |
|---|---|
| Five tools callable, correct types | §3A.1 table and failure-path cell |
| Autonomous tool selection | No sequence is coded. The order is printed from the agent's own tool calls (§3A.2). |
| Observe → replan cycle | A simulated `get_news` outage → `✗` observation → the THINK line picks `web_search` → `llm_sentiment` on those results (§3A.2) |
| Three-section report with evidence | Rendered `ResearchReport` (§3A.2). Evidence items are validated to cite tool + figure. |
| Error handling | Tools never raise. Malformed-call retries, step budget, failover on quota. |
| Distinct roles, enforced restriction | Role allowlists plus forged-call denial demo (§3B.1); `tests/test_agents_offline.py` |
| Structured handoff | `DataBrief`, `ClarificationRequest`, `ClarificationResponse`: Pydantic models printed in full at each handoff |
| Critique loop | B → A request (headlines to score) → A runs `llm_sentiment` → B's `clarification_incorporated` (§3B.2) |
| End-to-end automation | One `run_pipeline()` call, no human steps |
| Short-term memory | Follow-up answered with **0 tool calls**, asserted (§3C.1) |
| Persistent cache | Second run: `cache_hit=True`, 0 tool calls, asserted (§3C.2) |
| `agent_trace.jsonl` | `logs/agent_trace.jsonl`, summarised in §3C.3 |

## Run

```bash
pip install -r task3_agentic/requirements.txt
# GROQ_API_KEY in .env (repo root) or Colab secrets
cd task3_agentic && python -m pytest -q          # offline tests
jupyter nbconvert --to notebook --execute --inplace notebooks/Task3_Multi_Agent_Research.ipynb
streamlit run dashboard/app.py                   # bonus dashboard
```

## Bonus: observability dashboard
![Streamlit trace dashboard](dashboard/screenshot.png)
