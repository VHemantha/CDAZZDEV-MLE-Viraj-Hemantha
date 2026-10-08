"""Task 3 configuration: models per role, tool limits, paths, memory settings."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Config for a LangGraph multi-agent financial
# research system on Groq free tier', Date: 2026-10-07

from __future__ import annotations

import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = TASK_DIR.parent
LOG_DIR = TASK_DIR / "logs"
TRACE_PATH = LOG_DIR / "agent_trace.jsonl"
CACHE_DIR = TASK_DIR / "cache"

# --------------------------------------------------------------------------- #
# LLMs. Groq's free tier limits tokens-per-minute PER MODEL, so roles that run
# back-to-back are spread across models to avoid stalling on one bucket.
# --------------------------------------------------------------------------- #
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY_ENV = "GROQ_API_KEY"
MODELS = {
    "research_agent": os.getenv("T3_AGENT_MODEL", "openai/gpt-oss-120b"),   # 3A single agent
    "data_analyst": os.getenv("T3_ANALYST_MODEL", "openai/gpt-oss-20b"),    # 3B Agent A
    "research_writer": os.getenv("T3_WRITER_MODEL", "openai/gpt-oss-120b"), # 3B Agent B
    "sentiment_tool": os.getenv("T3_SENTIMENT_MODEL", "openai/gpt-oss-20b"),
}
# On RateLimitError (incl. an exhausted daily token quota) a role fails over to this model.
# (qwen3.8-27b was evaluated as a 3rd fallback but its free tier caps OUTPUT at 1K tokens/min,
#  too small for a structured report, so it is not used here.)
FALLBACK_MODELS = [os.getenv("T3_FALLBACK_MODEL", "openai/gpt-oss-20b")]
PRIMARY_MAX_RETRIES = 2          # fail over fast when the preferred model's quota is gone
LLM_TEMPERATURE = 0.1
LLM_REASONING_EFFORT = "low"     # tool-routing decisions don't need long chains of thought
LLM_MAX_RETRIES = 8              # SDK backs off on 429 using the server's retry-after
LLM_TIMEOUT_S = 120
# Groq counts prompt + max_tokens against the 8K tokens/minute budget, so the cap must
# leave room for ~4-5K-token prompts; reports are kept concise by field word limits.
LLM_MAX_OUTPUT_TOKENS = 3000

# --------------------------------------------------------------------------- #
# Agent loop guards
# --------------------------------------------------------------------------- #
MAX_AGENT_STEPS = 10             # model turns per agent run (prevents tool-call loops)
NODE_MAX_ATTEMPTS = 3            # LangGraph RetryPolicy on transient network/provider errors
MALFORMED_TOOL_CALL_RETRIES = 2  # provider 400 'tool_use_failed' (gpt-oss quirk) -> retry with nudge
TOOL_RESULT_MAX_CHARS = 1800     # what the LLM sees per observation (token budget)
KEEP_FULL_OBSERVATIONS = 2       # older observations are compacted in the prompt (see agent.compact_history)
COMPACTED_OBSERVATION_CHARS = 400
TRACE_OUTPUT_MAX_CHARS = 200     # what agent_trace.jsonl stores (brief requirement)

# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
PERIOD_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 365, "2y": 730, "5y": 1826}
VALID_PERIODS = tuple(PERIOD_DAYS)
TRADING_DAYS = 252
VAR_CONFIDENCE = 0.95
BENCHMARK_TICKER = "SPY"         # beta reference for hedge sizing
RECENT_BARS_RETURNED = 5
MAX_HEADLINES = 15
MAX_SEARCH_RESULTS = 6

# Fault injection (demo of graceful degradation): {"tool_name": n_failures}.
# Only enabled explicitly by the notebook; production default is no faults.
FAULT_INJECTION: dict[str, int] = {}

# --------------------------------------------------------------------------- #
# Persistent memory
# --------------------------------------------------------------------------- #
CACHE_MAX_AGE_DAYS = 0           # 0 = valid for the same calendar day only
