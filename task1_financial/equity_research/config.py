"""Central configuration: every tunable number in the pipeline lives here.

Keeping constants in one module means there are no "magic numbers" scattered
through the business logic, and a reviewer can see every assumption at a glance.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1 of the CDAZZDEV Senior MLE
# assessment - centralise all pipeline constants in a config module', Date: 2026-10-06

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
TASK_DIR = PACKAGE_DIR.parent
OUTPUT_DIR = TASK_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"

# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
DEFAULT_TICKER = os.getenv("EQUITY_TICKER", "MSFT")

# The brief requires >= 2 years of daily OHLCV. We fetch an extra warm-up window
# so that the 200-day SMA is already valid on the first day of the 2-year window.
LOOKBACK_YEARS = 2
CALENDAR_DAYS_PER_YEAR = 365
INDICATOR_WARMUP_CALENDAR_DAYS = 300  # ~205 trading days > 200-day SMA window
MIN_REQUIRED_TRADING_DAYS = 2 * 252   # sanity floor for "two years" of sessions

TRADING_DAYS_PER_YEAR = 252           # used for 52-week window & annualising vol
FETCH_MAX_RETRIES = 3
FETCH_RETRY_BACKOFF_SECONDS = 2.0

# --------------------------------------------------------------------------- #
# Technical indicator parameters (as specified in the assessment brief)
# --------------------------------------------------------------------------- #
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2.0

# RSI interpretation bands (Wilder's classic levels)
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
# Neutral value returned when there is no price movement at all in the window
RSI_NEUTRAL = 50.0

# Bollinger %B interpretation
PERCENT_B_UPPER = 1.0
PERCENT_B_LOWER = 0.0

# Look-back used to measure the slope of the MACD histogram (momentum acceleration)
MACD_HIST_SLOPE_LOOKBACK = 5
# Look-back for the Bollinger bandwidth percentile (squeeze detection)
BANDWIDTH_PERCENTILE_LOOKBACK = 126   # ~6 months of sessions
# A golden/death cross is "recent" if it happened within this many sessions
RECENT_CROSS_LOOKBACK = 20

# --------------------------------------------------------------------------- #
# Momentum signal (rule-based composite derived from the indicators)
# --------------------------------------------------------------------------- #
# Each component votes in [-1, +1]; weights express how much we trust each vote.
MOMENTUM_WEIGHTS = {
    "trend_long": 0.25,     # price vs 200-day SMA (primary trend)
    "trend_short": 0.15,    # price vs 50-day SMA (intermediate trend)
    "ma_cross": 0.15,       # 50-day vs 200-day SMA (golden / death cross regime)
    "macd": 0.25,           # MACD line vs signal line + histogram direction
    "rsi": 0.10,            # RSI zone
    "bollinger": 0.10,      # position inside the Bollinger envelope
}
MOMENTUM_STRONG_THRESHOLD = 0.5
MOMENTUM_WEAK_THRESHOLD = 0.15

# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
MIN_HEADLINES = 10
MAX_HEADLINES = 15
NEWS_HTTP_TIMEOUT_SECONDS = 15
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
HTTP_USER_AGENT = "Mozilla/5.0 (compatible; CDAZZDEV-EquityResearch/1.0)"

# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
# Both providers expose an OpenAI-compatible Chat Completions API, so a single
# client implementation serves both. Keys are ONLY read from the environment.
LLM_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        # Free-tier catalogues change; if the preferred model has been retired the
        # client picks the first of these that the key can actually access.
        "fallback_models": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b",
                            "llama-3.3-70b-versatile"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
        "fallback_models": ["openai/gpt-oss-120b:free", "meta-llama/llama-3.3-70b-instruct:free",
                            "qwen/qwen3-235b-a22b:free"],
    },
}
LLM_PROVIDER_PRIORITY = ("groq", "openrouter")

# Low temperature: classification and analysis should be repeatable, not creative.
SENTIMENT_TEMPERATURE = 0.0
SIGNAL_TEMPERATURE = 0.2
# gpt-oss models are reasoning models: max_tokens covers hidden reasoning AND the
# JSON answer, so the budgets are larger than the visible output alone needs.
SENTIMENT_MAX_TOKENS = 1024
SIGNAL_MAX_TOKENS = 4096
LLM_REASONING_EFFORT = "medium"       # sent only to reasoning-capable models (gpt-oss)
LLM_MAX_VALIDATION_RETRIES = 2        # re-prompts after a schema failure
LLM_MAX_TRANSPORT_RETRIES = 4         # retries on 429 / 5xx / timeouts
LLM_RETRY_BASE_DELAY_SECONDS = 2.0
LLM_REQUEST_TIMEOUT_SECONDS = 60
LLM_MIN_SECONDS_BETWEEN_CALLS = 2.1   # stays under Groq free tier's 30 req/min

# --------------------------------------------------------------------------- #
# Sentiment aggregation
# --------------------------------------------------------------------------- #
SENTIMENT_POLARITY = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
# Newer headlines matter more: weight halves every N days of age.
SENTIMENT_RECENCY_HALF_LIFE_DAYS = 3.0
SENTIMENT_UNKNOWN_DATE_WEIGHT = 0.5   # undated headlines count as "a few days old"
SENTIMENT_LABEL_THRESHOLD = 0.15      # |score| below this is "neutral"

# --------------------------------------------------------------------------- #
# Signal justification constraints (from the brief)
# --------------------------------------------------------------------------- #
JUSTIFICATION_MIN_SENTENCES = 3
JUSTIFICATION_MAX_SENTENCES = 5

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
REPORT_CHART_LOOKBACK_SESSIONS = 252  # chart the last ~12 months
REPORT_TOP_HEADLINES = 3
CHART_DPI = 110
