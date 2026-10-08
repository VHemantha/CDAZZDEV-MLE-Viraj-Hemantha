"""The five agent tools. Each one:
  * never raises -- failures come back as {"error": ..., "hint": ...} or an empty
    list, so the agent can observe the failure and choose an alternative;
  * is traced (inputs, truncated output, wall-clock duration, status);
  * returns compact, JSON-serialisable data (token budget on the free tier).

Indicators and news retrieval reuse the Task 1 package (already unit-tested)
instead of re-implementing them.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement five traced LangChain tools
# (price data, news, volatility, LLM sentiment, DuckDuckGo search) that never raise',
# Date: 2026-10-07

from __future__ import annotations

import functools
import json
import math
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import config
from .tracing import TRACER, timed

sys.path.insert(0, str(config.REPO_DIR / "task1_financial"))
from equity_research.indicators import add_all_indicators  # noqa: E402
from equity_research.news import fetch_news  # noqa: E402

_fault_budget: dict[str, int] = {}


def set_fault_injection(faults: dict[str, int]) -> None:
    """Make the next n calls of a tool fail with a simulated outage (demo only)."""
    _fault_budget.clear()
    _fault_budget.update(faults)


def _status(result: Any) -> str:
    if isinstance(result, dict) and "error" in result:
        return "error"
    if result in ([], {}, None):
        return "empty"
    return "ok"


def traced(fn: Callable) -> Callable:
    """Tracing + fault injection + last-resort exception guard."""

    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> Any:
        start = timed()
        if _fault_budget.get(fn.__name__, 0) > 0:
            _fault_budget[fn.__name__] -= 1
            result: Any = {"error": f"{fn.__name__} unavailable (simulated provider outage)",
                           "hint": "use an alternative tool or retry with different arguments"}
        else:
            try:
                result = fn(**kwargs)
            except Exception as exc:  # defensive: tools must never crash the agent
                result = {"error": f"{type(exc).__name__}: {exc}"[:300],
                          "hint": "check the arguments or use an alternative tool"}
        TRACER.tool_call(fn.__name__, kwargs, result, (timed() - start) * 1000, _status(result))
        return result

    return wrapper


def _r(x: Any, nd: int = 2) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _history(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise LookupError(f"no price data for '{ticker}' (period {period})")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df.dropna(subset=["Close"])


# --------------------------------------------------------------------------- #
# 1. get_price_data
# --------------------------------------------------------------------------- #
@traced
def get_price_data(ticker: str, period: str = "1y") -> dict:
    if period not in config.VALID_PERIODS:
        return {"error": f"invalid period '{period}'", "hint": f"use one of {config.VALID_PERIODS}"}
    # Fetch >= 2y so the 200-day SMA is defined, then report statistics on `period` only.
    full = _history(ticker.upper(), "5y" if period == "5y" else "2y")
    df = add_all_indicators(full)
    window = df.loc[df.index >= df.index[-1] - pd.Timedelta(days=config.PERIOD_DAYS[period])]
    last = df.iloc[-1]
    close = window["Close"]
    drawdown = (close / close.cummax() - 1).min() * 100
    recent = df.tail(config.RECENT_BARS_RETURNED)[["Open", "High", "Low", "Close", "Volume"]]
    return {
        "ticker": ticker.upper(), "period": period, "n_bars": int(len(window)),
        "start": str(window.index[0].date()), "end": str(window.index[-1].date()),
        "last_close": _r(last["Close"]),
        "period_return_pct": _r((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "high_in_period": _r(window["High"].max()), "low_in_period": _r(window["Low"].min()),
        "max_drawdown_pct": _r(drawdown),
        "indicators": {
            "sma_50": _r(last["sma_50"]), "sma_200": _r(last["sma_200"]),
            "pct_vs_sma_50": _r((last["Close"] / last["sma_50"] - 1) * 100),
            "pct_vs_sma_200": _r((last["Close"] / last["sma_200"] - 1) * 100),
            "rsi_14": _r(last["rsi_14"]), "macd": _r(last["macd"], 3),
            "macd_signal": _r(last["macd_signal"], 3), "macd_hist": _r(last["macd_hist"], 3),
            "bb_percent_b": _r(last["bb_percent_b"], 3),
        },
        "recent_ohlcv": [{"date": str(i.date()), **{k: _r(v) for k, v in row.items()}}
                         for i, row in recent.iterrows()],
    }


# --------------------------------------------------------------------------- #
# 2. get_news
# --------------------------------------------------------------------------- #
@traced
def get_news(ticker: str, n: int = 10) -> list[dict]:
    n = max(1, min(int(n), config.MAX_HEADLINES))
    try:
        name = yf.Ticker(ticker).info.get("shortName")
    except Exception:
        name = None
    items, _ = fetch_news(ticker.upper(), name, n=n)
    return [{"headline": it.headline, "publisher": it.publisher,
             "published_at": it.published_at.isoformat() if it.published_at else None}
            for it in items]


# --------------------------------------------------------------------------- #
# 3. calculate_volatility
# --------------------------------------------------------------------------- #
@traced
def calculate_volatility(ticker: str, window: int = 30) -> dict:
    """Annualised close-to-close volatility: std(log returns over `window`) * sqrt(252).
    Also returns the 1-year level and percentile (regime), 1-day 95% historical VaR,
    and beta vs SPY -- the inputs a hedge needs for sizing."""
    window = int(window)
    if not 5 <= window <= config.TRADING_DAYS:
        return {"error": f"window {window} out of range", "hint": "use 5-252 trading days"}
    px = _history(ticker.upper(), "2y")["Close"]
    r = np.log(px).diff().dropna()
    if len(r) < window + 1:
        return {"error": "not enough history", "hint": "use a smaller window"}
    ann = math.sqrt(config.TRADING_DAYS)
    rolling = r.rolling(window).std() * ann * 100
    last_year = rolling.tail(config.TRADING_DAYS).dropna()
    out = {
        "ticker": ticker.upper(), "window_days": window,
        "annualised_vol_pct": _r(rolling.iloc[-1]),
        "one_year_vol_pct": _r(r.tail(config.TRADING_DAYS).std() * ann * 100),
        "vol_percentile_1y": _r((last_year <= last_year.iloc[-1]).mean() * 100, 1),
        "var_95_1d_pct": _r(-np.percentile(r.tail(config.TRADING_DAYS), (1 - config.VAR_CONFIDENCE) * 100) * 100),
    }
    try:  # beta is optional; never fail the tool because the benchmark failed
        bench = np.log(_history(config.BENCHMARK_TICKER, "2y")["Close"]).diff().dropna()
        joined = pd.concat([r, bench], axis=1, join="inner").tail(config.TRADING_DAYS)
        out["beta_vs_spy"] = _r(joined.cov().iloc[0, 1] / joined.iloc[:, 1].var())
    except Exception:
        out["beta_vs_spy"] = None
    return out


# --------------------------------------------------------------------------- #
# 4. llm_sentiment
# --------------------------------------------------------------------------- #
class _HeadlineScore(BaseModel):
    headline: str
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")
    confidence: float = Field(ge=0, le=1)


class _SentimentBatch(BaseModel):
    scores: list[_HeadlineScore]


SENTIMENT_SYSTEM = (
    "You classify financial news headlines by their likely short-term impact on the named "
    "company's share price. Headlines mainly about other companies or generic market commentary "
    "are neutral. confidence = probability the label is correct.")


@traced
def llm_sentiment(headlines: list[str]) -> dict:
    """One batched LLM call, Pydantic-validated. Score = confidence-weighted mean
    polarity in [-1, 1] (neutral headlines dilute the score)."""
    from .llm import make_chat, structured_invoke  # local import: avoids a cycle at import time

    headlines = [h for h in (headlines or []) if isinstance(h, str) and h.strip()][: config.MAX_HEADLINES]
    if not headlines:
        return {"error": "no headlines supplied", "hint": "fetch headlines with get_news or web_search first"}
    batch = structured_invoke(
        make_chat("sentiment_tool", json_mode=True), SENTIMENT_SYSTEM,
        "Headlines:\n" + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines)),
        _SentimentBatch, "Score every headline, in the same order.")
    pol = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    w = sum(s.confidence for s in batch.scores)
    score = sum(pol[s.sentiment] * s.confidence for s in batch.scores) / w if w else 0.0
    counts = {k: sum(s.sentiment == k for s in batch.scores) for k in pol}
    return {"score": round(score, 3),
            "label": "positive" if score >= 0.15 else "negative" if score <= -0.15 else "neutral",
            "n_headlines": len(batch.scores), "counts": counts,
            "most_negative": [s.headline for s in batch.scores if s.sentiment == "negative"][:3],
            "most_positive": [s.headline for s in batch.scores if s.sentiment == "positive"][:3]}


# --------------------------------------------------------------------------- #
# 5. web_search
# --------------------------------------------------------------------------- #
@traced
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo text search; falls back to DuckDuckGo News if text search is empty."""
    from ddgs import DDGS

    k = max(1, min(int(max_results), config.MAX_SEARCH_RESULTS))
    with DDGS() as ddg:
        rows = list(ddg.text(query, max_results=k) or [])
        if not rows:
            rows = list(ddg.news(query, max_results=k) or [])
    return [{"title": r.get("title"), "snippet": (r.get("body") or r.get("excerpt") or "")[:300],
             "url": r.get("href") or r.get("url"), "date": r.get("date")} for r in rows]


# --------------------------------------------------------------------------- #
# LangChain tool objects + role-based access control
# --------------------------------------------------------------------------- #
class _PriceArgs(BaseModel):
    ticker: str = Field(description="Stock ticker, e.g. MSFT")
    period: str = Field(default="1y", description=f"one of {config.VALID_PERIODS}")


class _NewsArgs(BaseModel):
    ticker: str
    n: int = Field(default=10, description="number of headlines (max 15)")


class _VolArgs(BaseModel):
    ticker: str
    window: int = Field(default=30, description="rolling window in trading days (5-252)")


class _SentArgs(BaseModel):
    headlines: list[str] = Field(description="headline strings to score")


class _SearchArgs(BaseModel):
    query: str = Field(description="web search query, e.g. 'MSFT analyst downgrade risks'")
    max_results: int = Field(default=5)


def _tool(fn: Callable, args: type[BaseModel], desc: str) -> StructuredTool:
    def run(**kwargs: Any) -> str:
        out = json.dumps(fn(**kwargs), default=str, ensure_ascii=False)
        return out if len(out) <= config.TOOL_RESULT_MAX_CHARS else out[: config.TOOL_RESULT_MAX_CHARS] + "…(truncated)"
    return StructuredTool.from_function(func=run, name=fn.__name__, description=desc, args_schema=args)


ALL_TOOLS: dict[str, StructuredTool] = {
    "get_price_data": _tool(get_price_data, _PriceArgs,
        "Daily OHLCV summary with computed indicators (SMA50/200, RSI14, MACD, Bollinger %B), "
        "period return, max drawdown and the last 5 bars."),
    "get_news": _tool(get_news, _NewsArgs, "Recent news headlines for a ticker as a structured list."),
    "calculate_volatility": _tool(calculate_volatility, _VolArgs,
        "Annualised historical volatility for a rolling window, plus 1y level, percentile, "
        "1-day 95% VaR and beta vs SPY."),
    "llm_sentiment": _tool(llm_sentiment, _SentArgs,
        "Score headlines with an LLM; returns overall sentiment score in [-1,1], label and counts."),
    "web_search": _tool(web_search, _SearchArgs,
        "DuckDuckGo web search for analyst commentary, risks, events. Returns titles/snippets/urls."),
}

ROLE_TOOLS = {
    "research_agent": list(ALL_TOOLS),
    "data_analyst": ["get_price_data", "calculate_volatility", "llm_sentiment"],
    "research_writer": ["web_search", "get_news"],
}
