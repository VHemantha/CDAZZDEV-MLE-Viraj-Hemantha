"""Summary dictionary, rule-based momentum signal and LLM-ready technical snapshot."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build a summary dict (price, 52w range, P/E,
# YTD return) and a weighted multi-indicator momentum signal with None-safe handling',
# Date: 2026-10-06

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .data import safe_float

SMA_S = f"sma_{config.SMA_SHORT_WINDOW}"
SMA_L = f"sma_{config.SMA_LONG_WINDOW}"
RSI_COL = f"rsi_{config.RSI_PERIOD}"


def _last(df: pd.DataFrame, col: str) -> float | None:
    """Last non-null value of a column, or None."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return safe_float(s.iloc[-1]) if not s.empty else None


def _pct(a: float | None, b: float | None) -> float | None:
    """Percentage change from b to a, None-safe."""
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1.0) * 100.0


def _round(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


# --------------------------------------------------------------------------- #
# Fundamentals-style fields
# --------------------------------------------------------------------------- #
def fifty_two_week_range(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """52-week high/low from the intraday High/Low of the last 252 sessions."""
    window = df.tail(config.TRADING_DAYS_PER_YEAR)
    if window.empty:
        return None, None
    return safe_float(window["High"].max()), safe_float(window["Low"].min())


def ytd_return_pct(df: pd.DataFrame) -> float | None:
    """Year-to-date return: last close vs the final close of the previous year.

    If the history does not reach back into the prior year we fall back to the
    first close of the current year (and the caller sees the same field name).
    """
    if df.empty:
        return None
    last_date = df.index[-1]
    last_close = safe_float(df["Close"].iloc[-1])
    prior = df.loc[df.index.year < last_date.year, "Close"]
    if not prior.empty:
        base = safe_float(prior.iloc[-1])
    else:
        this_year = df.loc[df.index.year == last_date.year, "Close"]
        base = safe_float(this_year.iloc[0]) if not this_year.empty else None
    return _pct(last_close, base)


def pe_ratio(info: dict[str, Any], price: float | None) -> tuple[float | None, str]:
    """Trailing P/E where available, otherwise derived or forward P/E.

    Returns (value, source) so the report can say *which* P/E it shows.
    Negative earnings make P/E meaningless, so those return None.
    """
    trailing = safe_float(info.get("trailingPE"))
    if trailing is not None and trailing > 0:
        return trailing, "trailing (yfinance)"
    eps = safe_float(info.get("trailingEps"))
    if price is not None and eps is not None and eps > 0:
        return price / eps, "trailing (price / trailing EPS)"
    forward = safe_float(info.get("forwardPE"))
    if forward is not None and forward > 0:
        return forward, "forward (yfinance)"
    return None, "unavailable"


# --------------------------------------------------------------------------- #
# Momentum signal
# --------------------------------------------------------------------------- #
def _sign_vote(a: float | None, b: float | None) -> float | None:
    """+1 if a > b, -1 if a < b, None if either is unknown."""
    if a is None or b is None:
        return None
    return float(np.sign(a - b))


def momentum_components(df: pd.DataFrame) -> dict[str, float | None]:
    """Each component is a vote in [-1, +1] (None if the indicator is unavailable)."""
    close = _last(df, "Close")
    sma_s, sma_l = _last(df, SMA_S), _last(df, SMA_L)
    macd_v, macd_sig = _last(df, "macd"), _last(df, "macd_signal")
    hist = df["macd_hist"].dropna() if "macd_hist" in df else pd.Series(dtype=float)
    rsi_v = _last(df, RSI_COL)
    pct_b = _last(df, "bb_percent_b")

    # MACD: line-vs-signal sets direction; an accelerating histogram strengthens it.
    macd_vote = _sign_vote(macd_v, macd_sig)
    if macd_vote is not None and len(hist) > config.MACD_HIST_SLOPE_LOOKBACK:
        slope = hist.iloc[-1] - hist.iloc[-1 - config.MACD_HIST_SLOPE_LOOKBACK]
        macd_vote = macd_vote if np.sign(slope) == macd_vote else macd_vote * 0.5

    # RSI: bullish momentum above 50, but overbought/oversold readings are
    # treated as exhaustion risk (mean-reversion), so they vote *against* trend.
    rsi_vote: float | None = None
    if rsi_v is not None:
        if rsi_v >= config.RSI_OVERBOUGHT:
            rsi_vote = -0.5
        elif rsi_v <= config.RSI_OVERSOLD:
            rsi_vote = 0.5
        else:
            rsi_vote = (rsi_v - config.RSI_NEUTRAL) / (config.RSI_OVERBOUGHT - config.RSI_NEUTRAL)

    # Bollinger %B: upper half = buyers in control; outside the bands = stretched.
    bb_vote: float | None = None
    if pct_b is not None:
        if pct_b > config.PERCENT_B_UPPER or pct_b < config.PERCENT_B_LOWER:
            bb_vote = -0.5 if pct_b > config.PERCENT_B_UPPER else 0.5
        else:
            bb_vote = (pct_b - 0.5) * 2.0

    return {
        "trend_long": _sign_vote(close, sma_l),
        "trend_short": _sign_vote(close, sma_s),
        "ma_cross": _sign_vote(sma_s, sma_l),
        "macd": macd_vote,
        "rsi": rsi_vote,
        "bollinger": bb_vote,
    }


def momentum_signal(df: pd.DataFrame) -> dict[str, Any]:
    """Weighted composite of the component votes -> label + score in [-1, 1].

    Missing components are excluded and the remaining weights renormalised,
    so a short history degrades the signal gracefully instead of crashing.
    """
    comps = momentum_components(df)
    available = {k: v for k, v in comps.items() if v is not None}
    if not available:
        return {"label": "Insufficient data", "score": None, "components": comps}
    total_w = sum(config.MOMENTUM_WEIGHTS[k] for k in available)
    score = sum(config.MOMENTUM_WEIGHTS[k] * v for k, v in available.items()) / total_w

    if score >= config.MOMENTUM_STRONG_THRESHOLD:
        label = "Strong Bullish"
    elif score >= config.MOMENTUM_WEAK_THRESHOLD:
        label = "Bullish"
    elif score <= -config.MOMENTUM_STRONG_THRESHOLD:
        label = "Strong Bearish"
    elif score <= -config.MOMENTUM_WEAK_THRESHOLD:
        label = "Bearish"
    else:
        label = "Neutral"
    return {
        "label": label,
        "score": round(score, 3),
        "components": {k: _round(v, 3) for k, v in comps.items()},
    }


# --------------------------------------------------------------------------- #
# Technical snapshot handed to the LLM
# --------------------------------------------------------------------------- #
def _sessions_since_cross(df: pd.DataFrame) -> tuple[str | None, int | None]:
    """Most recent 50/200 SMA crossover and how many sessions ago it happened."""
    both = df[[SMA_S, SMA_L]].dropna()
    if len(both) < 2:
        return None, None
    regime = np.sign(both[SMA_S] - both[SMA_L])
    changes = regime.ne(regime.shift()) & regime.shift().notna()
    if not changes.any():
        return None, None
    last_change = changes[changes].index[-1]
    kind = "golden_cross" if regime.loc[last_change] > 0 else "death_cross"
    return kind, int(len(both.loc[last_change:]) - 1)


def technical_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """Relational features (not just raw levels) so the LLM can reason over
    *combinations*: distances to the averages, MACD acceleration, squeeze
    state, realised volatility, recent returns and cross history."""
    close = _last(df, "Close")
    sma_s, sma_l = _last(df, SMA_S), _last(df, SMA_L)
    hist = df["macd_hist"].dropna()
    bw = df["bb_bandwidth"].dropna()
    log_ret = np.log(df["Close"]).diff().dropna()
    cross_kind, cross_age = _sessions_since_cross(df)

    hist_slope = None
    if len(hist) > config.MACD_HIST_SLOPE_LOOKBACK:
        hist_slope = safe_float(hist.iloc[-1] - hist.iloc[-1 - config.MACD_HIST_SLOPE_LOOKBACK])

    bw_pctile = None
    if len(bw) >= config.BANDWIDTH_PERCENTILE_LOOKBACK:
        recent = bw.tail(config.BANDWIDTH_PERCENTILE_LOOKBACK)
        bw_pctile = safe_float((recent <= recent.iloc[-1]).mean() * 100.0)

    def ret(n: int) -> float | None:
        return _pct(close, safe_float(df["Close"].iloc[-1 - n])) if len(df) > n else None

    vol_20d = None
    if len(log_ret) >= config.BOLLINGER_WINDOW:
        vol_20d = safe_float(log_ret.tail(config.BOLLINGER_WINDOW).std()
                             * math.sqrt(config.TRADING_DAYS_PER_YEAR) * 100.0)

    return {
        "as_of": df.index[-1].date().isoformat(),
        "close": _round(close),
        f"sma_{config.SMA_SHORT_WINDOW}": _round(sma_s),
        f"sma_{config.SMA_LONG_WINDOW}": _round(sma_l),
        "pct_vs_sma_50": _round(_pct(close, sma_s)),
        "pct_vs_sma_200": _round(_pct(close, sma_l)),
        "sma_50_vs_200_pct": _round(_pct(sma_s, sma_l)),
        "last_ma_cross": cross_kind,
        "sessions_since_ma_cross": cross_age,
        "rsi_14": _round(_last(df, RSI_COL)),
        "macd": _round(_last(df, "macd"), 3),
        "macd_signal": _round(_last(df, "macd_signal"), 3),
        "macd_hist": _round(_last(df, "macd_hist"), 3),
        f"macd_hist_change_{config.MACD_HIST_SLOPE_LOOKBACK}d": _round(hist_slope, 3),
        "bb_upper": _round(_last(df, "bb_upper")),
        "bb_middle": _round(_last(df, "bb_middle")),
        "bb_lower": _round(_last(df, "bb_lower")),
        "bb_percent_b": _round(_last(df, "bb_percent_b"), 3),
        "bb_bandwidth_pct": _round(bw_last * 100.0) if (bw_last := _last(df, "bb_bandwidth")) is not None else None,
        "bb_bandwidth_percentile_6m": _round(bw_pctile, 1),
        "realised_vol_20d_annualised_pct": _round(vol_20d),
        "return_5d_pct": _round(ret(5)),
        "return_20d_pct": _round(ret(20)),
        "return_60d_pct": _round(ret(60)),
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_summary(ticker: str, df: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    """The clean summary dictionary required by Task 1A."""
    price = _last(df, "Close")
    hi52, lo52 = fifty_two_week_range(df)
    pe, pe_source = pe_ratio(info, price)
    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency", "USD"),
        "as_of": df.index[-1].date().isoformat() if not df.empty else None,
        "current_price": _round(price),
        "52_week_high": _round(hi52),
        "52_week_low": _round(lo52),
        "pct_below_52w_high": _round(_pct(price, hi52)),
        "pe_ratio": _round(pe),
        "pe_ratio_source": pe_source,
        "market_cap": safe_float(info.get("marketCap")),
        "ytd_return_pct": _round(ytd_return_pct(df)),
        "momentum_signal": momentum_signal(df),
    }
