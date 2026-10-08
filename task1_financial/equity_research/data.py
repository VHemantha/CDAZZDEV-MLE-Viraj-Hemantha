"""Market-data ingestion via yfinance with retries and defensive cleaning."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write a robust yfinance OHLCV + fundamentals
# fetcher with relative date windows, retries and null handling', Date: 2026-10-06

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from . import config

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataFetchError(RuntimeError):
    """Raised when price history cannot be retrieved after all retries."""


@dataclass
class MarketData:
    """Everything fetched for one ticker. Fundamentals may be partially empty."""

    ticker: str
    ohlcv: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def history_window(today: date | None = None) -> tuple[date, date]:
    """Compute the fetch window relative to *today* -- no hard-coded date strings.

    start = today - (2 years + indicator warm-up), end = tomorrow (yfinance's
    ``end`` is exclusive, so tomorrow guarantees today's bar is included).
    """
    today = today or date.today()
    lookback_days = config.LOOKBACK_YEARS * config.CALENDAR_DAYS_PER_YEAR
    start = today - timedelta(days=lookback_days + config.INDICATOR_WARMUP_CALENDAR_DAYS)
    end = today + timedelta(days=1)
    return start, end


def _clean_ohlcv(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalise the yfinance frame and repair/remove bad rows.

    * keep only OHLCV columns, coerce to numeric, drop the timezone;
    * drop rows with no Close (a bar without a close is unusable);
    * forward-fill isolated gaps in O/H/L from the close, volume -> 0;
    * remove duplicated timestamps and sort chronologically.
    """
    warnings: list[str] = []
    missing_cols = [c for c in OHLCV_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise DataFetchError(f"yfinance response missing columns: {missing_cols}")

    df = raw[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    n_null_close = int(df["Close"].isna().sum())
    if n_null_close:
        warnings.append(f"Dropped {n_null_close} bar(s) with missing Close.")
        df = df.dropna(subset=["Close"])

    for col in ("Open", "High", "Low"):
        n_null = int(df[col].isna().sum())
        if n_null:
            warnings.append(f"Filled {n_null} missing {col} value(s) with Close.")
            df[col] = df[col].fillna(df["Close"])
    n_null_vol = int(df["Volume"].isna().sum())
    if n_null_vol:
        warnings.append(f"Filled {n_null_vol} missing Volume value(s) with 0.")
        df["Volume"] = df["Volume"].fillna(0)

    return df, warnings


def fetch_ohlcv(ticker: str, today: date | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Download daily OHLCV with exponential-backoff retries."""
    start, end = history_window(today)
    last_error: Exception | None = None
    for attempt in range(1, config.FETCH_MAX_RETRIES + 1):
        try:
            raw = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=True,   # split/dividend-adjusted prices for indicators
                actions=False,
            )
            if raw is None or raw.empty:
                raise DataFetchError(f"No price data returned for '{ticker}'.")
            df, warnings = _clean_ohlcv(raw)
            if len(df) < config.MIN_REQUIRED_TRADING_DAYS:
                warnings.append(
                    f"Only {len(df)} sessions available (< {config.MIN_REQUIRED_TRADING_DAYS}); "
                    "long-window indicators may be NaN."
                )
            logger.info("Fetched %d daily bars for %s (%s -> %s)", len(df), ticker,
                        df.index.min().date(), df.index.max().date())
            return df, warnings
        except Exception as exc:  # network errors, rate limits, empty frames
            last_error = exc
            wait = config.FETCH_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.warning("OHLCV fetch attempt %d/%d failed for %s: %s",
                           attempt, config.FETCH_MAX_RETRIES, ticker, exc)
            if attempt < config.FETCH_MAX_RETRIES:
                time.sleep(wait)
    raise DataFetchError(f"Could not fetch OHLCV for '{ticker}': {last_error}") from last_error


def fetch_info(ticker: str) -> tuple[dict[str, Any], list[str]]:
    """Fetch company metadata/fundamentals. Never raises -- returns {} on failure."""
    try:
        info = yf.Ticker(ticker).info or {}
        return dict(info), []
    except Exception as exc:
        logger.warning("Fundamentals unavailable for %s: %s", ticker, exc)
        return {}, [f"Fundamentals unavailable ({type(exc).__name__}); P/E and profile omitted."]


def fetch_market_data(ticker: str = config.DEFAULT_TICKER, today: date | None = None) -> MarketData:
    ticker = ticker.strip().upper()
    ohlcv, warnings = fetch_ohlcv(ticker, today)
    info, info_warnings = fetch_info(ticker)
    return MarketData(ticker=ticker, ohlcv=ohlcv, info=info, warnings=warnings + info_warnings)


def safe_float(value: Any) -> float | None:
    """Convert to a finite float, or None for missing / NaN / inf / non-numeric."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
