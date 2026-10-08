"""Technical indicators implemented from first principles (no TA-Lib).

Every function takes a price ``pd.Series`` and returns a Series/DataFrame aligned
on the same index. Values are ``NaN`` until enough history exists for the window
-- we never back-fill an indicator with a value it could not have known.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement SMA, RSI with Wilder smoothing,
# MACD(12,26,9) and Bollinger Bands(20,2) from first principles with pandas/numpy, no TA-Lib',
# Date: 2026-10-06

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average: arithmetic mean of the last ``window`` closes.

    ``min_periods=window`` ensures the first ``window - 1`` values are NaN
    rather than a misleading average over a shorter history.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    return close.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with smoothing factor alpha = 2 / (span + 1).

    ``adjust=False`` gives the recursive definition used by charting platforms:
        EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
    """
    if span <= 0:
        raise ValueError("span must be positive")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index using J. Welles Wilder's original smoothing.

    Algorithm (Wilder, 1978):
      1. delta_t = close_t - close_{t-1}; split into gains (>0) and losses (>0).
      2. Seed: the first average gain/loss is the *simple* mean of the first
         ``period`` deltas.
      3. Thereafter: avg_t = (avg_{t-1} * (period - 1) + value_t) / period
         (equivalent to an EMA with alpha = 1/period).
      4. RS = avg_gain / avg_loss;  RSI = 100 - 100 / (1 + RS).

    Edge cases: avg_loss == 0 -> RSI 100 (only up moves); both averages zero
    (flat price) -> neutral 50. NaN prices are dropped before the recursion and
    the result is re-aligned to the original index.
    """
    if period <= 0:
        raise ValueError("period must be positive")

    clean = close.dropna()
    result = pd.Series(np.nan, index=close.index, dtype=float)
    if len(clean) <= period:
        return result  # not enough data for even the seed value

    delta = clean.diff().to_numpy()[1:]           # first diff is NaN by definition
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.empty(len(delta))
    avg_loss = np.empty(len(delta))
    avg_gain[: period - 1] = np.nan
    avg_loss[: period - 1] = np.nan
    avg_gain[period - 1] = gains[:period].mean()  # step 2: SMA seed
    avg_loss[period - 1] = losses[:period].mean()
    for i in range(period, len(delta)):            # step 3: Wilder recursion
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        values = 100.0 - 100.0 / (1.0 + rs)
    values = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, values)
    values = np.where((avg_loss == 0) & (avg_gain == 0), config.RSI_NEUTRAL, values)

    # delta[i] corresponds to clean.index[i + 1]
    result.loc[clean.index[1:]] = values
    return result


def macd(
    close: pd.Series,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (Gerald Appel).

    macd_line   = EMA_fast(close) - EMA_slow(close)
    signal_line = EMA_signal(macd_line)
    histogram   = macd_line - signal_line
    """
    if fast >= slow:
        raise ValueError("fast span must be shorter than slow span")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line.dropna(), signal).reindex(close.index)
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        },
        index=close.index,
    )


def bollinger_bands(
    close: pd.Series,
    window: int = config.BOLLINGER_WINDOW,
    num_std: float = config.BOLLINGER_NUM_STD,
) -> pd.DataFrame:
    """Bollinger Bands (John Bollinger).

    middle = SMA_window(close); upper/lower = middle +/- num_std * sigma.
    sigma uses the *population* standard deviation (ddof=0), as in Bollinger's
    own definition -- the bands describe the window itself, not a sample
    estimate of a wider population.

    Also returns two derived features the LLM can reason over:
      %B        = (close - lower) / (upper - lower)  (position in the envelope)
      bandwidth = (upper - lower) / middle           (volatility / squeeze gauge)
    """
    middle = sma(close, window)
    sigma = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * sigma
    lower = middle - num_std * sigma
    width = (upper - lower).replace(0.0, np.nan)   # flat prices -> undefined %B
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_percent_b": (close - lower) / width,
            "bb_bandwidth": (upper - lower) / middle,
        },
        index=close.index,
    )


def add_all_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``ohlcv`` with every indicator column appended."""
    if "Close" not in ohlcv.columns:
        raise KeyError("OHLCV frame must contain a 'Close' column")
    out = ohlcv.copy()
    close = out["Close"].astype(float)
    out[f"sma_{config.SMA_SHORT_WINDOW}"] = sma(close, config.SMA_SHORT_WINDOW)
    out[f"sma_{config.SMA_LONG_WINDOW}"] = sma(close, config.SMA_LONG_WINDOW)
    out[f"rsi_{config.RSI_PERIOD}"] = rsi(close, config.RSI_PERIOD)
    out = out.join(macd(close))
    out = out.join(bollinger_bands(close))
    return out
