"""Indicator correctness: each vectorised implementation is checked against an
independent, deliberately naive pure-Python loop written straight from the
textbook definition, plus edge cases."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest suite validating SMA/RSI/MACD/Bollinger
# against naive reference implementations and edge cases', Date: 2026-10-06

import math

import numpy as np
import pandas as pd
import pytest

from equity_research import indicators as ind


@pytest.fixture
def close() -> pd.Series:
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400)))
    return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=400))


# ---------------- naive references ---------------- #
def ref_sma(x, n):
    return [float("nan") if i < n - 1 else sum(x[i - n + 1 : i + 1]) / n for i in range(len(x))]


def ref_ema(x, span):
    a, out, prev = 2 / (span + 1), [], None
    for i, v in enumerate(x):
        if v != v:  # NaN
            out.append(float("nan"))
            continue
        prev = v if prev is None else a * v + (1 - a) * prev
        out.append(prev)
    return out


def ref_rsi(x, n=14):
    out = [float("nan")] * len(x)
    d = [x[i] - x[i - 1] for i in range(1, len(x))]
    g = [max(v, 0) for v in d]
    l_ = [max(-v, 0) for v in d]
    ag, al = sum(g[:n]) / n, sum(l_[:n]) / n
    out[n] = 100 - 100 / (1 + ag / al) if al else 100.0
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l_[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al) if al else 100.0
    return out


def assert_close(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN warm-up pattern differs"
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=tol, atol=tol)


# ---------------- tests ---------------- #
@pytest.mark.parametrize("n", [5, 50, 200])
def test_sma_matches_reference(close, n):
    assert_close(ind.sma(close, n), ref_sma(close.tolist(), n))


def test_rsi_matches_wilder_reference(close):
    assert_close(ind.rsi(close, 14), ref_rsi(close.tolist(), 14))


def test_rsi_bounds_and_warmup(close):
    r = ind.rsi(close, 14)
    assert r.iloc[:14].isna().all() and not math.isnan(r.iloc[14])
    assert r.dropna().between(0, 100).all()


def test_rsi_edge_cases():
    up = pd.Series(np.arange(1.0, 31.0))
    flat = pd.Series(np.full(30, 10.0))
    assert ind.rsi(up, 14).dropna().eq(100.0).all()
    assert ind.rsi(flat, 14).dropna().eq(50.0).all()
    assert ind.rsi(pd.Series([1.0, 2.0]), 14).isna().all()  # too short -> all NaN, no error


def test_rsi_handles_interior_nan(close):
    c = close.copy()
    c.iloc[100] = np.nan
    r = ind.rsi(c, 14)
    assert math.isnan(r.iloc[100]) and r.iloc[101:].notna().all()


def test_macd_matches_reference(close):
    m = ind.macd(close, 12, 26, 9)
    x = close.tolist()
    fast, slow = ref_ema(x, 12), ref_ema(x, 26)
    line = [f - s for f, s in zip(fast, slow)]
    # min_periods: EMA is NaN until `span` observations exist
    line = [float("nan") if i < 25 else v for i, v in enumerate(line)]
    sig = ref_ema(line, 9)
    sig = [float("nan") if i < 25 + 8 else v for i, v in enumerate(sig)]
    assert_close(m["macd"], line)
    assert_close(m["macd_signal"], sig)
    assert_close(m["macd_hist"], np.array(line) - np.array(sig))


def test_bollinger_matches_reference(close):
    bb = ind.bollinger_bands(close, 20, 2.0)
    x = close.tolist()
    mid = ref_sma(x, 20)
    sd = [float("nan") if i < 19 else float(np.std(x[i - 19 : i + 1])) for i in range(len(x))]
    up = [m + 2 * s for m, s in zip(mid, sd)]
    lo = [m - 2 * s for m, s in zip(mid, sd)]
    assert_close(bb["bb_middle"], mid)
    assert_close(bb["bb_upper"], up)
    assert_close(bb["bb_lower"], lo)
    pb = (close - bb["bb_lower"]) / (bb["bb_upper"] - bb["bb_lower"])
    assert_close(bb["bb_percent_b"], pb)


def test_bollinger_flat_prices_no_div_zero():
    bb = ind.bollinger_bands(pd.Series(np.full(30, 5.0)), 20, 2.0)
    assert bb["bb_percent_b"].isna().all()  # undefined, not inf


def test_add_all_indicators_columns(close):
    df = pd.DataFrame({"Close": close})
    out = ind.add_all_indicators(df)
    for col in ["sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist",
                "bb_upper", "bb_middle", "bb_lower", "bb_percent_b", "bb_bandwidth"]:
        assert col in out.columns
    assert out["sma_200"].iloc[-1] == pytest.approx(close.tail(200).mean())


def test_invalid_params():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ind.sma(s, 0)
    with pytest.raises(ValueError):
        ind.macd(s, 26, 12, 9)
