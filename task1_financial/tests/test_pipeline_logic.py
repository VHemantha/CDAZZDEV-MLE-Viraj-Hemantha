"""Offline tests for summary fields, schema validation, JSON extraction and
sentiment aggregation (no network or API key needed)."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest for summary dict null-handling, Pydantic
# LLM schemas, JSON extraction and sentiment aggregation', Date: 2026-10-06

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from equity_research.indicators import add_all_indicators
from equity_research.llm import extract_json
from equity_research.schemas import HeadlineSentiment, ScoredHeadline, TradingSignal, count_sentences
from equity_research.sentiment import aggregate_sentiment, recency_weight
from equity_research.summary import build_summary, pe_ratio, technical_snapshot, ytd_return_pct


@pytest.fixture
def frame() -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(idx))))
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                       "Close": close, "Volume": 1e6}, index=idx)
    return add_all_indicators(df)


def test_summary_has_all_required_fields(frame):
    s = build_summary("TEST", frame, {"trailingPE": 31.2, "longName": "Test Corp"})
    for k in ["current_price", "52_week_high", "52_week_low", "pe_ratio", "ytd_return_pct",
              "momentum_signal"]:
        assert k in s
    assert s["52_week_low"] <= s["current_price"] <= s["52_week_high"]
    assert s["momentum_signal"]["label"] in {"Strong Bullish", "Bullish", "Neutral", "Bearish",
                                             "Strong Bearish"}


def test_summary_survives_empty_info_and_short_history(frame):
    s = build_summary("TEST", frame.tail(30), {})
    assert s["pe_ratio"] is None and s["pe_ratio_source"] == "unavailable"
    assert s["momentum_signal"]["score"] is not None  # SMA200 missing -> renormalised weights
    technical_snapshot(frame.tail(30))  # must not raise


def test_pe_fallbacks():
    assert pe_ratio({"trailingPE": "nan"}, 100.0) == (None, "unavailable")
    assert pe_ratio({"trailingEps": 4.0}, 100.0)[0] == pytest.approx(25.0)
    assert pe_ratio({"trailingPE": -5, "forwardPE": 20}, 100.0) == (20.0, "forward (yfinance)")


def test_ytd_uses_prior_year_close():
    idx = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-03-01"])
    df = pd.DataFrame({"Close": [90.0, 100.0, 101.0, 110.0]}, index=idx)
    assert ytd_return_pct(df) == pytest.approx(10.0)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 2} hope it helps') == {"a": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_headline_schema():
    ok = HeadlineSentiment.model_validate({"headline": "h", "sentiment": "Positive",
                                           "confidence": 0.8, "brief_reason": "beats estimates"})
    assert ok.sentiment == "positive"
    with pytest.raises(ValidationError):
        HeadlineSentiment.model_validate({"headline": "h", "sentiment": "bullish",
                                          "confidence": 1.4, "brief_reason": "x"})


def _signal(justification: str) -> dict:
    return {"signal": "buy", "conviction": 0.7, "justification": justification,
            "key_risks": ["r"],
            "indicator_interactions": [
                {"indicators": ["rsi", "macd"], "interpretation": "momentum confirmed", "implication": "Bullish"},
                {"indicators": ["sma_50", "sma_200"], "interpretation": "golden cross regime", "implication": "bullish"}]}


def test_signal_sentence_validator():
    good = "Trend is up. Momentum confirms at 1.5x volume. Risk is stretched RSI."
    assert count_sentences(good) == 3
    assert TradingSignal.model_validate(_signal(good)).signal == "Buy"
    with pytest.raises(ValidationError):
        TradingSignal.model_validate(_signal("Only one sentence."))


def test_aggregation_weights_confidence_and_ignores_fallback():
    now = datetime.now(timezone.utc)
    rows = [
        ScoredHeadline(headline="a", sentiment="positive", confidence=0.9, brief_reason="xyz", published_at=now),
        ScoredHeadline(headline="b", sentiment="negative", confidence=0.3, brief_reason="xyz", published_at=now),
        ScoredHeadline(headline="c", sentiment="neutral", confidence=0.0, brief_reason="xyz", is_fallback=True),
    ]
    agg = aggregate_sentiment(rows)
    assert agg.n_valid == 2 and agg.score == pytest.approx((0.9 - 0.3) / 1.2)
    assert agg.label == "positive"
    assert aggregate_sentiment([]).score == 0.0


def test_recency_weight_halves():
    now = datetime.now(timezone.utc)
    assert recency_weight(now - timedelta(days=3), now) == pytest.approx(0.5)
