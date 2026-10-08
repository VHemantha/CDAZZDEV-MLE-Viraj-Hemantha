"""LLM Buy/Hold/Sell signal reasoned over indicator combinations, with a
deterministic rule-based fallback so the report is always produced."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate a validated Buy/Hold/Sell signal from
# technical snapshot + sentiment via LLM, falling back to the momentum composite', Date: 2026-10-06

from __future__ import annotations

import json
import logging
from typing import Any

from . import config
from .llm import LLMClient
from .prompts import SIGNAL_SYSTEM_PROMPT, SIGNAL_USER_TEMPLATE
from .schemas import IndicatorInteraction, SentimentAggregate, SignalResult, TradingSignal

logger = logging.getLogger(__name__)


def build_signal_prompt(summary: dict[str, Any], technicals: dict[str, Any],
                        sentiment: SentimentAggregate) -> str:
    return SIGNAL_USER_TEMPLATE.substitute(
        ticker=summary["ticker"], company=summary["company_name"], as_of=technicals["as_of"],
        technicals=json.dumps(technicals, indent=2),
        momentum=json.dumps(summary["momentum_signal"], indent=2),
        sentiment_score=f"{sentiment.score:+.2f}", sentiment_label=sentiment.label,
        n_headlines=sentiment.n_valid,
        sentiment_counts=", ".join(f"{k}: {v}" for k, v in sentiment.counts.items()))


def rule_based_signal(summary: dict[str, Any], technicals: dict[str, Any],
                      sentiment: SentimentAggregate) -> TradingSignal:
    """Fallback used only when the LLM is unavailable or never validates.
    Maps the momentum composite (+ a sentiment nudge) to a signal."""
    mom = summary["momentum_signal"]
    score = (mom.get("score") or 0.0) + 0.2 * sentiment.score
    signal = "Buy" if score >= config.MOMENTUM_WEAK_THRESHOLD else (
        "Sell" if score <= -config.MOMENTUM_WEAK_THRESHOLD else "Hold")
    trend = "above" if (technicals.get("pct_vs_sma_200") or 0) > 0 else "below"
    macd_dir = "positive" if (technicals.get("macd_hist") or 0) > 0 else "negative"
    return TradingSignal(
        signal=signal,
        conviction=round(min(0.5 + abs(score) / 2, 0.9), 2),
        indicator_interactions=[
            IndicatorInteraction(indicators=["price", "sma_200", "sma_50"],
                                 interpretation=f"Price trades {trend} its 200-day average.",
                                 implication="bullish" if trend == "above" else "bearish"),
            IndicatorInteraction(indicators=["macd", "rsi_14"],
                                 interpretation=f"MACD histogram is {macd_dir} with RSI at {technicals.get('rsi_14')}.",
                                 implication="bullish" if macd_dir == "positive" else "bearish"),
        ],
        justification=(
            f"This is a rule-based fallback because the LLM response was unavailable. "
            f"The weighted momentum composite reads {mom.get('label')} (score {mom.get('score')}). "
            f"News sentiment is {sentiment.label} at {sentiment.score:+.2f}, which adjusts the "
            f"combined score to {score:+.2f} and maps to {signal}."),
        key_risks=["Rule-based output: no qualitative reasoning was applied."],
    )


def generate_signal(client: LLMClient | None, summary: dict[str, Any], technicals: dict[str, Any],
                    sentiment: SentimentAggregate) -> SignalResult:
    if client is not None:
        result = client.structured(
            task="trading_signal", system=SIGNAL_SYSTEM_PROMPT,
            user=build_signal_prompt(summary, technicals, sentiment), schema=TradingSignal,
            temperature=config.SIGNAL_TEMPERATURE, max_tokens=config.SIGNAL_MAX_TOKENS)
        if result.value is not None:
            return SignalResult(signal=result.value, source="llm", model=client.model,
                                attempts=result.attempts)
        logger.error("LLM signal failed validation; using rule-based fallback. Errors: %s",
                     result.errors)
    return SignalResult(signal=rule_based_signal(summary, technicals, sentiment),
                        source="rule_based_fallback")
