"""Pydantic models: the contract for every piece of structured data, including
every LLM response. Nothing produced by the LLM is used until it validates."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic v2 schemas for per-headline sentiment
# and a Buy/Hold/Sell signal with a 3-5 sentence justification validator', Date: 2026-10-06

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config

Sentiment = Literal["positive", "negative", "neutral"]
Signal = Literal["Buy", "Hold", "Sell"]

# A sentence ends with . ! or ? followed by whitespace and a capital/digit/quote.
# Decimals ("1.5") and tickers are not split because they lack trailing whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()])


class NewsItem(BaseModel):
    headline: str
    publisher: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    source: str


# --------------------------------------------------------------------------- #
# LLM output #1: one object per headline (the four fields the brief requires)
# --------------------------------------------------------------------------- #
class HeadlineSentiment(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    headline: str = Field(min_length=1)
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = Field(min_length=3, max_length=400)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _lower(cls, v: object) -> object:
        # Tolerate "Positive"/"NEGATIVE" -- casing is not a semantic error.
        return v.strip().lower() if isinstance(v, str) else v


class ScoredHeadline(HeadlineSentiment):
    """HeadlineSentiment plus pipeline metadata (not produced by the LLM)."""

    published_at: datetime | None = None
    publisher: str | None = None
    recency_weight: float = 1.0
    is_fallback: bool = False   # True when the LLM result failed validation


class SentimentAggregate(BaseModel):
    score: float = Field(ge=-1.0, le=1.0, description="Confidence- and recency-weighted polarity")
    label: Sentiment
    n_headlines: int
    n_valid: int
    counts: dict[str, int]
    mean_confidence: float
    method: str


# --------------------------------------------------------------------------- #
# LLM output #2: the trading signal
# --------------------------------------------------------------------------- #
class IndicatorInteraction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    indicators: list[str] = Field(min_length=2, description="Two or more indicators considered jointly")
    interpretation: str = Field(min_length=10)
    implication: Literal["bullish", "bearish", "neutral"]

    @field_validator("implication", mode="before")
    @classmethod
    def _lower(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v


class TradingSignal(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    signal: Signal
    conviction: float = Field(ge=0.0, le=1.0)
    indicator_interactions: list[IndicatorInteraction] = Field(min_length=2, max_length=5)
    justification: str
    key_risks: list[str] = Field(min_length=1, max_length=4)
    time_horizon: str = Field(default="1-3 months")

    @field_validator("signal", mode="before")
    @classmethod
    def _canonical_signal(cls, v: object) -> object:
        return v.strip().capitalize() if isinstance(v, str) else v

    @field_validator("justification")
    @classmethod
    def _three_to_five_sentences(cls, v: str) -> str:
        n = count_sentences(v)
        if not config.JUSTIFICATION_MIN_SENTENCES <= n <= config.JUSTIFICATION_MAX_SENTENCES:
            raise ValueError(
                f"justification must be {config.JUSTIFICATION_MIN_SENTENCES}-"
                f"{config.JUSTIFICATION_MAX_SENTENCES} sentences, got {n}"
            )
        return v


class SignalResult(BaseModel):
    """TradingSignal plus provenance, so the report is honest about its source."""

    signal: TradingSignal
    source: Literal["llm", "rule_based_fallback"]
    model: str | None = None
    attempts: int = 1
