"""Typed contracts: the single-agent report, the Agent A -> Agent B handoff,
the critique-loop messages, and the final multi-agent report."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schemas for research report, data
# brief handoff, clarification request/response and final report', Date: 2026-10-07

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# 3A: single research agent
# --------------------------------------------------------------------------- #
_EVIDENCE_RE = re.compile(r"^\[[A-Za-z0-9_ .\-]+\]\s*\S")


class Risk(_Base):
    title: str = Field(min_length=5)
    description: str = Field(min_length=20)
    evidence: list[str] = Field(min_length=1, description="'[tool_name] concrete figure or quoted headline'")
    likelihood: Literal["low", "medium", "high"]

    @field_validator("evidence")
    @classmethod
    def _cited(cls, items: list[str]) -> list[str]:
        """Evidence must name its source tool and contain a figure or a quote --
        this is what makes the report 'evidence-backed' rather than generic."""
        for it in items:
            if not _EVIDENCE_RE.match(it):
                raise ValueError(f"evidence must start with '[source]' (a tool name or a handoff such as DataBrief): {it[:60]!r}")
            if not (re.search(r"\d", it) or re.search(r"['\"‘’“”]", it)):
                raise ValueError(f"evidence must quote a figure or headline: {it[:60]!r}")
        return items


class HedgeStrategy(_Base):
    strategy: str = Field(min_length=5, description="e.g. protective put, collar, beta hedge with index short")
    implementation: str = Field(min_length=20, description="instrument, strike/tenor or hedge ratio")
    data_rationale: str = Field(min_length=20, description="which computed metrics justify it")
    estimated_cost_or_tradeoff: str = Field(min_length=5)


class ResearchReport(_Base):
    ticker: str
    financial_health_summary: str = Field(min_length=80)
    market_sentiment_summary: str = Field(min_length=30)
    top_risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy: HedgeStrategy
    data_sources_used: list[str] = Field(min_length=1)


# --------------------------------------------------------------------------- #
# 3B: Agent A -> Agent B structured handoff
# --------------------------------------------------------------------------- #
class PriceSnapshot(_Base):
    last_close: float
    period_return_pct: float | None = None
    pct_vs_sma_50: float | None = None
    pct_vs_sma_200: float | None = None
    rsi_14: float | None = None
    macd_hist: float | None = None
    max_drawdown_pct: float | None = None


class VolatilitySnapshot(_Base):
    window_days: int
    annualised_vol_pct: float
    one_year_vol_pct: float | None = None
    var_95_1d_pct: float | None = None
    beta_vs_spy: float | None = None


class SentimentSnapshot(_Base):
    score: float = Field(ge=-1, le=1)
    label: Literal["positive", "negative", "neutral"]
    n_headlines: int


class DataBrief(_Base):
    """Agent A's output -- the ONLY channel through which Agent B sees price data."""

    ticker: str
    as_of: str
    price: PriceSnapshot
    volatility: VolatilitySnapshot
    sentiment: SentimentSnapshot | None = Field(
        default=None, description="None when Agent A had no headlines to score")
    quantitative_flags: list[str] = Field(min_length=1, description="data-driven observations")
    data_gaps: list[str] = Field(default_factory=list)


class BriefNarrative(_Base):
    """The only part of the DataBrief an LLM writes; all figures come from tools."""

    quantitative_flags: list[str] = Field(min_length=1)
    data_gaps: list[str] = Field(default_factory=list)


class ClarificationSummary(_Base):
    answer_summary: str = Field(min_length=20)


class ClarificationRequest(_Base):
    """Agent B -> Agent A: exactly one specific, answerable request."""

    question: str = Field(min_length=15)
    requested_analyses: list[Literal["sentiment_of_headlines", "volatility_at_window", "price_period"]] = Field(min_length=1)
    headlines: list[str] = Field(default_factory=list, description="payload for sentiment scoring")
    volatility_window: int | None = Field(default=None, ge=5, le=252)
    price_period: str | None = None
    reason: str = Field(min_length=15)


class ClarificationResponse(_Base):
    """Agent A -> Agent B answer, computed with Agent A's own tools."""

    answer_summary: str = Field(min_length=20)
    sentiment: SentimentSnapshot | None = None
    volatility: VolatilitySnapshot | None = None
    price: PriceSnapshot | None = None


class FinalReport(ResearchReport):
    """Agent B's report: the 3A report plus explicit use of the clarification."""

    clarification_incorporated: str = Field(min_length=30, description="how A's answer changed the report")
