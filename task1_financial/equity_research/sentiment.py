"""Per-headline LLM sentiment and aggregation into one overall score."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Score each headline with an LLM into validated
# JSON and aggregate with confidence x recency weighting', Date: 2026-10-06

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import config
from .llm import LLMClient
from .prompts import SENTIMENT_SYSTEM_PROMPT, SENTIMENT_USER_TEMPLATE
from .schemas import HeadlineSentiment, NewsItem, ScoredHeadline, SentimentAggregate

logger = logging.getLogger(__name__)

FALLBACK_REASON = "LLM output unavailable or failed validation; excluded from the aggregate."


def recency_weight(published_at: datetime | None, now: datetime | None = None) -> float:
    """Exponential decay: weight = 0.5 ** (age_days / half_life)."""
    if published_at is None:
        return config.SENTIMENT_UNKNOWN_DATE_WEIGHT
    now = now or datetime.now(timezone.utc)
    age_days = max((now - published_at).total_seconds() / 86_400, 0.0)
    return 0.5 ** (age_days / config.SENTIMENT_RECENCY_HALF_LIFE_DAYS)


def score_headline(client: LLMClient | None, item: NewsItem, ticker: str, company: str) -> ScoredHeadline:
    """One LLM call per headline. Falls back to a zero-confidence neutral record
    (which carries zero weight in the aggregate) if the LLM fails."""
    meta = dict(published_at=item.published_at, publisher=item.publisher,
                recency_weight=round(recency_weight(item.published_at), 4))
    if client is not None:
        user = SENTIMENT_USER_TEMPLATE.substitute(
            company=company, ticker=ticker, headline=item.headline,
            publisher=item.publisher or "unknown")
        result = client.structured(
            task="headline_sentiment", system=SENTIMENT_SYSTEM_PROMPT, user=user,
            schema=HeadlineSentiment, temperature=config.SENTIMENT_TEMPERATURE,
            max_tokens=config.SENTIMENT_MAX_TOKENS)
        if result.value is not None:
            data = result.value.model_dump()
            data["headline"] = item.headline   # keep our verbatim copy, not the model's echo
            return ScoredHeadline(**data, **meta)
    return ScoredHeadline(headline=item.headline, sentiment="neutral", confidence=0.0,
                          brief_reason=FALLBACK_REASON, is_fallback=True, **meta)


def aggregate_sentiment(scored: list[ScoredHeadline]) -> SentimentAggregate:
    """Overall score in [-1, 1]:

        score = sum(p_i * c_i * r_i) / sum(c_i * r_i)

    p = polarity (+1 / 0 / -1), c = LLM confidence, r = recency weight.
    * Confidence weighting stops ambiguous headlines from swinging the score.
    * Neutral headlines stay in the denominator, so a mostly-neutral news flow
      pulls the score towards 0 rather than being ignored.
    * Fallback rows have c = 0 and therefore contribute nothing.
    """
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    num = den = 0.0
    valid = [h for h in scored if not h.is_fallback]
    for h in valid:
        counts[h.sentiment] += 1
        w = h.confidence * h.recency_weight
        num += config.SENTIMENT_POLARITY[h.sentiment] * w
        den += w
    score = num / den if den > 0 else 0.0
    if score >= config.SENTIMENT_LABEL_THRESHOLD:
        label = "positive"
    elif score <= -config.SENTIMENT_LABEL_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    mean_conf = sum(h.confidence for h in valid) / len(valid) if valid else 0.0
    return SentimentAggregate(
        score=round(max(-1.0, min(1.0, score)), 4), label=label, n_headlines=len(scored),
        n_valid=len(valid), counts=counts, mean_confidence=round(mean_conf, 3),
        method=(f"confidence x recency (half-life {config.SENTIMENT_RECENCY_HALF_LIFE_DAYS:g}d) "
                "weighted mean polarity"))


def analyse_headlines(client: LLMClient | None, news: list[NewsItem], ticker: str,
                      company: str) -> tuple[list[ScoredHeadline], SentimentAggregate]:
    scored = [score_headline(client, item, ticker, company) for item in news]
    return scored, aggregate_sentiment(scored)
