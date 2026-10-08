"""Headline retrieval from free sources with fallbacks and de-duplication.

Source order: yfinance news endpoint -> Yahoo Finance RSS -> Google News RSS.
The yfinance endpoint is frequently empty or rate-limited, so the RSS feeds
make sure the pipeline still reaches the >= 10 headline requirement.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Fetch >=10 recent headlines for a ticker from
# yfinance news with Yahoo/Google News RSS fallbacks, normalise and de-duplicate', Date: 2026-10-06

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import requests
import yfinance as yf

from . import config
from .schemas import NewsItem

logger = logging.getLogger(__name__)


def _parse_datetime(value: Any) -> datetime | None:
    """Accept epoch seconds, ISO-8601 or RFC-822 strings; return aware UTC or None."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        text = str(value)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = parsedate_to_datetime(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _from_yfinance(ticker: str) -> list[NewsItem]:
    """yfinance >= 0.2.50 nests fields under 'content'; older versions are flat."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return []
    items: list[NewsItem] = []
    for entry in raw:
        content = entry.get("content", entry) if isinstance(entry, dict) else {}
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider") or {}
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        items.append(NewsItem(
            headline=title,
            publisher=provider.get("displayName") if isinstance(provider, dict) else content.get("publisher"),
            url=url_obj.get("url") if isinstance(url_obj, dict) else content.get("link"),
            published_at=_parse_datetime(content.get("pubDate") or content.get("providerPublishTime")),
            source="yfinance",
        ))
    return items


def _from_rss(url: str, source: str) -> list[NewsItem]:
    """Minimal RSS 2.0 reader (stdlib XML parser -- no extra dependency)."""
    try:
        resp = requests.get(url, timeout=config.NEWS_HTTP_TIMEOUT_SECONDS,
                            headers={"User-Agent": config.HTTP_USER_AGENT})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        logger.warning("RSS source %s failed: %s", source, exc)
        return []
    items: list[NewsItem] = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        if not title:
            continue
        publisher = (node.findtext("source") or "").strip() or None
        # Google News appends " - Publisher" to titles; strip it for cleaner LLM input.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(f" - {publisher}")].strip()
        items.append(NewsItem(
            headline=title,
            publisher=publisher,
            url=(node.findtext("link") or "").strip() or None,
            published_at=_parse_datetime(node.findtext("pubDate")),
            source=source,
        ))
    return items


def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


_LEGAL_SUFFIX = re.compile(r"[,.]?\s+(corporation|corp|incorporated|inc|plc|ltd|limited|co|company|"
                           r"holdings|group|n\.?v|s\.?a|ag)\.?$", re.IGNORECASE)


def company_keywords(ticker: str, company_name: str | None) -> list[str]:
    """Terms that mark a headline as company-specific, e.g. ['msft', 'microsoft']."""
    words = [ticker.lower()]
    if company_name:
        short = company_name.strip()
        while (stripped := _LEGAL_SUFFIX.sub("", short)) != short:
            short = stripped
        words.append(short.lower())
    return words


def is_relevant(headline: str, keywords: list[str]) -> bool:
    text = headline.lower()
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def fetch_news(ticker: str, company_name: str | None = None,
               n: int = config.MAX_HEADLINES) -> tuple[list[NewsItem], list[str]]:
    """Return up to ``n`` unique headlines plus any warnings.

    All sources are queried and merged. Ticker feeds (Yahoo RSS especially) mix
    in general-market stories, so headlines that name the company or ticker are
    ranked first, then by recency; generic stories only fill remaining slots.
    """
    keywords = company_keywords(ticker, company_name)
    query = quote_plus(f'"{keywords[-1]}" {ticker} stock')
    sources = [
        lambda: _from_yfinance(ticker),
        lambda: _from_rss(config.YAHOO_RSS_URL.format(ticker=ticker), "yahoo_rss"),
        lambda: _from_rss(config.GOOGLE_NEWS_RSS_URL.format(query=query), "google_news_rss"),
    ]
    seen: set[str] = set()
    collected: list[NewsItem] = []
    for fetch in sources:
        for item in fetch():
            key = _normalise(item.headline)
            if key and key not in seen:
                seen.add(key)
                collected.append(item)

    oldest = datetime.min.replace(tzinfo=timezone.utc)
    collected.sort(key=lambda it: (is_relevant(it.headline, keywords), it.published_at or oldest),
                   reverse=True)
    collected = collected[:n]

    warnings: list[str] = []
    if len(collected) < config.MIN_HEADLINES:
        warnings.append(f"Only {len(collected)} headline(s) found (< {config.MIN_HEADLINES}).")
    n_relevant = sum(is_relevant(it.headline, keywords) for it in collected)
    if n_relevant < config.MIN_HEADLINES:
        warnings.append(f"Only {n_relevant} of {len(collected)} headlines name {ticker} directly.")
    by_source: dict[str, int] = {}
    for it in collected:
        by_source[it.source] = by_source.get(it.source, 0) + 1
    logger.info("Collected %d headlines for %s (%d company-specific): %s",
                len(collected), ticker, n_relevant, by_source)
    return collected, warnings
