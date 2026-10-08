"""Persistent memory: final research briefs cached as JSON keyed by ticker + date.

Short-term memory is provided by LangGraph's checkpointer (see research.py):
all messages -- including earlier tool observations -- persist per thread_id.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'JSON brief cache keyed by ticker and date with
# max-age policy', Date: 2026-10-07

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import config


def cache_path(ticker: str, day: date | None = None) -> Path:
    day = day or date.today()
    return config.CACHE_DIR / f"{ticker.upper()}_{day.isoformat()}.json"


def load_cached(ticker: str, today: date | None = None) -> tuple[dict[str, Any], Path] | None:
    """Most recent cached brief within CACHE_MAX_AGE_DAYS, or None."""
    today = today or date.today()
    for age in range(config.CACHE_MAX_AGE_DAYS + 1):
        p = cache_path(ticker, today - timedelta(days=age))
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")), p
            except json.JSONDecodeError:
                continue  # corrupt cache -> treat as miss, it will be overwritten
    return None


def save_brief(ticker: str, payload: dict[str, Any]) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(ticker)
    p.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return p
