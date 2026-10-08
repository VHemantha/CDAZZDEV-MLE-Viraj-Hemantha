"""End-to-end orchestration: 1A (data + features) -> 1B (LLM) -> report.

Run from the task folder:  python -m equity_research.pipeline --ticker MSFT
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Orchestrate data pipeline, LLM sentiment/signal
# and report rendering with file logging and JSON artefacts', Date: 2026-10-06

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .data import fetch_market_data
from .indicators import add_all_indicators
from .llm import LLMClient, LLMUnavailableError, ValidationEvent
from .news import fetch_news
from .recommendation import generate_signal
from .report import render_chart, render_html, render_markdown
from .schemas import NewsItem, ScoredHeadline, SentimentAggregate, SignalResult
from .sentiment import analyse_headlines
from .summary import build_summary, technical_snapshot

logger = logging.getLogger("equity_research")


def setup_logging(level: int = logging.INFO) -> Path:
    """Console + file logging. Validation failures end up in the log file."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / "pipeline.log"
    root = logging.getLogger("equity_research")
    root.setLevel(level)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root.propagate = False
    return log_path


@dataclass
class Task1AResult:
    ticker: str
    frame: pd.DataFrame
    summary: dict[str, Any]
    technicals: dict[str, Any]
    news: list[NewsItem]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Task1BResult:
    scored: list[ScoredHeadline]
    sentiment: SentimentAggregate
    signal: SignalResult
    llm_provider: str | None
    llm_calls: int
    validation_events: list[ValidationEvent]


def run_task_1a(ticker: str = config.DEFAULT_TICKER) -> Task1AResult:
    md = fetch_market_data(ticker)
    frame = add_all_indicators(md.ohlcv)
    summary = build_summary(md.ticker, frame, md.info)
    technicals = technical_snapshot(frame)
    news, news_warnings = fetch_news(md.ticker, summary["company_name"])
    warnings = md.warnings + news_warnings
    for w in warnings:
        logger.warning(w)
    return Task1AResult(md.ticker, frame, summary, technicals, news, warnings)


def make_llm_client(provider: str | None = None) -> LLMClient | None:
    try:
        return LLMClient(provider)
    except LLMUnavailableError as exc:
        logger.error("%s -- continuing with rule-based fallbacks.", exc)
        return None


def run_task_1b(a: Task1AResult, client: LLMClient | None) -> Task1BResult:
    company = a.summary["company_name"]
    scored, agg = analyse_headlines(client, a.news, a.ticker, company)
    sig = generate_signal(client, a.summary, a.technicals, agg)
    return Task1BResult(scored, agg, sig, client.provider if client else None,
                        client.n_calls if client else 0,
                        list(client.validation_events) if client else [])


def write_outputs(a: Task1AResult, b: Task1BResult, out_dir: Path = config.OUTPUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.ticker}_{a.summary['as_of']}"
    paths = {
        "chart": out_dir / f"{stem}_chart.png",
        "markdown": out_dir / f"{stem}_brief.md",
        "html": out_dir / f"{stem}_brief.html",
        "json": out_dir / f"{stem}_analysis.json",
    }
    png = render_chart(a.frame, a.ticker, paths["chart"])
    paths["markdown"].write_text(render_markdown(a.summary, a.technicals, b.scored, b.sentiment,
                                                 b.signal, paths["chart"].name), encoding="utf-8")
    paths["html"].write_text(render_html(a.summary, a.technicals, b.scored, b.sentiment,
                                         b.signal, png), encoding="utf-8")
    payload = {
        "summary": a.summary,
        "technicals": a.technicals,
        "headline_sentiment": [h.model_dump(mode="json") for h in b.scored],
        "aggregate_sentiment": b.sentiment.model_dump(),
        "signal": b.signal.model_dump(),
        "llm": {"provider": b.llm_provider, "calls": b.llm_calls,
                "validation_failures": [asdict(ev) for ev in b.validation_events]},
        "warnings": a.warnings,
    }
    paths["json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-powered equity research brief")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--provider", choices=list(config.LLM_PROVIDERS), default=None)
    args = parser.parse_args()
    setup_logging()
    try:
        from dotenv import load_dotenv  # optional convenience for local runs
        load_dotenv(override=True)
    except ImportError:
        pass
    a = run_task_1a(args.ticker)
    b = run_task_1b(a, make_llm_client(args.provider))
    for name, path in write_outputs(a, b).items():
        print(f"{name:>9}: {path}")


if __name__ == "__main__":
    main()
