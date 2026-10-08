# Task 1: Financial AI, LLM-Powered Equity Research Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName/blob/main/task1_financial/notebooks/Task1_Equity_Research.ipynb)

**Notebook (with executed outputs):** [`notebooks/Task1_Equity_Research.ipynb`](notebooks/Task1_Equity_Research.ipynb)
**Sample brief:** [`outputs/`](outputs/) (HTML, Markdown, chart PNG, and the full analysis JSON)

## Architecture

```
          ┌──────────── Task 1A ─────────────┐        ┌─────────── Task 1B ───────────┐
yfinance ─► data.py ─► indicators.py ─► summary.py ──► technical snapshot ─┐
 (OHLCV,   (relative    (SMA/RSI/MACD/   (summary dict,                    ├─► recommendation.py ─► SignalResult
  info)     window,      Bollinger,       momentum composite)              │     (LLM, validated)
            retries)     first principles)                                 │
RSS/yf  ─► news.py ──────────────────────────────────────► sentiment.py ───┘
           (3 sources, dedupe, relevance-ranked)            (1 LLM call/headline,
                                                             confidence×recency agg.)
                         prompts.py (all prompt text) · schemas.py (Pydantic) · llm.py (client)
                                               │
                                     report.py ─► chart PNG + brief.md + brief.html + analysis.json
```

| Module | Responsibility |
|---|---|
| `config.py` | Every constant (windows, thresholds, weights, model names). No magic numbers anywhere else. |
| `data.py` | Fetches 2 years plus a 300-day warm-up of daily OHLCV, with the window computed from `date.today()`. Retries with exponential backoff, cleans NaN bars, raises a typed `DataFetchError`. |
| `indicators.py` | SMA, EMA, **Wilder** RSI (SMA seed, then recursive smoothing), MACD(12,26,9), Bollinger(20,2) with %B and bandwidth. No TA-Lib. |
| `news.py` | Merges headlines from the yfinance news API, Yahoo Finance RSS, and Google News RSS. De-duplicates them and ranks company-specific headlines first, then by recency. |
| `summary.py` | Builds the summary dict (price, 52w high/low, P/E with fallbacks, YTD, momentum signal) and the *relational* technical snapshot passed to the LLM. |
| `prompts.py` | System and user prompts as constants / `string.Template`. |
| `schemas.py` | Pydantic models for every LLM output. These include a validator that requires the justification to be 3–5 sentences. |
| `llm.py` | OpenAI-compatible client for Groq or OpenRouter. Uses JSON mode, rate limiting, `extract_json`, and validation. On failure it **logs** the error, re-prompts the model with it, and finally returns `None` (it never raises). |
| `sentiment.py` / `recommendation.py` | Per-headline sentiment and aggregation. Buy/Hold/Sell signal with a labelled rule-based fallback. |
| `report.py` | 3-panel matplotlib chart, one-page Markdown, and styled HTML (chart embedded as base64, risk disclaimer). |
| `pipeline.py` | Orchestration, file logging, artefact writing, and the CLI. |

## Key design decisions

- **Indicator accuracy is tested, not just claimed.** `tests/test_indicators.py` compares each vectorised indicator with a separate plain-loop implementation of the textbook formula (agreement within 1e-9), and covers edge cases: flat prices, only-up moves, interior NaNs, short series. The notebook also cross-checks RSI against a pandas `ewm(alpha=1/14)` formulation (max difference ~1e-14).
- **The LLM gets relationships, not just levels.** Examples: % distance from each SMA, sessions since the last golden/death cross, the 5-day change in the MACD histogram (acceleration), Bollinger %B, bandwidth percentile (squeeze), and 20-day realised volatility. The system prompt asks for 2–4 `indicator_interactions`, each over **at least two** indicators, and gives examples of confluence and divergence reasoning. It also explicitly forbids simply restating values.
- **Sentiment aggregation** is `Σ p·c·r / Σ c·r`, where p is polarity, c is LLM confidence, and r is a recency weight with a 3-day half-life. Neutral headlines stay in the denominator so they dilute the score. Headlines that failed validation get zero weight instead of being counted as fake "neutral" votes.
- **Graceful degradation.** Bad price rows are repaired, and a missing P/E falls back to price / EPS, then forward P/E, then `None`. If indicators are missing, the momentum weights are renormalised over what's left. Invalid LLM JSON is logged and repaired. Without any LLM, the pipeline uses a deterministic fallback, and every output records its `source`.
- **No secrets in code.** Keys are read only from Colab Secrets, environment variables, or a git-ignored `.env`.

## Run it

```bash
cd task1_financial
pip install -r requirements.txt
export GROQ_API_KEY=...            # or OPENROUTER_API_KEY (free tiers)
python -m pytest -q                # 21 offline tests
python -m equity_research.pipeline --ticker MSFT
jupyter nbconvert --to notebook --execute --inplace notebooks/Task1_Equity_Research.ipynb
```

In **Colab**, add `GROQ_API_KEY` under *Secrets* (key icon), give the notebook access, and choose *Run all*.

## Rubric mapping

| Criterion | Where |
|---|---|
| OHLCV ≥2y, no hard-coded dates | `data.history_window`, notebook §1A.1 |
| Five indicators from first principles | `indicators.py`, tests, notebook §1A.2 |
| ≥10 headlines, free source | `news.py`, notebook §1A.3 |
| Summary dictionary | `summary.build_summary`, notebook §1A.4 |
| Robustness / no magic numbers | `config.py`, notebook §1A.5 |
| Per-headline JSON + aggregation | `schemas.HeadlineSentiment`, `sentiment.py`, §1B.2 |
| Signal reasoning over combinations | `prompts.SIGNAL_SYSTEM_PROMPT`, `summary.technical_snapshot`, §1B.3 |
| Validation, failures logged | `llm.LLMClient.structured`, `outputs/logs/pipeline.log`, §1B.4 |
| Prompt separation, system/user roles | `prompts.py`, §1B.1 |
| Bonus report | `report.py`, `outputs/*_brief.html` |
