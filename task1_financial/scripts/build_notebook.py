"""Generates notebooks/Task1_Equity_Research.ipynb (unexecuted).

Execute afterwards with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/Task1_Equity_Research.ipynb
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate a Colab-ready walkthrough notebook for
# the Task 1 equity research pipeline with nbformat', Date: 2026-10-06

from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName"
COLAB_URL = ("https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName/"
             "blob/main/task1_financial/notebooks/Task1_Equity_Research.ipynb")

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md(f"""# Task 1 — LLM-Powered Equity Research Assistant
**CDAZZDEV Senior MLE Assessment · Financial AI**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook is the walkthrough for the `equity_research` package. Every step runs live, and the outputs are kept.

| Stage | Module | What it does |
|---|---|---|
| 1A data | `data.py` | ≥2y daily OHLCV from yfinance, window computed relative to today, retries, null repair |
| 1A features | `indicators.py` | SMA50/200, Wilder RSI(14), MACD(12,26,9), Bollinger(20,2), all written by hand (no TA-Lib) |
| 1A news | `news.py` | yfinance → Yahoo RSS → Google News RSS, de-duplicated, company-specific headlines first |
| 1A summary | `summary.py` | price, 52w range, P/E (with fallbacks), YTD, weighted momentum composite |
| 1B LLM | `llm.py`, `prompts.py`, `schemas.py` | OpenAI-compatible client (Groq/OpenRouter), JSON mode, Pydantic validation, logged self-repair |
| 1B reasoning | `sentiment.py`, `recommendation.py` | per-headline JSON → confidence×recency aggregate; Buy/Hold/Sell over indicator *interactions* |
| Bonus | `report.py` | one-page Markdown + styled HTML brief with an embedded matplotlib chart |

All tunable numbers are in `config.py`, and all prompt text is in `prompts.py`. API keys are read only from Colab secrets or environment variables. None are stored in the repo."""),

    md("## 0. Setup\nIn Colab this cell clones the repo and installs dependencies. Run locally, it uses the checked-out package."),
    code(f"""import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "{REPO_URL}"
    if not Path("/content/repo").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/repo"], check=True)
    os.chdir("/content/repo/task1_financial")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
else:
    # local run: notebook lives in task1_financial/notebooks
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)

sys.path.insert(0, str(Path.cwd()))
print("Working dir:", Path.cwd())"""),

    code("""# API keys come from Colab Secrets (key icon in the sidebar) or the environment. Never hard-code them.
if IN_COLAB:
    from google.colab import userdata
    for name in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
        try:
            os.environ[name] = userdata.get(name)
        except Exception:
            pass
else:
    try:
        from dotenv import load_dotenv  # override=True: edits to .env win over stale values in the kernel
        load_dotenv(Path.cwd().parent / ".env", override=True); load_dotenv(override=True)
    except ImportError:
        pass

print({k: ("set" if os.getenv(k) else "missing") for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY")})"""),

    code("""import json, logging
import pandas as pd
from IPython.display import HTML, Image, Markdown, display

from equity_research import config
from equity_research.pipeline import setup_logging
log_path = setup_logging()
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

TICKER = "MSFT"
print("Log file:", log_path.relative_to(Path.cwd()))"""),

    md("""---
# Task 1A — Financial Data Pipeline
## 1A.1 OHLCV fetch (≥ 2 years, no hard-coded dates)
`history_window()` computes `start = today − (2 years + 300-day warm-up)`. The warm-up means the 200-day SMA already has a value on the first day of the 2-year analysis window."""),
    code("""from equity_research.data import fetch_market_data, history_window

start, end = history_window()
print(f"Requested window: {start} -> {end} (computed relative to today)")

md_ = fetch_market_data(TICKER)
ohlcv = md_.ohlcv
span_years = (ohlcv.index[-1] - ohlcv.index[0]).days / 365.25
print(f"{md_.ticker}: {len(ohlcv)} daily bars, {ohlcv.index[0].date()} -> {ohlcv.index[-1].date()} ({span_years:.2f} years)")
print("Columns:", list(ohlcv.columns), "| nulls:", int(ohlcv.isna().sum().sum()), "| warnings:", md_.warnings)
assert span_years >= 2, "need at least two years of data"
ohlcv.tail()"""),

    md("""## 1A.2 Technical indicators, written from first principles
| Indicator | Definition used |
|---|---|
| SMA(n) | rolling mean, `min_periods=n` (NaN until the window is full) |
| RSI(14) | **Wilder smoothing**: seeded with the simple mean of the first 14 gains/losses, then `avg = (prev·13 + x)/14` |
| MACD(12,26,9) | EMA12 − EMA26 (α = 2/(n+1), recursive); signal = EMA9 of MACD; histogram = MACD − signal |
| Bollinger(20,2) | SMA20 ± 2σ (population σ, as Bollinger defined it), plus %B and bandwidth |"""),
    code("""from equity_research.indicators import add_all_indicators

frame = add_all_indicators(ohlcv)
cols = ["Close", "sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_lower", "bb_middle", "bb_upper", "bb_percent_b"]
frame[cols].tail(8).round(3)"""),

    md("""### Accuracy check
1. The pytest suite compares every indicator with a separate, plain-loop reference implementation written from the textbook formula. It also covers edge cases: flat prices, only-up moves, interior NaNs, and series that are too short.
2. A second check: Wilder's RSI equals an EMA with α = 1/14 once both are seeded from the same SMA. We compute it that way with pandas and compare."""),
    code("""r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", "tests"], capture_output=True, text=True)
print(r.stdout[-1500:])"""),
    code("""import numpy as np
close = frame["Close"]
delta = close.diff()
gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
# seed with the SMA of the first 14 deltas, then alpha=1/14 recursion via ewm
def wilder(x):
    x = x.iloc[1:].copy()
    seed = x.iloc[:14].mean()
    x.iloc[:13] = np.nan; x.iloc[13] = seed
    return x.ewm(alpha=1/14, adjust=False, ignore_na=True).mean()
ag, al = wilder(gain), wilder(loss)
rsi_alt = 100 - 100 / (1 + ag / al)
diff = (frame["rsi_14"] - rsi_alt).abs().max()
print(f"Max |RSI(loop) - RSI(ewm alpha=1/14)| = {diff:.2e}")
sma_check = abs(frame["sma_200"].iloc[-1] - close.tail(200).mean())
print(f"SMA200 last value vs direct mean of last 200 closes: diff = {sma_check:.2e}")"""),

    md("## 1A.3 News headlines (≥ 10, free sources)"),
    code("""from equity_research.news import fetch_news

company = md_.info.get("longName") or TICKER
news, news_warnings = fetch_news(TICKER, company)
print(f"{len(news)} headlines | warnings: {news_warnings}")
assert len(news) >= config.MIN_HEADLINES
pd.DataFrame([{"published_at": n.published_at, "source": n.source, "publisher": n.publisher,
               "headline": n.headline} for n in news])"""),

    md("""## 1A.4 Summary dictionary
The momentum signal is a weighted vote over six indicator components, each scored in [−1, +1]. Overbought RSI and a %B above 1 vote *against* the trend, because they signal exhaustion. If a component is missing (for example, a short history with no SMA200), its weight is spread over the components that remain."""),
    code("""from equity_research.summary import build_summary, technical_snapshot

summary = build_summary(md_.ticker, frame, md_.info)
technicals = technical_snapshot(frame)
print(json.dumps(summary, indent=2))"""),
    code("""print("Relational technical snapshot passed to the LLM:")
print(json.dumps(technicals, indent=2))"""),

    md("""## 1A.5 Robustness: missing and invalid data
The pipeline should degrade gracefully, without unhandled exceptions."""),
    code("""from equity_research.data import DataFetchError, _clean_ohlcv

# (a) invalid ticker -> a typed, catchable error after retries
config.FETCH_MAX_RETRIES, _orig = 1, config.FETCH_MAX_RETRIES
try:
    fetch_market_data("THIS_IS_NOT_A_TICKER_123")
except DataFetchError as e:
    print("Invalid ticker handled ->", type(e).__name__, ":", str(e)[:120])
finally:
    config.FETCH_MAX_RETRIES = _orig

# (b) corrupted bars: NaN close / open / volume
dirty = ohlcv.tail(260).copy()
dirty.iloc[5, dirty.columns.get_loc("Close")] = np.nan
dirty.iloc[10, dirty.columns.get_loc("Open")] = np.nan
dirty.iloc[12, dirty.columns.get_loc("Volume")] = np.nan
cleaned, warns = _clean_ohlcv(dirty)
print("Cleaning warnings:", warns)

# (c) short history + empty fundamentals -> summary still builds, missing fields are None
short = add_all_indicators(cleaned.tail(60))
s = build_summary("TEST", short, info={})
print("Short-history summary: pe_ratio =", s["pe_ratio"], "| momentum =", s["momentum_signal"]["label"],
      "| components =", s["momentum_signal"]["components"])"""),

    md("""---
# Task 1B — LLM Sentiment and Signal Reasoning
## 1B.1 Prompts are kept separate from logic
All prompt text is in `equity_research/prompts.py` as constants and `string.Template`s, with distinct **system** prompts (role, rules, output contract) and **user** prompts (per-call data)."""),
    code("""from equity_research import prompts
print("=== SENTIMENT_SYSTEM_PROMPT ===\\n" + prompts.SENTIMENT_SYSTEM_PROMPT)
print("\\n=== SENTIMENT_USER_TEMPLATE (rendered for headline #1) ===")
print(prompts.SENTIMENT_USER_TEMPLATE.substitute(company=company, ticker=TICKER,
      headline=news[0].headline, publisher=news[0].publisher))"""),

    md("## 1B.2 Per-headline sentiment: one LLM call per headline, each validated against `HeadlineSentiment`"),
    code("""from equity_research.pipeline import make_llm_client
from equity_research.sentiment import analyse_headlines

client = make_llm_client()
print("Provider:", client.provider if client else None, "| model:", client.model if client else None)
scored, agg = analyse_headlines(client, news, TICKER, company)
per_headline = [h.model_dump(include={"headline", "sentiment", "confidence", "brief_reason"}) for h in scored]
print(json.dumps(per_headline[:3], indent=2))
pd.DataFrame([{**d, "recency_w": h.recency_weight, "fallback": h.is_fallback}
              for d, h in zip(per_headline, scored)])"""),

    md("""### Aggregation
`score = Σ pᵢ·cᵢ·rᵢ / Σ cᵢ·rᵢ`, where p ∈ {+1, 0, −1} is polarity, c is the LLM's confidence, and r = 0.5^(age/3 days) is a recency weight.
- Weighting by confidence keeps ambiguous headlines from moving the score much.
- Neutral headlines stay in the denominator, so a quiet news flow pulls the score toward 0.
- Headlines that failed validation get c = 0, so they're excluded rather than counted as fake "neutral" votes."""),
    code("print(json.dumps(agg.model_dump(), indent=2))"),

    md("## 1B.3 Buy / Hold / Sell signal reasoned over indicator *combinations*"),
    code("""from equity_research.recommendation import build_signal_prompt, generate_signal
print("=== USER PROMPT SENT TO THE LLM ===")
print(build_signal_prompt(summary, technicals, agg))"""),
    code("""sig = generate_signal(client, summary, technicals, agg)
print(f"Source: {sig.source} | model: {sig.model} | attempts: {sig.attempts}")
print(json.dumps(sig.signal.model_dump(), indent=2))"""),

    md("""## 1B.4 Structured-output validation: failures are caught, logged and repaired
Every LLM response goes through `extract_json` and then `Model.model_validate`. When validation fails:
1. the error is **logged** to `outputs/logs/pipeline.log` and stored in `client.validation_events`;
2. the exact error is sent back to the model in a **repair** turn, up to 2 retries;
3. if it still fails, the pipeline uses a deterministic fallback that's clearly labelled: zero-weight neutral for a headline, rule-based for the signal.

### Controlled test: these failures are deliberate
The live run above had no validation failures, so the failure path is tested here on purpose. `ScriptedClient` replaces **only** the network call with pre-written replies. The JSON extraction, Pydantic validation, logging, repair loop and fallback are the real production code.

| Scenario | Scripted LLM replies | Expected behaviour |
|---|---|---|
| A. Self-repair | 1) prose + JSON with `"Strong Buy"`, conviction `1.7`, missing fields → 2) valid JSON | failure logged, error fed back, **valid on attempt 2** |
| B. Never valid | `"not json"` ×3 | 3 failures logged, returns `None` after 3 attempts, **no exception** |
| C. Graceful degradation | always invalid, called through the real pipeline functions | headline → zero-weight fallback; signal → `rule_based_fallback` |

Any `validation failed` entries this cell writes to the log are **expected output**. Console logging is muted while the cell runs so the results tables stay readable. Every event is still written to `pipeline.log`, and the excerpt below shows that."""),
    code("""import contextlib
from equity_research.llm import LLMClient
from equity_research.schemas import NewsItem, TradingSignal
from equity_research.sentiment import score_headline

class ScriptedClient(LLMClient):
    \"\"\"Test double: replaces only the network call; validation/repair/logging are the real code.\"\"\"
    def __init__(self, replies):
        self.provider, self.model, self._last_call = "scripted", "stub", 0.0
        self.validation_events, self.n_calls, self._replies = [], 0, list(replies)
        self.disabled_reason = None
    def _chat(self, messages, temperature, max_tokens):
        self.n_calls += 1
        return self._replies.pop(0) if self._replies else "not json"

@contextlib.contextmanager
def console_logging_muted():
    \"\"\"Silence console handlers only; the file handler keeps recording to pipeline.log.\"\"\"
    handlers = [h for h in logging.getLogger("equity_research").handlers if type(h) is logging.StreamHandler]
    levels = [h.level for h in handlers]
    for h in handlers: h.setLevel(logging.CRITICAL + 1)
    try: yield
    finally:
        for h, lvl in zip(handlers, levels): h.setLevel(lvl)

VALID_SIGNAL = json.dumps({"signal": "Hold", "conviction": 0.6, "time_horizon": "1-3 months",
    "justification": "Trend is up. Momentum is fading. Wait for confirmation.",
    "key_risks": ["Breakdown below SMA50"],
    "indicator_interactions": [
        {"indicators": ["price", "sma_200"], "interpretation": "Primary uptrend intact", "implication": "bullish"},
        {"indicators": ["macd_hist", "rsi_14"], "interpretation": "Momentum decelerating", "implication": "neutral"}]})
INVALID_SIGNAL = 'Sure! {"signal": "Strong Buy", "conviction": 1.7, "justification": "Looks good."}'

with console_logging_muted():
    # A. invalid -> repaired
    stub_a = ScriptedClient([INVALID_SIGNAL, VALID_SIGNAL])
    res_a = stub_a.structured(task="demo_repair", system="s", user="u", schema=TradingSignal, temperature=0, max_tokens=10)
    # B. never valid -> None, no exception
    stub_b = ScriptedClient(["not json"] * 3)
    res_b = stub_b.structured(task="demo_never_valid", system="s", user="u", schema=TradingSignal, temperature=0, max_tokens=10)
    # C. graceful degradation through the real pipeline functions
    stub_c = ScriptedClient([])
    head_c = score_headline(stub_c, NewsItem(headline="Microsoft beats estimates", source="demo"), TICKER, company)
    sig_c = generate_signal(stub_c, summary, technicals, agg)

results = pd.DataFrame([
    {"scenario": "A. self-repair", "expected": "valid on attempt 2", "actual": f"{res_a.value.signal} @ {res_a.value.conviction} after {res_a.attempts} attempts",
     "failures_logged": len(stub_a.validation_events), "exception_raised": False, "pass": res_a.value is not None and res_a.attempts == 2},
    {"scenario": "B. never valid", "expected": "None after 3 attempts", "actual": f"{res_b.value} after {res_b.attempts} attempts",
     "failures_logged": len(stub_b.validation_events), "exception_raised": False, "pass": res_b.value is None and res_b.attempts == 3},
    {"scenario": "C1. headline fallback", "expected": "is_fallback, weight 0", "actual": f"is_fallback={head_c.is_fallback}, confidence={head_c.confidence}",
     "failures_logged": None, "exception_raised": False, "pass": head_c.is_fallback and head_c.confidence == 0},
    {"scenario": "C2. signal fallback", "expected": "rule_based_fallback", "actual": f"{sig_c.source} -> {sig_c.signal.signal}",
     "failures_logged": len(stub_c.validation_events), "exception_raised": False, "pass": sig_c.source == "rule_based_fallback"},
])
assert results["pass"].all(), "validation path regression"
print("All controlled-failure scenarios behaved as designed.\\n")
display(results)

print("Validation events captured (scenarios A and B):")
display(pd.DataFrame([{"task": e.task, "attempt": e.attempt, "error": e.error}
                      for e in stub_a.validation_events + stub_b.validation_events]))"""),
    code("""live_failures = client.validation_events if client else []
print(f"Validation failures in the LIVE LLM run: {len(live_failures)}"
      + ("" if client else " (no LLM client configured)"))
for ev in live_failures:
    print(" -", ev.task, "attempt", ev.attempt, ":", ev.error[:200])

print(f"\\nEvidence that failures are persisted -- demo entries in {log_path.name}:")
print("".join(line for line in log_path.read_text(encoding="utf-8").splitlines(keepends=True)
              if "[demo_" in line))"""),

    md("""---
# Bonus — One-page Equity Research Brief
Combines 1A and 1B into Markdown and styled HTML (with the chart embedded as base64), and saves the full analysis JSON."""),
    code("""from equity_research.pipeline import Task1AResult, Task1BResult, write_outputs

a = Task1AResult(md_.ticker, frame, summary, technicals, news, md_.warnings + news_warnings)
b = Task1BResult(scored, agg, sig, client.provider if client else None,
                 client.n_calls if client else 0, list(client.validation_events) if client else [])
paths = write_outputs(a, b)
for k, p in paths.items():
    print(f"{k:>9}: {p.relative_to(Path.cwd())}")
display(Image(filename=str(paths["chart"])))"""),
    code("display(Markdown(paths['markdown'].read_text(encoding='utf-8').replace(f\"![Technical chart]({paths['chart'].name})\", '*(chart shown above)*')))"),
    code("display(HTML(paths['html'].read_text(encoding='utf-8')))"),

    md("""---
## Appendix — signal system prompt (full text)"""),
    code("print(prompts.SIGNAL_SYSTEM_PROMPT)"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "Task1_Equity_Research.ipynb"
nbf.write(nb, out)
print("wrote", out)
