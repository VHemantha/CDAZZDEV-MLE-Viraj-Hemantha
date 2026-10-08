"""One-page equity research brief: matplotlib chart + Markdown + styled HTML."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Render a one-page equity research brief in
# Markdown and styled HTML with an embedded 3-panel matplotlib chart (price/SMA/Bollinger,
# MACD, RSI)', Date: 2026-10-06

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: works in Colab, CI and scripts alike
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .schemas import ScoredHeadline, SentimentAggregate, SignalResult  # noqa: E402

# Validated categorical palette (light mode) -- fixed slot order, see CITATIONS.md
C_PRICE, C_SMA_S, C_SMA_L = "#2a78d6", "#eb6834", "#4a3aa7"
C_UP, C_DOWN = "#1baf7a", "#e34948"
C_BAND, C_GRID, C_INK, C_MUTED = "#d9d8d4", "#ecebe8", "#0b0b0b", "#52514e"

DISCLAIMER = (
    "This report was generated automatically by a machine-learning pipeline for educational and "
    "assessment purposes only. It is not investment advice, an offer, or a solicitation to buy or "
    "sell any security. Technical indicators are backward-looking, LLM output can be wrong or "
    "incomplete, and news sentiment is inferred from headlines only. Past performance does not "
    "guarantee future results. Consult a licensed financial adviser before making any investment "
    "decision.")


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
def render_chart(df: pd.DataFrame, ticker: str, path: Path | None = None) -> bytes:
    """Three stacked panels sharing the date axis (never a dual y-axis)."""
    d = df.tail(config.REPORT_CHART_LOOKBACK_SESSIONS)
    sma_s, sma_l = f"sma_{config.SMA_SHORT_WINDOW}", f"sma_{config.SMA_LONG_WINDOW}"
    rsi_col = f"rsi_{config.RSI_PERIOD}"

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": C_GRID, "axes.labelcolor": C_MUTED,
                         "xtick.color": C_MUTED, "ytick.color": C_MUTED})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color=C_GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ax1.fill_between(d.index, d["bb_lower"], d["bb_upper"], color=C_BAND, alpha=0.55,
                     linewidth=0, label=f"Bollinger ({config.BOLLINGER_WINDOW}, {config.BOLLINGER_NUM_STD:g}σ)")
    ax1.plot(d.index, d["Close"], color=C_PRICE, linewidth=1.6, label="Close")
    ax1.plot(d.index, d[sma_s], color=C_SMA_S, linewidth=1.2, label=f"SMA {config.SMA_SHORT_WINDOW}")
    ax1.plot(d.index, d[sma_l], color=C_SMA_L, linewidth=1.2, label=f"SMA {config.SMA_LONG_WINDOW}")
    ax1.set_title(f"{ticker} — price, moving averages and Bollinger Bands (last 12 months)",
                  loc="left", fontsize=9, color=C_INK)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", frameon=False, ncol=4, fontsize=7)

    hist = d["macd_hist"]
    ax2.bar(d.index, hist, color=[C_UP if v >= 0 else C_DOWN for v in hist.fillna(0)],
            width=1.0, linewidth=0, label="Histogram")
    ax2.plot(d.index, d["macd"], color=C_PRICE, linewidth=1.1, label="MACD")
    ax2.plot(d.index, d["macd_signal"], color=C_SMA_S, linewidth=1.1, label="Signal")
    ax2.axhline(0, color=C_MUTED, linewidth=0.6)
    ax2.set_ylabel(f"MACD\n({config.MACD_FAST},{config.MACD_SLOW},{config.MACD_SIGNAL})")
    ax2.legend(loc="upper left", frameon=False, ncol=3, fontsize=7)

    ax3.plot(d.index, d[rsi_col], color=C_PRICE, linewidth=1.1, label=f"RSI {config.RSI_PERIOD}")
    for level in (config.RSI_OVERSOLD, config.RSI_OVERBOUGHT):
        ax3.axhline(level, color=C_MUTED, linewidth=0.7, linestyle="--")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel(f"RSI ({config.RSI_PERIOD})")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=config.CHART_DPI, facecolor="white")
    plt.close(fig)
    png = buf.getvalue()
    if path is not None:
        path.write_bytes(png)
    return png


# --------------------------------------------------------------------------- #
# Content helpers
# --------------------------------------------------------------------------- #
def _fmt(v: Any, nd: int = 2, suffix: str = "", sign: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:+,.{nd}f}{suffix}" if sign else f"{v:,.{nd}f}{suffix}"
    return str(v)


def _market_cap(v: float | None) -> str:
    if v is None:
        return "n/a"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.0f}"


def top_headlines(scored: list[ScoredHeadline], k: int = config.REPORT_TOP_HEADLINES) -> list[ScoredHeadline]:
    """Most informative headlines: highest confidence x recency among non-neutral,
    padded with neutral ones if fewer than k are directional."""
    valid = [h for h in scored if not h.is_fallback]
    if not valid:  # LLM unavailable: still show the newest headlines (marked as unscored)
        return scored[:k]
    key = lambda h: h.confidence * h.recency_weight  # noqa: E731
    directional = sorted([h for h in valid if h.sentiment != "neutral"], key=key, reverse=True)
    neutral = sorted([h for h in valid if h.sentiment == "neutral"], key=key, reverse=True)
    return (directional + neutral)[:k]


def technical_outlook_lines(t: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    cross = t.get("last_ma_cross")
    cross_txt = (f"{cross.replace('_', ' ')} {t['sessions_since_ma_cross']} sessions ago"
                 if cross else "no 50/200 crossover in window")
    mom = summary["momentum_signal"]
    return [
        f"Trend: price {_fmt(t['pct_vs_sma_50'], sign=True, suffix='%')} vs SMA50, "
        f"{_fmt(t['pct_vs_sma_200'], sign=True, suffix='%')} vs SMA200 ({cross_txt}).",
        f"Momentum: RSI(14) {_fmt(t['rsi_14'], 1)}; MACD {_fmt(t['macd'], 2)} vs signal "
        f"{_fmt(t['macd_signal'], 2)}, histogram {_fmt(t['macd_hist'], 2, sign=True)} "
        f"(5-day change {_fmt(t.get(f'macd_hist_change_{config.MACD_HIST_SLOPE_LOOKBACK}d'), 2, sign=True)}).",
        f"Volatility: Bollinger %B {_fmt(t['bb_percent_b'], 2)}, bandwidth {_fmt(t['bb_bandwidth_pct'], 1, '%')} "
        f"({_fmt(t['bb_bandwidth_percentile_6m'], 0)}th pct of 6m); 20d realised vol "
        f"{_fmt(t['realised_vol_20d_annualised_pct'], 1, '%')} annualised.",
        f"Rule-based momentum composite: **{mom['label']}** (score {_fmt(mom['score'], 2, sign=True)}).",
    ]


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_markdown(summary: dict[str, Any], technicals: dict[str, Any], scored: list[ScoredHeadline],
                    agg: SentimentAggregate, sig: SignalResult, chart_filename: str) -> str:
    s, ts = summary, sig.signal
    cur = s["currency"]
    lines = [
        f"# Equity Research Brief — {s['company_name']} ({s['ticker']})",
        f"*As of {s['as_of']} · Generated by the CDAZZDEV Task 1 pipeline*",
        "",
        "## Company Snapshot",
        "| Metric | Value |", "|---|---|",
        f"| Sector / Industry | {s.get('sector') or 'n/a'} / {s.get('industry') or 'n/a'} |",
        f"| Current price | {_fmt(s['current_price'])} {cur} |",
        f"| 52-week range | {_fmt(s['52_week_low'])} – {_fmt(s['52_week_high'])} "
        f"({_fmt(s['pct_below_52w_high'], 1, '%', sign=True)} from high) |",
        f"| P/E ratio | {_fmt(s['pe_ratio'])} ({s['pe_ratio_source']}) |",
        f"| Market cap | {_market_cap(s.get('market_cap'))} {cur} |",
        f"| YTD return | {_fmt(s['ytd_return_pct'], 2, '%', sign=True)} |",
        "",
        "## Technical Outlook",
        f"![Technical chart]({chart_filename})",
        "",
        *[f"- {line}" for line in technical_outlook_lines(technicals, summary)],
        "",
        "## News Sentiment Summary",
        f"Overall sentiment **{agg.label}** (score {agg.score:+.2f} on −1..+1) across {agg.n_valid} "
        f"headlines — {agg.counts['positive']} positive, {agg.counts['neutral']} neutral, "
        f"{agg.counts['negative']} negative; mean confidence {agg.mean_confidence:.2f}.",
        "",
        "**Top headlines**",
        "",
    ]
    for i, h in enumerate(top_headlines(scored), 1):
        lines.append(f"{i}. **[{h.sentiment.upper()} · {h.confidence:.2f}]** {h.headline} "
                     f"*({h.publisher or 'unknown'})* — {h.brief_reason}")
    lines += [
        "",
        f"## Recommendation: **{ts.signal.upper()}** (conviction {ts.conviction:.0%})",
        ts.justification,
        "",
        "**Indicator interactions considered**",
        "",
        *[f"- *{' + '.join(ix.indicators)}* → {ix.implication}: {ix.interpretation}"
          for ix in ts.indicator_interactions],
        "",
        "**Key risks to the call:** " + "; ".join(ts.key_risks),
        "",
        f"*Signal source: {sig.source}{f' ({sig.model})' if sig.model else ''}; horizon {ts.time_horizon}.*",
        "",
        "## Risk Disclaimer",
        f"> {DISCLAIMER}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root{--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--bg:#f4f3ef;--card:#fff;
--buy:#0f7a50;--hold:#9a6700;--sell:#b42318;--accent:#2a78d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:13px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.page{max-width:900px;margin:24px auto;background:var(--card);padding:28px 32px;
border:1px solid var(--line);border-radius:8px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:14px}
h1{font-size:20px;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;
color:var(--muted);margin:18px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--muted);font-size:12px}
.badge{padding:8px 14px;border-radius:6px;color:#fff;font-weight:700;font-size:18px;text-align:center}
.badge small{display:block;font-size:11px;font-weight:500;opacity:.9}
.Buy{background:var(--buy)}.Hold{background:var(--hold)}.Sell{background:var(--sell)}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.kpi{border:1px solid var(--line);border-radius:6px;padding:8px}
.kpi b{display:block;font-size:15px}.kpi span{color:var(--muted);font-size:11px}
img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
ul{margin:6px 0;padding-left:18px}li{margin:2px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.pill{display:inline-block;padding:0 6px;border-radius:9px;font-size:10px;font-weight:700;
color:#fff;margin-right:4px}.positive{background:var(--buy)}.negative{background:var(--sell)}
.neutral{background:#8a8984}.reason{color:var(--muted);font-size:12px}
.disclaimer{font-size:10.5px;color:var(--muted);border-top:1px solid var(--line);
margin-top:16px;padding-top:8px}
@media (max-width:700px){.kpis{grid-template-columns:repeat(2,1fr)}.cols{grid-template-columns:1fr}
.page{margin:0;padding:16px;border-radius:0}}
@media print{body{background:#fff}.page{margin:0;border:0}}
"""


def render_html(summary: dict[str, Any], technicals: dict[str, Any], scored: list[ScoredHeadline],
                agg: SentimentAggregate, sig: SignalResult, chart_png: bytes) -> str:
    e = html.escape
    s, ts = summary, sig.signal
    cur = e(s["currency"])
    img = base64.b64encode(chart_png).decode("ascii")
    kpis = [
        ("Price", f"{_fmt(s['current_price'])} {cur}"),
        ("52w high", _fmt(s["52_week_high"])), ("52w low", _fmt(s["52_week_low"])),
        ("P/E", _fmt(s["pe_ratio"])), ("YTD", _fmt(s["ytd_return_pct"], 2, "%", sign=True)),
        ("Mkt cap", _market_cap(s.get("market_cap"))),
    ]
    tech = "".join(
        f"<li>{e(line).replace('**', '')}</li>" for line in technical_outlook_lines(technicals, summary))
    heads = "".join(
        f"<li><span class='pill {h.sentiment}'>{h.sentiment.upper()} {h.confidence:.2f}</span>"
        f"{e(h.headline)} <span class='reason'>({e(h.publisher or 'unknown')}) — {e(h.brief_reason)}</span></li>"
        for h in top_headlines(scored))
    inter = "".join(f"<li><b>{e(' + '.join(ix.indicators))}</b> → {e(ix.implication)}: "
                    f"{e(ix.interpretation)}</li>" for ix in ts.indicator_interactions)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(s['ticker'])} Research Brief</title><style>{_CSS}</style></head>
<body><div class="page">
<header><div><h1>{e(s['company_name'])} ({e(s['ticker'])})</h1>
<div class="sub">Equity research brief · as of {e(s['as_of'])} · {e(s.get('sector') or '')}
{(' / ' + e(s['industry'])) if s.get('industry') else ''}</div></div>
<div class="badge {ts.signal}">{ts.signal.upper()}<small>conviction {ts.conviction:.0%}</small></div></header>
<h2>Company snapshot</h2>
<div class="kpis">{''.join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis)}</div>
<div class="sub" style="margin-top:4px">P/E source: {e(s['pe_ratio_source'])} ·
{_fmt(s['pct_below_52w_high'], 1, '%', sign=True)} from 52-week high</div>
<h2>Technical outlook</h2>
<img alt="Price with SMA50, SMA200 and Bollinger Bands; MACD; RSI" src="data:image/png;base64,{img}">
<ul>{tech}</ul>
<div class="cols"><div>
<h2>News sentiment</h2>
<p><b>{agg.label.capitalize()}</b> · score {agg.score:+.2f} (−1..+1) · {agg.n_valid} headlines
({agg.counts['positive']} pos / {agg.counts['neutral']} neu / {agg.counts['negative']} neg) ·
mean confidence {agg.mean_confidence:.2f}</p>
<ol>{heads}</ol></div><div>
<h2>Recommendation — {ts.signal}</h2>
<p>{e(ts.justification)}</p>
<ul>{inter}</ul>
<p class="reason"><b>Key risks:</b> {e('; '.join(ts.key_risks))}<br>
Source: {e(sig.source)}{(' · ' + e(sig.model)) if sig.model else ''} · horizon {e(ts.time_horizon)}</p>
</div></div>
<div class="disclaimer"><b>Risk disclaimer.</b> {e(DISCLAIMER)}</div>
</div></body></html>"""
