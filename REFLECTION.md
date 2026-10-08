# Reflection

## Architectural decisions

**Common to all tasks.** Each task is a small, tested Python package; the notebook only walks through it. No prompt text sits in the logic, and no LLM output is used before Pydantic validation. Validation failures are logged and sent back to the model for repair, and anything that still fails takes a *labelled* fallback.

**Task 1.** The indicators are written by hand and checked against plain-loop reference implementations. RSI uses Wilder's SMA-seeded smoothing. The signal prompt gets *relational* features (distance from each SMA, MACD acceleration, squeeze percentile), and the schema requires at least two multi-indicator interactions, so "reasoning over combinations" is enforced by structure, not only requested. Sentiment is weighted by confidence and recency, and failed headlines get zero weight.

**Task 2.** I chose AML alert triage because it is financial-domain, has objective labels (typology, action), and gives hallucination a concrete meaning: a fact absent from the alert. Diversity comes from stratified seeds (typology × segment × jurisdiction × difficulty). Quality comes from an arithmetic grounding gate. In the teacher's first trial batch I found a gold label quoting "1.07M" for a 106,501.25 total. The validator now expands abbreviations and accepts only figures that match an amount, a subset-sum of transactions, or a regulatory threshold. QLoRA settings follow the QLoRA paper (NF4, all linear layers, r=16, lr 2e-4), with completion-only loss and per-epoch validation. Every value is justified in one dict, and the notebook prints that dict. A CPU smoke test of the real training path found that current TRL defaults to bf16, which crashes on a T4.

**Task 3.** The ReAct loop is a hand-built LangGraph graph rather than a prebuilt agent, so that tool permissions are enforced when a call *executes*: a forged call to a forbidden tool is denied and traced. Numbers handed between agents are copied from tool JSON, never re-typed by an LLM, which removed a mis-copied drawdown I saw in development. Evidence items must cite their source tool and a figure. The trace shows the provider's reasoning summary, so observe→replan is real, not a scripted message.

## What I would improve with more time

- **Task 1:** backtest the signal against forward returns, and calibrate the sentiment confidence on hand-labelled headlines.
- **Task 2:** a larger test set (15 is small for confident accuracy estimates); DPO on grounded vs. hallucinated answer pairs; more borderline cases for the most-confused typology pairs; human double-review of gold labels.
- **Task 3:** an evaluation harness that scores reports for factual consistency against the trace; parallel tool calls; LangSmith-style span tracing.

## Limitations encountered

- **Free-tier quotas shaped the design.** Groq allows about 8K tokens per minute and 200K per day per model. The daily limit ran out mid-generation, so the teacher pool rotates models and records which teacher wrote each case. Task 3 roles fail over from gpt-oss-120b to gpt-oss-20b. Groq counts `max_tokens` against the per-minute budget, which forced concise output schemas.
- **Provider quirks:** OpenRouter delisted the free gpt-oss-120b mid-run; gpt-oss sometimes emits malformed tool names (now retried with a nudge); a network outage broke a run, so pipeline nodes now have retry policies.
- **No local GPU:** training and evaluation for Task 2 run on Colab. Hallucination labels need a human reviewer; the notebook only pre-labels.
- **Data limits:** yfinance news is often empty, so RSS fallbacks are used, and headline-only sentiment ignores article bodies. Synthetic alerts are realistic but simpler than production alerts.
