# Citations

This file follows Section 2.2 of the assessment brief. Inline `# AI-ASSISTED:` comments also appear at the top of every source file.

## AI assistance

| File(s) | Citation |
|---|---|
| `task1_financial/equity_research/config.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1 of the CDAZZDEV Senior MLE assessment - centralise all pipeline constants in a config module', Date: 2026-10-06` |
| `task1_financial/equity_research/indicators.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement SMA, RSI with Wilder smoothing, MACD(12,26,9) and Bollinger Bands(20,2) from first principles with pandas/numpy, no TA-Lib', Date: 2026-10-06` |
| `task1_financial/equity_research/data.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write a robust yfinance OHLCV + fundamentals fetcher with relative date windows, retries and null handling', Date: 2026-10-06` |
| `task1_financial/equity_research/summary.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build a summary dict (price, 52w range, P/E, YTD return) and a weighted multi-indicator momentum signal with None-safe handling', Date: 2026-10-06` |
| `task1_financial/equity_research/news.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Fetch >=10 recent headlines for a ticker from yfinance news with Yahoo/Google News RSS fallbacks, normalise and de-duplicate', Date: 2026-10-06` |
| `task1_financial/equity_research/schemas.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic v2 schemas for per-headline sentiment and a Buy/Hold/Sell signal with a 3-5 sentence justification validator', Date: 2026-10-06` |
| `task1_financial/equity_research/prompts.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write system/user prompt templates for financial headline sentiment and an indicator-confluence Buy/Hold/Sell signal', Date: 2026-10-06` |
| `task1_financial/equity_research/llm.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'OpenAI-compatible client for Groq/OpenRouter with JSON-mode, Pydantic validation, logged validation failures and a repair re-prompt', Date: 2026-10-06` |
| `task1_financial/equity_research/sentiment.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Score each headline with an LLM into validated JSON and aggregate with confidence x recency weighting', Date: 2026-10-06` |
| `task1_financial/equity_research/recommendation.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate a validated Buy/Hold/Sell signal from technical snapshot + sentiment via LLM, falling back to the momentum composite', Date: 2026-10-06` |
| `task1_financial/equity_research/report.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Render a one-page equity research brief in Markdown and styled HTML with an embedded 3-panel matplotlib chart (price/SMA/Bollinger, MACD, RSI)', Date: 2026-10-06` |
| `task1_financial/equity_research/pipeline.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Orchestrate data pipeline, LLM sentiment/signal and report rendering with file logging and JSON artefacts', Date: 2026-10-06` |
| `task1_financial/tests/*.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest suite validating SMA/RSI/MACD/Bollinger against naive reference implementations and edge cases', Date: 2026-10-06` |
| `task1_financial/scripts/build_notebook.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate a Colab-ready walkthrough notebook for the Task 1 equity research pipeline with nbformat', Date: 2026-10-06` |
| `task2_genai/aml_triage/config.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Config for an AML alert-triage fine-tuning pipeline (teacher generation, QLoRA training, evaluation)', Date: 2026-10-07` |
| `task2_genai/aml_triage/schema.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schemas for AML alerts and triage outputs with a deterministic alert-to-text renderer', Date: 2026-10-07` |
| `task2_genai/aml_triage/prompts.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Teacher, student, judge and RAG prompts for AML transaction-alert triage', Date: 2026-10-07` |
| `task2_genai/aml_triage/generate.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Stratified seed sampling and resumable teacher generation for AML triage cases with validation', Date: 2026-10-07` |
| `task2_genai/aml_triage/dataset.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Chat-format, stratified split and diversity metrics (length histogram, keyword frequency, shingle Jaccard near-duplicates) for a JSONL dataset', Date: 2026-10-07` |
| `task2_genai/aml_triage/train.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'QLoRA SFT with TRL/PEFT/bitsandbytes on a T4, justified hyperparameters, per-epoch train/val loss logging, merge_and_unload', Date: 2026-10-07` |
| `task2_genai/aml_triage/evaluate.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Evaluate base vs fine-tuned triage model: greedy generation with perplexity, ROUGE-L, label accuracy, BERTScore, Groq LLM judge with Pydantic output, grounding heuristic and review CSV', Date: 2026-10-07` |
| `task2_genai/aml_triage/rag.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Perplexity-gated RAG fallback with ChromaDB over a typology handbook and training cases', Date: 2026-10-07` |
| `task2_genai/data/knowledge_base/*.md` | AI-ASSISTED: Claude (claude-opus-5-5) wrote this AML typology handbook text from general AML/FATF typology knowledge, 2026-10-07 |
| `task2_genai/scripts/*.py`, `task2_genai/tests/*.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'CLI for resumable teacher generation; Colab notebook generator; pytest for the data pipeline', Date: 2026-10-07` |
| `task3_agentic/research_agents/*.py` (config, tracing, schemas, llm, tools, agent, prompts, memory, research, multi_agent, render) | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph multi-agent financial research system: traced tools, hand-built ReAct graph with execution-time tool allowlist, Pydantic handoffs, critique loop, checkpointer memory and JSON cache', Date: 2026-10-07` (each file carries its own specific prompt inline) |
| `task3_agentic/dashboard/app.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Streamlit dashboard that visualises an agent tool-call JSONL trace: KPIs, per-tool latency, timeline and handoffs', Date: 2026-10-07` |
| `task3_agentic/scripts/*.py`, `task3_agentic/tests/*.py` | `# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Task 3 notebook generator; pytest for tool allowlist, trace fields, cache, deterministic snapshots and evidence validator', Date: 2026-10-07` |
| Documentation (`README.md` files, this file, a first draft of `REFLECTION.md`) | AI-drafted with Claude (claude-opus-5-5), then reviewed and edited by the author, 2026-10-06/07 |

## Runtime LLM usage (part of the product, not code generation)

| Purpose | Model / provider | Prompt |
|---|---|---|
| Per-headline sentiment | `openai/gpt-oss-120b` via Groq (the client checks the model is available and falls back to `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, or OpenRouter) | `SENTIMENT_SYSTEM_PROMPT` in `task1_financial/equity_research/prompts.py` |
| Buy/Hold/Sell signal | same | `SIGNAL_SYSTEM_PROMPT` in the same file (also printed in full in the notebook appendix) |
| Task 2 teacher (synthetic data) | `openai/gpt-oss-120b` (Groq), then `qwen/qwen3.8-27b` (Groq) and `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter) when daily quotas ran out. Teacher recorded per case in `raw_cases.jsonl`. | `TEACHER_SYSTEM_PROMPT` in `task2_genai/aml_triage/prompts.py`, printed in full in the notebook (§2A.1 and Appendix) |
| Task 2 LLM-as-judge | `openai/gpt-oss-120b` (Groq) | `JUDGE_SYSTEM_PROMPT` in the same file |
| Task 3 agents and sentiment tool | `openai/gpt-oss-120b` with automatic failover to `openai/gpt-oss-20b` (Groq) | `task3_agentic/research_agents/prompts.py` |

## Open-source models and libraries used (not modified)

- `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0): Task 2 student base model.
- Hugging Face `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`; `rouge_score`; `bert_score`; `chromadb`; LangGraph / LangChain; `ddgs` (DuckDuckGo search); Streamlit.
- QLoRA method and recommended defaults: Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023).

## Algorithms and references (definitions only; no code was copied)

- J. Welles Wilder Jr., *New Concepts in Technical Trading Systems* (1978): RSI and Wilder smoothing.
- Gerald Appel, *Technical Analysis: Power Tools for Active Investors* (2005): MACD (12, 26, 9).
- John Bollinger, *Bollinger on Bollinger Bands* (2001): bands use the population standard deviation; %B and bandwidth.
- yfinance (https://github.com/ranaroussi/yfinance): used as a library, along with its `Ticker.news` response format.
- Chart colors: the light-mode categorical palette from the Claude Code `dataviz` skill reference palette.
