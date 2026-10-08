# CDAZZDEV Senior MLE Technical Assessment

| Task | Folder | Notebook | Highlights |
|---|---|---|---|
| **1 · Financial AI**: LLM equity research assistant | [`task1_financial/`](task1_financial/) | [Task1_Equity_Research.ipynb](task1_financial/notebooks/Task1_Equity_Research.ipynb) | Indicators written by hand and tested against reference implementations; Pydantic-validated LLM output with logged self-repair; one-page HTML brief (bonus) |
| **2 · Generative AI**: QLoRA fine-tuning for AML alert triage | [`task2_genai/`](task2_genai/) | [Task2_AML_Triage_QLoRA.ipynb](task2_genai/notebooks/Task2_AML_Triage_QLoRA.ipynb) (Colab T4) | Teacher data with an arithmetic grounding gate; hyperparameter justification table; ROUGE-L, BERTScore, LLM judge; RAG fallback (bonus) |
| **3 · Agentic workflows**: multi-agent research system | [`task3_agentic/`](task3_agentic/) | [Task3_Multi_Agent_Research.ipynb](task3_agentic/notebooks/Task3_Multi_Agent_Research.ipynb) | LangGraph ReAct with tool allowlists enforced at execution time; Pydantic handoffs; critique loop; memory; [`agent_trace.jsonl`](task3_agentic/logs/agent_trace.jsonl); Streamlit dashboard (bonus) |

- [`CITATIONS.md`](CITATIONS.md): AI-tool usage, teacher and judge models, and libraries
- [`REFLECTION.md`](REFLECTION.md): decisions, improvements, and limitations (≤ 600 words)

## Quick start

```bash
cp .env.example .env          # add GROQ_API_KEY (free, console.groq.com); .env is git-ignored
pip install -r task1_financial/requirements.txt -r task3_agentic/requirements.txt
(cd task1_financial && python -m pytest -q)    # 21 tests
(cd task2_genai && python -m pytest -q)        # 10 tests (data pipeline; training runs on Colab)
(cd task3_agentic && python -m pytest -q)      # 7 tests
```

Each task folder has a README with a Colab badge. Only free-tier services are used (yfinance, RSS, DuckDuckGo, Groq, OpenRouter free models, Colab T4, Hugging Face Hub). The repository contains no credentials.
