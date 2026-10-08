"""Offline tests (no LLM / network): access control, tracing, memory, handoff
determinism and evidence validation."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest for tool allowlist enforcement, JSONL
# trace fields, brief cache, deterministic snapshots and evidence validator', Date: 2026-10-07

import json
import sys
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agents import config, memory  # noqa: E402
from research_agents.agent import ToolExecutor  # noqa: E402
from research_agents.multi_agent import snapshots_from_tools  # noqa: E402
from research_agents.schemas import Risk  # noqa: E402
from research_agents.tools import ROLE_TOOLS, traced  # noqa: E402
from research_agents.tracing import TRACER  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(TRACER, "path", tmp_path / "trace.jsonl")
    monkeypatch.setattr(TRACER, "echo", False)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    yield tmp_path / "trace.jsonl"


def test_role_tool_sets_match_brief():
    assert set(ROLE_TOOLS["data_analyst"]) == {"get_price_data", "calculate_volatility", "llm_sentiment"}
    assert set(ROLE_TOOLS["research_writer"]) == {"web_search", "get_news"}
    assert not set(ROLE_TOOLS["data_analyst"]) & set(ROLE_TOOLS["research_writer"])


def test_forbidden_tool_is_denied_not_executed(tmp_trace):
    ex = ToolExecutor("data_analyst")
    msg = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1"}])
    out = ex({"messages": [msg], "steps": 1})["messages"][0]
    assert "PermissionError" in out.content
    rec = json.loads(tmp_trace.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["status"] == "denied" and rec["tool"] == "web_search" and rec["agent"] == "unknown"


def test_trace_record_has_required_fields(tmp_trace):
    @traced
    def demo_tool(x: int) -> dict:
        return {"value": "y" * 500}

    demo_tool(x=1)
    rec = json.loads(tmp_trace.read_text(encoding="utf-8").splitlines()[-1])
    for key in ("tool", "inputs", "output", "duration_ms", "status", "ts", "session_id", "agent"):
        assert key in rec
    assert len(rec["output"]) <= config.TRACE_OUTPUT_MAX_CHARS
    assert rec["inputs"] == {"x": 1}


def test_tool_exceptions_become_observations(tmp_trace):
    @traced
    def broken_tool() -> dict:
        raise RuntimeError("boom")

    out = broken_tool()
    assert "error" in out and "boom" in out["error"]


def test_cache_roundtrip_and_miss(tmp_path):
    assert memory.load_cached("ZZZZ") is None
    memory.save_brief("ZZZZ", {"final_report": {"x": 1}})
    payload, path = memory.load_cached("ZZZZ")
    assert payload["final_report"] == {"x": 1}
    assert path.name == f"ZZZZ_{date.today().isoformat()}.json"


def test_snapshots_copy_numbers_verbatim():
    price = {"last_close": 526.61, "period_return_pct": 1.33, "max_drawdown_pct": -23.38,
             "indicators": {"pct_vs_sma_50": 6.22, "pct_vs_sma_200": 21.84, "rsi_14": 65.6, "macd_hist": 1.4}}
    vol = {"window_days": 30, "annualised_vol_pct": 22.11, "one_year_vol_pct": 32.31,
           "var_95_1d_pct": 2.84, "beta_vs_spy": 0.95}
    msgs = [ToolMessage(json.dumps(price), tool_call_id="a", name="get_price_data"),
            ToolMessage(json.dumps({"error": "x"}), tool_call_id="b", name="calculate_volatility"),
            ToolMessage(json.dumps(vol), tool_call_id="c", name="calculate_volatility")]
    snaps = snapshots_from_tools(msgs)
    assert snaps["price"].max_drawdown_pct == -23.38
    assert snaps["volatility"].annualised_vol_pct == 22.11
    assert "sentiment" not in snaps


def test_evidence_must_cite_source_and_figure():
    ok = {"title": "Antitrust risk", "description": "FTC probe could constrain Azure bundling.",
          "evidence": ["[web_search] 'Microsoft Antitrust Probe Widens'", "[calculate_volatility] vol 22.1%"],
          "likelihood": "medium"}
    assert Risk.model_validate(ok)
    # Agent B has no price tools: citing Agent A's typed handoff is legitimate evidence
    assert Risk.model_validate({**ok, "evidence": ["[DataBrief] RSI 68.21, 30d vol 22.1%"]})
    with pytest.raises(ValidationError):
        Risk.model_validate({**ok, "evidence": ["Beta close to 1 means it moves with the market"]})
    with pytest.raises(ValidationError):
        Risk.model_validate({**ok, "evidence": ["[get_price_data] trend is strong"]})


def test_compact_history_keeps_recent_observations_intact():
    from research_agents.agent import compact_history
    msgs = [ToolMessage("x" * 2000, tool_call_id=str(i), name="get_news") for i in range(4)]
    out = compact_history(msgs)
    assert [len(m.content) for m in out[-2:]] == [2000, 2000]
    assert all(m.content.endswith("(compacted)") for m in out[:2])
    assert msgs[0].content == "x" * 2000          # original state is not mutated
