"""Streamlit trace viewer for logs/agent_trace.jsonl (Task 3 observability bonus).

    streamlit run task3_agentic/dashboard/app.py
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Streamlit dashboard that visualises an agent
# tool-call JSONL trace: KPIs, per-tool latency, timeline and handoffs', Date: 2026-10-07

import json
from pathlib import Path

import pandas as pd
import streamlit as st

TRACE = Path(__file__).resolve().parents[1] / "logs" / "agent_trace.jsonl"

st.set_page_config(page_title="Agent Trace", layout="wide")
st.title("Agent trace viewer")
st.caption(f"Source: {TRACE}")

if not TRACE.exists():
    st.warning("No trace file yet. Run the Task 3 notebook first.")
    st.stop()

rows = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"])

sessions = sorted(df["session_id"].unique(), key=lambda s: df.loc[df.session_id == s, "ts"].min())
session = st.selectbox("Session", sessions, index=len(sessions) - 1,
                       format_func=lambda s: f"{s} · {df.loc[df.session_id == s, 'ts'].min():%Y-%m-%d %H:%M:%S}")
d = df[df.session_id == session].sort_values("ts")
tools = d[d.event == "tool_call"].copy()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tool calls", len(tools))
c2.metric("Agents", tools["agent"].nunique() if len(tools) else 0)
c3.metric("Errors / denied", int(tools["status"].isin(["error", "denied"]).sum()) if len(tools) else 0)
c4.metric("Tool time (s)", f"{tools['duration_ms'].sum() / 1000:.1f}" if len(tools) else "0")
c5.metric("Handoffs", int((d.event == "handoff").sum()))

if len(tools):
    st.subheader("Latency by tool")
    lat = tools.groupby("tool")["duration_ms"].agg(["count", "mean", "max"]).round(0)
    lat.columns = ["calls", "mean ms", "max ms"]
    st.bar_chart(lat["mean ms"], horizontal=True)
    st.dataframe(lat, use_container_width=True)

    st.subheader("Tool-call timeline")
    tools["t (s)"] = (tools["ts"] - d["ts"].min()).dt.total_seconds().round(1)
    view = tools[["t (s)", "agent", "tool", "status", "duration_ms", "inputs", "output"]]
    agents = st.multiselect("Agents", sorted(tools["agent"].unique()), default=sorted(tools["agent"].unique()))
    st.dataframe(view[view.agent.isin(agents)], use_container_width=True, hide_index=True)

other = d[d.event != "tool_call"]
if len(other):
    st.subheader("Handoffs, cache hits and other events")
    for _, r in other.iterrows():
        label = f"{r['ts']:%H:%M:%S} · {r['event']}"
        if isinstance(r.get("sender"), str):
            label += f" · {r['sender']} → {r['receiver']} ({r.get('schema', '')})"
        with st.expander(label):
            st.code(r.get("payload", ""), language="json")
