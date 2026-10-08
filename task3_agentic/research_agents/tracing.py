"""Observability: every tool call (and agent handoff) is appended to
logs/agent_trace.jsonl and echoed to the notebook as a readable message trace."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'JSONL tracer for agent tool calls with
# truncated outputs, durations, agent context and console echo', Date: 2026-10-07

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

_current_agent: contextvars.ContextVar[str] = contextvars.ContextVar("agent", default="unknown")


def _short(obj: Any, n: int) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str, ensure_ascii=False)
    return text if len(text) <= n else text[: n - 1] + "…"


class Tracer:
    """Thread-safe append-only JSONL writer + pretty console printer."""

    def __init__(self, path: Path = config.TRACE_PATH, echo: bool = True):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.session_id = uuid.uuid4().hex[:8]
        self._lock = threading.Lock()

    # ----- context ---------------------------------------------------------
    def new_session(self) -> str:
        self.session_id = uuid.uuid4().hex[:8]
        return self.session_id

    @staticmethod
    def set_agent(name: str) -> contextvars.Token:
        return _current_agent.set(name)

    @staticmethod
    def reset_agent(token: contextvars.Token) -> None:
        _current_agent.reset(token)

    @staticmethod
    def agent() -> str:
        return _current_agent.get()

    # ----- writing ---------------------------------------------------------
    def _write(self, record: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def tool_call(self, tool: str, inputs: dict, output: Any, duration_ms: float, status: str) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "event": "tool_call",
            "agent": self.agent(),
            "tool": tool,
            "inputs": inputs,
            "output": _short(output, config.TRACE_OUTPUT_MAX_CHARS),
            "duration_ms": round(duration_ms, 1),
            "status": status,
        }
        self._write(rec)
        if self.echo:
            icon = {"ok": "✓", "error": "✗", "denied": "⛔", "empty": "∅"}.get(status, "?")
            print(f"    {icon} [{rec['agent']}] TOOL {tool}({_short(inputs, 90)}) "
                  f"-> {status} in {rec['duration_ms']:.0f} ms | {_short(output, 110)}")

    def event(self, event: str, payload: Any, **extra: Any) -> None:
        """Non-tool events: handoffs, clarification requests, cache hits."""
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "session_id": self.session_id, "event": event, "agent": self.agent(),
               "payload": _short(payload, config.TRACE_OUTPUT_MAX_CHARS), **extra}
        self._write(rec)

    # ----- console only ----------------------------------------------------
    def say(self, text: str) -> None:
        if self.echo:
            print(text)


TRACER = Tracer()


def timed() -> float:
    return time.perf_counter()
