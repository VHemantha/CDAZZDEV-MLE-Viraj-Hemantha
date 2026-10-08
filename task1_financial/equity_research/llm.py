"""Thin LLM client: provider selection, rate limiting, JSON extraction and
Pydantic-validated structured output with a self-repair loop.

Groq and OpenRouter both expose OpenAI-compatible endpoints, so the official
``openai`` SDK is used for both with a different ``base_url``.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'OpenAI-compatible client for Groq/OpenRouter
# with JSON-mode, Pydantic validation, logged validation failures and a repair re-prompt',
# Date: 2026-10-06

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeVar

from openai import APIError, APIStatusError, OpenAI
from pydantic import BaseModel, ValidationError

from . import config
from .prompts import REPAIR_USER_TEMPLATE

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

NON_RETRYABLE_STATUS = {401, 403, 404}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMUnavailableError(RuntimeError):
    """No provider has an API key configured."""


@dataclass
class ValidationEvent:
    """One logged validation failure -- kept in memory for the notebook/report."""

    timestamp: str
    task: str
    attempt: int
    error: str
    raw_excerpt: str


@dataclass
class StructuredResult:
    value: BaseModel | None
    attempts: int
    errors: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating ``` fences or prose
    around it. Raises ValueError if no object can be decoded."""
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def format_validation_error(exc: Exception, max_len: int = 500) -> str:
    """One-line, human-readable error: ``field: problem (got value); ...``.

    Pydantic's default ``str(exc)`` is multi-line and embeds documentation URLs,
    which bloats logs and wastes tokens when the error is fed back to the model.
    """
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors(include_url=False):
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            got = "" if err["type"] == "missing" else f" (got {err.get('input')!r:.60})"
            parts.append(f"{loc}: {err['msg']}{got}")
        text = f"{exc.error_count()} schema error(s): " + "; ".join(parts)
    else:
        text = f"invalid JSON: {exc}"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


class LLMClient:
    # Class-level default so subclasses / test doubles that skip __init__ still work.
    disabled_reason: str | None = None   # set on non-retryable errors (401/403/404)

    def __init__(self, provider: str | None = None):
        self.provider, self.model, self._client = self._resolve(provider)
        self._last_call = 0.0
        self.validation_events: list[ValidationEvent] = []
        self.n_calls = 0
        self.disabled_reason = None

    @staticmethod
    def _pick_model(client: OpenAI, name: str, spec: dict) -> str | None:
        """Return the configured model if the key can access it, else the first
        accessible fallback. If the catalogue can't be listed, trust the config."""
        try:
            available = {m.id for m in client.models.list().data}
        except Exception as exc:
            logger.warning("Could not list %s models (%s); using configured model.", name, exc)
            return spec["model"]
        for candidate in [spec["model"], *spec.get("fallback_models", [])]:
            if candidate in available:
                if candidate != spec["model"]:
                    logger.warning("%s model '%s' unavailable; falling back to '%s'.",
                                   name, spec["model"], candidate)
                return candidate
        logger.error("None of the configured %s models are available to this key.", name)
        return None

    @classmethod
    def _resolve(cls, provider: str | None) -> tuple[str, str, OpenAI]:
        order = [provider] if provider else list(config.LLM_PROVIDER_PRIORITY)
        for name in order:
            spec = config.LLM_PROVIDERS[name]
            key = os.getenv(spec["api_key_env"])
            if not key:
                continue
            client = OpenAI(
                api_key=key,
                base_url=spec["base_url"],
                timeout=config.LLM_REQUEST_TIMEOUT_SECONDS,
                # SDK retries 429/5xx/connection errors with exponential backoff
                max_retries=config.LLM_MAX_TRANSPORT_RETRIES,
            )
            model = cls._pick_model(client, name, spec)
            if model:
                logger.info("LLM provider: %s (%s)", name, model)
                return name, model, client
        envs = ", ".join(config.LLM_PROVIDERS[n]["api_key_env"] for n in order)
        raise LLMUnavailableError(f"No usable LLM provider; set a valid key in one of: {envs}")

    def _throttle(self) -> None:
        wait = config.LLM_MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        self._throttle()
        self.n_calls += 1
        kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, response_format={"type": "json_object"})
        if "gpt-oss" in self.model:
            # reasoning models: bound the hidden chain-of-thought budget
            kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except APIError as exc:
            # Some free models reject JSON mode; retry once without it.
            if "response_format" in str(exc).lower():
                kwargs.pop("response_format")
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""

    def structured(self, *, task: str, system: str, user: str, schema: type[T],
                   temperature: float, max_tokens: int) -> StructuredResult:
        """Call the LLM and return a validated ``schema`` instance.

        On JSON or schema errors the failure is logged, the error message is fed
        back to the model (self-repair), and the call is retried up to
        LLM_MAX_VALIDATION_RETRIES times. Transport errors are logged and end
        the attempt. Never raises -- the caller decides on a fallback when
        ``value`` is None.
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        errors: list[str] = []
        max_attempts = 1 + config.LLM_MAX_VALIDATION_RETRIES
        if self.disabled_reason:
            return StructuredResult(None, 0, [f"client disabled: {self.disabled_reason}"])
        for attempt in range(1, max_attempts + 1):
            try:
                raw = self._chat(messages, temperature, max_tokens)
            except Exception as exc:  # network / auth / quota after SDK retries
                msg = f"transport error: {type(exc).__name__}: {exc}"
                logger.error("[%s] attempt %d %s", task, attempt, msg)
                errors.append(msg)
                if isinstance(exc, APIStatusError) and exc.status_code in NON_RETRYABLE_STATUS:
                    # bad key / missing model: every further call would fail identically
                    self.disabled_reason = f"HTTP {exc.status_code} from {self.provider}"
                    logger.error("Disabling LLM client for this run (%s).", self.disabled_reason)
                return StructuredResult(None, attempt, errors)
            try:
                value = schema.model_validate(extract_json(raw))
                if errors:
                    logger.info("[%s] repaired on attempt %d", task, attempt)
                return StructuredResult(value, attempt, errors)
            except (ValueError, ValidationError) as exc:
                # ValidationError subclasses ValueError; both are schema failures here.
                msg = format_validation_error(exc)
                errors.append(msg)
                self.validation_events.append(ValidationEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    task=task, attempt=attempt, error=msg, raw_excerpt=raw[:300]))
                logger.warning("[%s] validation failed (attempt %d/%d): %s",
                               task, attempt, max_attempts, msg)
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": REPAIR_USER_TEMPLATE.substitute(error=msg)},
                ]
        logger.error("[%s] giving up after %d attempts", task, max_attempts)
        return StructuredResult(None, max_attempts, errors)
