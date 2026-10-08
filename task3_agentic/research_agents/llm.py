"""Chat-model factory and a validated structured-output helper."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'ChatOpenAI factory for Groq and a JSON +
# Pydantic structured invoke with one repair turn', Date: 2026-10-07

from __future__ import annotations

import json
import logging
import os
import re
from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, LengthFinishReasonError, RateLimitError
from pydantic import BaseModel, ValidationError

from . import config

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI that keeps the provider's ``reasoning`` text.

    Groq returns gpt-oss's chain-of-thought summary in ``message.reasoning``;
    langchain-openai discards unknown fields. We copy it into
    ``additional_kwargs["reasoning"]`` so the trace can show WHY each tool was
    chosen (the observe -> re-plan evidence). It is display-only: it is never
    sent back to the API.
    """

    def _create_chat_result(self, response, generation_info=None):  # type: ignore[override]
        result = super()._create_chat_result(response, generation_info)
        try:
            choices = response.choices if hasattr(response, "choices") else response.get("choices", [])
            for gen, choice in zip(result.generations, choices):
                msg = choice.message if hasattr(choice, "message") else choice.get("message", {})
                reasoning = getattr(msg, "reasoning", None) if not isinstance(msg, dict) else msg.get("reasoning")
                if reasoning:
                    gen.message.additional_kwargs["reasoning"] = reasoning
        except Exception:  # never let observability break inference
            pass
        return result


# Errors a different model/endpoint attempt can fix. langchain-openai re-raises the SDK's
# connection/timeout errors as subclasses of these, so they are caught too.
TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


def _single_chat(model: str, json_mode: bool, max_retries: int) -> ReasoningChatOpenAI:
    key = os.getenv(config.LLM_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {config.LLM_API_KEY_ENV} (Colab secret or .env)")
    kwargs = dict(model=model, api_key=key, base_url=config.LLM_BASE_URL,
                  temperature=config.LLM_TEMPERATURE, max_retries=max_retries,
                  timeout=config.LLM_TIMEOUT_S, max_tokens=config.LLM_MAX_OUTPUT_TOKENS)
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    if "qwen3" in model:
        # Qwen3 thinks in <think> tags; "parsed" moves it to message.reasoning (kept for the trace)
        kwargs["extra_body"] = {"reasoning_format": "parsed"}
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ReasoningChatOpenAI(**kwargs)


def _chain(role: str, json_mode: bool) -> list[ReasoningChatOpenAI]:
    """Preferred model + fallbacks. Every model except the last gets few SDK retries
    so an exhausted DAILY quota (which retrying cannot fix) fails over quickly."""
    models = [config.MODELS[role], *[m for m in config.FALLBACK_MODELS if m != config.MODELS[role]]]
    last = len(models) - 1
    return [_single_chat(m, json_mode, config.PRIMARY_MAX_RETRIES if i < last else config.LLM_MAX_RETRIES)
            for i, m in enumerate(models)]


def make_chat(role: str, json_mode: bool = False, tools: list | None = None) -> Runnable:
    """Chat model for ``role`` (optionally tool-bound) with automatic failover to
    the fallback model on rate-limit / quota errors (RateLimitError)."""
    chats = _chain(role, json_mode)
    runnables = [c.bind_tools(tools) if tools else c for c in chats]
    if len(runnables) == 1:
        return runnables[0]
    return runnables[0].with_fallbacks(runnables[1:], exceptions_to_handle=TRANSIENT_ERRORS)


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def structured_invoke(llm: Runnable, system: str, context: list[BaseMessage] | str,
                      schema: type[T], instruction: str, max_repairs: int = 2) -> T:
    """Ask for JSON matching ``schema``; on failure feed the validation error back.

    Raises ValueError after ``max_repairs`` -- callers decide how to degrade.
    """
    schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    msgs: list[BaseMessage] = [SystemMessage(system)]
    if isinstance(context, str):
        msgs.append(HumanMessage(context))
    else:
        msgs += context
    msgs.append(HumanMessage(f"{instruction}\nReturn ONLY a JSON object valid against this JSON "
                             f"schema (no markdown):\n{schema_json}"))
    last_err = ""
    for attempt in range(1 + max_repairs):
        try:
            reply = llm.invoke(msgs)
        except LengthFinishReasonError:
            # The JSON was cut off at max_tokens: ask again for a shorter answer
            # (raising the cap is not an option: prompt + max_tokens must fit the 8K TPM budget).
            last_err = "response truncated at the output-token limit"
            logger.warning("structured output for %s truncated (attempt %d); asking for brevity",
                           schema.__name__, attempt + 1)
            msgs.append(HumanMessage("Your previous answer was cut off at the length limit. Answer again "
                                     "with the same JSON keys but much more concisely (short sentences, "
                                     "at most 2 evidence items per risk)."))
            continue
        try:
            return schema.model_validate(_extract_json(str(reply.content)))
        except (ValueError, ValidationError) as exc:
            last_err = str(exc)[:600]
            logger.warning("structured output for %s failed (attempt %d): %s",
                           schema.__name__, attempt + 1, last_err[:200])
            msgs += [reply, HumanMessage(f"That failed validation: {last_err}\nReturn a corrected JSON object only.")]
    raise ValueError(f"{schema.__name__} could not be validated: {last_err}")
