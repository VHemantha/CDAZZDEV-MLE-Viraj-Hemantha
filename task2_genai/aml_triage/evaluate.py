"""Evaluation: generation with per-token confidence, ROUGE-L, field-level
accuracy, BERTScore, LLM-as-judge (structured JSON), grounding checks and the
manual-review sheet used for the hallucination rate."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Evaluate base vs fine-tuned triage model:
# greedy generation with perplexity, ROUGE-L, label accuracy, BERTScore, Groq LLM judge with
# Pydantic output, grounding heuristic and review CSV', Date: 2026-10-07

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from . import config
from .generate import ungrounded_figures
from .prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, STUDENT_SYSTEM_PROMPT
from .schema import Alert, Triage


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(model, tok, messages: list[dict], max_new_tokens: int = config.GEN_MAX_NEW_TOKENS) -> dict:
    """Greedy decoding (deterministic, so base vs fine-tuned is a fair comparison).
    Also returns the answer's perplexity = exp(-mean token log-prob), the
    confidence signal used by the RAG fallback."""
    import torch

    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             output_scores=True, return_dict_in_generate=True,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    logprobs = []
    for step, token_id in enumerate(gen_ids):
        if token_id.item() == tok.eos_token_id:
            break
        lp = torch.log_softmax(out.scores[step][0].float(), dim=-1)[token_id].item()
        logprobs.append(lp)
    ppl = math.exp(-sum(logprobs) / len(logprobs)) if logprobs else float("inf")
    return {"text": tok.decode(gen_ids, skip_special_tokens=True).strip(), "perplexity": ppl,
            "n_tokens": len(logprobs)}


def parse_triage(text: str) -> Triage | None:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return Triage.model_validate(json.loads(t[s:e + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None


def run_inference(model, tok, rows: list[dict], label: str) -> list[dict]:
    """Same system prompt for base and fine-tuned model (the brief's baseline)."""
    preds = []
    for i, r in enumerate(rows, 1):
        g = generate(model, tok, r["prompt"])
        preds.append({"case_id": r["case_id"], "model": label, **g})
        print(f"  [{label}] {i}/{len(rows)} {r['case_id']} ppl={g['perplexity']:.3f} tokens={g['n_tokens']}")
    return preds


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rouge_l(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def field_metrics(pred_texts: list[str], gold_texts: list[str]) -> dict[str, float]:
    """Task-level correctness, which ROUGE alone cannot capture: a fluent answer
    with the wrong action is still wrong."""
    n = len(gold_texts)
    parsed = [parse_triage(p) for p in pred_texts]
    golds = [parse_triage(g) for g in gold_texts]
    ok = [p is not None for p in parsed]
    def acc(field: str) -> float:
        return sum(1 for p, g in zip(parsed, golds) if p and getattr(p, field) == getattr(g, field)) / n
    return {"valid_json_schema": sum(ok) / n, "typology_acc": acc("typology"),
            "action_acc": acc("recommended_action"), "risk_acc": acc("risk_level")}


def bertscore_f1(preds: list[str], refs: list[str]) -> list[float]:
    from bert_score import score

    _, _, f1 = score(preds, refs, lang="en", rescale_with_baseline=True, verbose=False)
    return [float(x) for x in f1]


def grounding_flags(pred_text: str, alert: Alert) -> list[str]:
    """Automated hallucination signal: monetary figures not traceable to the alert."""
    t = parse_triage(pred_text)
    if t is None:
        return []
    return [f for s in [*t.red_flags, *t.mitigating_factors, t.rationale] for f in ungrounded_figures(s, alert)]


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #
class JudgeScore(BaseModel):
    decision_accuracy: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    reasoning_quality: int = Field(ge=1, le=5)
    format_compliance: int = Field(ge=1, le=5)
    hallucination: bool
    comment: str


def judge(alert_text: str, gold: str, candidate: str, retries: int = 2) -> JudgeScore | None:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1",
                    max_retries=8, timeout=120)
    msgs = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.substitute(alert=alert_text, gold=gold, candidate=candidate)}]
    for _ in range(retries + 1):
        resp = client.chat.completions.create(model=config.JUDGE_MODEL, messages=msgs, temperature=0,
                                              max_tokens=2000, reasoning_effort="low",
                                              response_format={"type": "json_object"})
        raw = resp.choices[0].message.content or ""
        try:
            return JudgeScore.model_validate_json(raw)
        except ValidationError as exc:
            msgs += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": f"Invalid: {str(exc)[:300]}. Return corrected JSON only."}]
            time.sleep(1)
    return None


# --------------------------------------------------------------------------- #
# Manual review sheet
# --------------------------------------------------------------------------- #
REVIEW_LABELS = ("correct", "partially_correct", "hallucinated")


def suggest_label(pred_text: str, gold_text: str, alert: Alert, judge_score: JudgeScore | None) -> tuple[str, str]:
    """Pre-label to speed up (NOT replace) the human review. Rules:
    hallucinated  = invented figure, or judge flags a fact absent from the alert, or unparseable output
    correct       = typology AND action match gold, no hallucination
    partially     = everything else (one label wrong / weak grounding)"""
    p, g = parse_triage(pred_text), parse_triage(gold_text)
    flags = grounding_flags(pred_text, alert)
    if p is None:
        return "hallucinated", "output is not valid triage JSON"
    if flags or (judge_score and judge_score.hallucination):
        why = f"ungrounded figures {flags}" if flags else f"judge: {judge_score.comment}"
        return "hallucinated", why
    if p.typology == g.typology and p.recommended_action == g.recommended_action:
        return "correct", "typology and action match gold; no ungrounded facts detected"
    return "partially_correct", (f"typology {p.typology} vs {g.typology}; "
                                 f"action {p.recommended_action} vs {g.recommended_action}")


def hallucination_rate(labels: list[str]) -> float:
    return 100.0 * sum(l == "hallucinated" for l in labels) / len(labels)


__all__ = ["generate", "parse_triage", "run_inference", "rouge_l", "field_metrics", "bertscore_f1",
           "grounding_flags", "judge", "JudgeScore", "suggest_label", "hallucination_rate",
           "REVIEW_LABELS", "STUDENT_SYSTEM_PROMPT"]
