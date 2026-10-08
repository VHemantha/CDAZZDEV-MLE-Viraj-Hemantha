"""Teacher-model dataset generation: stratified scenario seeds -> batched teacher
calls -> Pydantic validation + consistency checks -> append-only JSONL (resumable)."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Stratified seed sampling and resumable teacher
# generation for AML triage cases with validation', Date: 2026-10-07

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from . import config
from .prompts import TEACHER_SYSTEM_PROMPT, TEACHER_USER_TEMPLATE
from .schema import Alert, Case, Triage

logger = logging.getLogger(__name__)


def build_seeds(n: int = config.N_CASES, rng_seed: int = config.RANDOM_SEED) -> list[dict]:
    """Stratified seeds: typologies are balanced (with a fixed share of benign
    cases), while segment / jurisdiction / difficulty / size are sampled so no
    two seeds are alike. This is what guarantees diversity *by construction*."""
    rng = random.Random(rng_seed)
    n_benign = round(n * config.NO_SUSPICIOUS_SHARE)
    suspicious = [t for t in config.TYPOLOGIES if t != "no_suspicious_activity"]
    typologies = ["no_suspicious_activity"] * n_benign
    typologies += [suspicious[i % len(suspicious)] for i in range(n - n_benign)]
    rng.shuffle(typologies)
    seeds = []
    for i, typ in enumerate(typologies):
        seeds.append({
            "seed_id": i,
            "true_typology": typ,
            "customer_segment": rng.choice(config.CUSTOMER_SEGMENTS),
            "jurisdiction_focus": rng.sample(config.JURISDICTIONS, k=rng.choice([1, 2])),
            "difficulty": rng.choices(config.DIFFICULTY, weights=[0.45, 0.35, 0.20])[0],
            "n_transactions": rng.randint(config.MIN_TXNS, config.MAX_TXNS),
            "prior_alerts_hint": rng.choice([0, 0, 0, 1, 2, 4]),
        })
    return seeds


class TeacherPool:
    """Rotates through config.TEACHER_POOL when a provider's daily quota (or
    key) is exhausted. Per-minute 429s are handled by the SDK's own retries."""

    QUOTA_MARKERS = ("per day", "tpd", "rpd", "daily", "free-models-per-day", "quota")

    def __init__(self) -> None:
        self.members = [m for m in config.TEACHER_POOL if os.getenv(m["api_key_env"])]
        if not self.members:
            raise RuntimeError("Set GROQ_API_KEY and/or OPENROUTER_API_KEY to generate data")
        # start position lets a resumed run skip a teacher whose daily quota is reserved/used
        self.idx = min(int(os.getenv("TEACHER_START", "0")), len(self.members) - 1)
        self._clients: dict[str, OpenAI] = {}

    @property
    def current(self) -> dict:
        return self.members[self.idx]

    def client(self) -> OpenAI:
        m = self.current
        if m["name"] not in self._clients:
            self._clients[m["name"]] = OpenAI(api_key=os.getenv(m["api_key_env"]), base_url=m["base_url"],
                                              max_retries=6, timeout=240)
        return self._clients[m["name"]]

    def is_quota_error(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        if status in (401, 402, 403, 404):          # bad key / no credit / model delisted
            return True
        return status == 429 and any(k in text for k in self.QUOTA_MARKERS)

    def rotate(self) -> bool:
        logger.warning("teacher %s exhausted; rotating", self.current["name"])
        self.idx += 1
        return self.idx < len(self.members)

    def complete(self, messages: list[dict]) -> str:
        m = self.current
        kwargs = dict(model=m["model"], messages=messages, temperature=config.TEACHER_TEMPERATURE,
                      max_tokens=m.get("max_tokens", config.TEACHER_MAX_TOKENS), response_format={"type": "json_object"})
        if "gpt-oss" in m["model"]:
            if "groq" in m["base_url"]:
                kwargs["reasoning_effort"] = config.TEACHER_REASONING_EFFORT
            else:
                kwargs["extra_body"] = {"reasoning": {"effort": config.TEACHER_REASONING_EFFORT}}
        elif "qwen3" in m["model"] and "groq" in m["base_url"]:
            # keep Qwen3's <think> trace out of the JSON content
            kwargs["extra_body"] = {"reasoning_format": "hidden"}
        resp = self.client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def consistency_errors(seed: dict, case: Case) -> list[str]:
    """Checks beyond the schema: label fidelity to the seed and grounding."""
    errs = []
    t = case.triage
    if t.typology != seed["true_typology"]:
        errs.append(f"typology {t.typology} != seed {seed['true_typology']}")
    if t.recommended_action == "escalate_sar" and t.risk_level != "high":
        errs.append("escalate_sar requires risk_level high")
    if t.recommended_action == "close_no_action" and t.risk_level == "high":
        errs.append("close_no_action contradicts risk_level high")
    if t.typology != "no_suspicious_activity" and not t.red_flags:
        errs.append("suspicious typology without red flags")
    for text in [*t.red_flags, *t.mitigating_factors, t.rationale]:
        errs += [f"ungrounded figure: {raw!r}" for raw in ungrounded_figures(text, case.alert)]
    return errs


# Regulatory thresholds an expert may legitimately quote from domain knowledge.
REGULATORY_THRESHOLDS = {1_000, 3_000, 5_000, 10_000, 15_000}
_FIGURE_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k|K|m|M|mn|million|bn|billion)?\b(?!\s*%)")
_MULTIPLIER = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "mn": 1e6, "million": 1e6,
               "bn": 1e9, "billion": 1e9}


def ungrounded_figures(text: str, alert: Alert) -> list[str]:
    """Monetary-looking figures in ``text`` that cannot be traced to the alert.

    A figure is grounded if it equals a transaction amount / declared turnover,
    a plausible total of transactions, or a well-known regulatory threshold.
    Small numbers (< 1,000: counts, days, percentages) and years are ignored.
    Abbreviations ("1.07M", "100k") are expanded first - that is exactly how
    the teacher hid a 10x error in the first trial batch.
    """
    known = {round(t.amount, 2) for t in alert.transactions}
    known.add(round(alert.customer.declared_monthly_turnover, 2))
    bad = []
    for m in _FIGURE_RE.finditer(text):
        raw, suffix = m.group(1), m.group(2)
        value = float(raw.replace(",", "")) * _MULTIPLIER.get(suffix or "", 1.0)
        if value < 1_000 or (suffix is None and re.fullmatch(r"20\d\d", raw)):
            continue
        if round(value, 2) in known or value in REGULATORY_THRESHOLDS:
            continue
        if _is_plausible_total(f"{value:.2f}", alert):
            continue
        bad.append(m.group(0).strip())
    return bad


def _is_plausible_total(num: str, alert: Alert) -> bool:
    """A cited figure may be a total of several transactions (e.g. "three
    deposits totalling 27,450.00"). Accept it if it matches - within rounding -
    the sum of ANY subset of transaction amounts, or the net in-out flow.
    With <= 12 transactions that is at most 4,096 sums, so brute force is fine."""
    value = float(num.replace(",", ""))
    amounts = [t.amount for t in alert.transactions]
    tol = max(1.0, 0.001 * value)
    sums = {0.0}
    for a in amounts:
        sums |= {s + a for s in sums}
    net = sum(t.amount if t.direction == "in" else -t.amount for t in alert.transactions)
    return any(abs(value - s) <= tol for s in sums) or abs(value - abs(net)) <= tol


def _parse(raw: str) -> list[dict]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start : end + 1])
    return obj.get("cases", [])


def generate(out_path: Path = config.RAW_PATH, n: int = config.N_CASES) -> int:
    """Generate until ``n`` valid cases exist in ``out_path``. Resumable: seeds
    already present are skipped, so a rate-limit crash costs nothing."""
    seeds = build_seeds(n)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["seed"]["seed_id"])
    pending = [s for s in seeds if s["seed_id"] not in done]
    logger.info("%d/%d cases present, %d to generate", len(done), n, len(pending))
    pool = TeacherPool()
    attempts: dict[int, int] = {}
    transport_failures = 0
    while pending:
        size = pool.current.get("cases_per_call", config.CASES_PER_CALL)
        batch, pending = pending[:size], pending[size:]
        user = TEACHER_USER_TEMPLATE.substitute(seeds=json.dumps(batch, indent=1))
        teacher = pool.current["name"]
        try:
            raw = pool.complete([{"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                                 {"role": "user", "content": user}])
            produced = {c.get("seed_id"): c for c in _parse(raw)}
        except Exception as exc:  # transport or JSON error -> retry whole batch later
            if pool.is_quota_error(exc):
                logger.warning("teacher %s unavailable: %s", teacher, str(exc)[:300])
                pending = batch + pending
                if not pool.rotate():
                    logger.error("all teachers exhausted; resume later (progress is saved)")
                    break
                continue
            # Transport/JSON failures are not the seed's fault: re-queue without
            # spending its attempts; repeated failures mean the teacher is unusable.
            transport_failures += 1
            logger.warning("batch %s failed on %s (%d in a row): %s", [s["seed_id"] for s in batch],
                           teacher, transport_failures, str(exc)[:200])
            pending = batch + pending
            if transport_failures >= 5:
                transport_failures = 0
                if not pool.rotate():
                    logger.error("all teachers exhausted; resume later (progress is saved)")
                    break
            # per-minute windows (TPM/OTPM) and network blips both clear with time
            time.sleep(30 if getattr(exc, "status_code", None) == 429 else 10)
            continue
        transport_failures = 0
        retry = []
        with out_path.open("a", encoding="utf-8") as fh:
            for seed in batch:
                sid = seed["seed_id"]
                item = produced.get(sid)
                try:
                    if item is None:
                        raise ValueError("missing from teacher output")
                    alert = Alert.model_validate({**item["alert"], "alert_id": f"TM-{26000 + sid:05d}"})
                    case = Case(case_id=f"case-{sid:04d}", seed=seed, teacher=teacher, alert=alert,
                                triage=Triage.model_validate(item["triage"]))
                    errs = [e for e in consistency_errors(seed, case) if e]
                    if errs:
                        raise ValueError("; ".join(errs))
                    fh.write(case.model_dump_json() + "\n")
                except (ValidationError, ValueError, KeyError, TypeError) as exc:
                    attempts[sid] = attempts.get(sid, 0) + 1
                    logger.info("seed %d rejected (attempt %d): %s", sid, attempts[sid], str(exc)[:160])
                    if attempts[sid] < 4:
                        retry.append(seed)
        pending += retry
        logger.info("progress: %d remaining", len(pending))
    return sum(1 for _ in out_path.open(encoding="utf-8"))
