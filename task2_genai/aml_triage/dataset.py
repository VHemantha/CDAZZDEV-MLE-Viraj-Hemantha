"""Dataset engineering: chat formatting, stratified 80/10/10 split and the
diversity report (length distribution, label/topic frequencies, near-duplicates)."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Chat-format, stratified split and diversity
# metrics (length histogram, keyword frequency, shingle Jaccard near-duplicates) for a JSONL
# dataset', Date: 2026-10-07

from __future__ import annotations

import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from . import config
from .prompts import STUDENT_SYSTEM_PROMPT
from .schema import Case, render_alert, render_triage

STOPWORDS = set("""a an and are as at be by for from has have in into is it its of on or that the
their this to was were with within than over under per same each all any more most no not only
other such so very can will one two three customer account accounts transaction transactions""".split())


def load_cases(path: Path = config.RAW_PATH) -> list[Case]:
    return [Case.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_chat(case: Case) -> dict:
    """Conversational prompt/completion record. TRL applies the base model's own
    chat template (Qwen2.5 ChatML) at train time; keeping prompt and completion
    separate lets the loss be computed on the assistant turn only."""
    return {
        "case_id": case.case_id,
        "typology": case.triage.typology,
        "recommended_action": case.triage.recommended_action,
        "prompt": [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                   {"role": "user", "content": render_alert(case.alert)}],
        "completion": [{"role": "assistant", "content": render_triage(case.triage)}],
    }


def stratified_split(cases: list[Case], ratios=config.SPLIT_RATIOS,
                     seed: int = config.RANDOM_SEED) -> dict[str, list[Case]]:
    """Split each typology 80/10/10 so every split covers every label, then
    top up val/test to their exact global sizes from the training pool."""
    rng = random.Random(seed)
    n = len(cases)
    n_val, n_test = round(n * ratios[1]), round(n * ratios[2])
    by_typ: dict[str, list[Case]] = {}
    for c in cases:
        by_typ.setdefault(c.triage.typology, []).append(c)
    train, val, test = [], [], []
    for group in by_typ.values():
        rng.shuffle(group)
        k_test = int(len(group) * ratios[2])
        k_val = int(len(group) * ratios[1])
        test += group[:k_test]
        val += group[k_test:k_test + k_val]
        train += group[k_test + k_val:]
    rng.shuffle(train)
    while len(test) < n_test:
        test.append(train.pop())
    while len(val) < n_val:
        val.append(train.pop())
    return {"train": train, "val": val, "test": test}


def write_splits(splits: dict[str, list[Case]]) -> dict[str, int]:
    sizes = {}
    for name, items in splits.items():
        path = config.SPLIT_PATHS[name]
        with path.open("w", encoding="utf-8") as fh:
            for c in items:
                fh.write(json.dumps(to_chat(c), ensure_ascii=False) + "\n")
        sizes[name] = len(items)
    return sizes


def read_split(name: str) -> list[dict]:
    return [json.loads(l) for l in config.SPLIT_PATHS[name].read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Diversity metrics
# --------------------------------------------------------------------------- #
def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z_\-']+", text.lower())


def _shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def near_duplicate_stats(texts: list[str], k: int = 5, threshold: float = 0.5) -> dict:
    """Pairwise Jaccard similarity of word k-gram shingles (exhaustive: n=150 ->
    11,175 pairs). Template-variation datasets typically score > 0.6."""
    sh = [_shingles(t, k) for t in texts]
    sims = [len(a & b) / len(a | b) for a, b in itertools.combinations(sh, 2) if a | b]
    sims_arr = np.array(sims)
    return {"pairs": len(sims), "mean_jaccard": float(sims_arr.mean()), "p95_jaccard": float(np.percentile(sims_arr, 95)),
            "max_jaccard": float(sims_arr.max()), f"pairs_above_{threshold}": int((sims_arr > threshold).sum()),
            "_all": sims_arr}


def distinct_n(texts: list[str], n: int) -> float:
    grams, total = set(), 0
    for t in texts:
        w = re.findall(r"\w+", t.lower())
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / total if total else 0.0


def keyword_frequency(cases: list[Case], top: int = 25) -> list[tuple[str, int]]:
    """Content words across rule names, red flags and rationales (topic spread)."""
    cnt: Counter = Counter()
    for c in cases:
        text = " ".join([c.alert.rule_triggered, *c.triage.red_flags, c.triage.rationale])
        cnt.update(w for w in _words(text) if w not in STOPWORDS and len(w) > 3)
    return cnt.most_common(top)


def diversity_report(cases: list[Case]) -> dict:
    prompts = [render_alert(c.alert) for c in cases]
    answers = [render_triage(c.triage) for c in cases]
    p_words = np.array([len(p.split()) for p in prompts])
    a_words = np.array([len(a.split()) for a in answers])
    countries = Counter(t.counterparty_country for c in cases for t in c.alert.transactions)
    return {
        "n_cases": len(cases),
        "prompt_words": {"mean": float(p_words.mean()), "std": float(p_words.std()), "min": int(p_words.min()),
                         "p50": float(np.median(p_words)), "max": int(p_words.max())},
        "answer_words": {"mean": float(a_words.mean()), "min": int(a_words.min()), "max": int(a_words.max())},
        "typology": Counter(c.triage.typology for c in cases),
        "recommended_action": Counter(c.triage.recommended_action for c in cases),
        "risk_level": Counter(c.triage.risk_level for c in cases),
        "customer_segment": Counter(c.seed["customer_segment"] for c in cases),
        "difficulty": Counter(c.seed["difficulty"] for c in cases),
        "teacher": Counter(c.teacher or "unknown" for c in cases),
        "n_distinct_counterparty_countries": len(countries),
        "n_distinct_currencies": len({t.currency for c in cases for t in c.alert.transactions}),
        "n_distinct_rules": len({c.alert.rule_triggered.lower() for c in cases}),
        "distinct_1": distinct_n(prompts, 1), "distinct_2": distinct_n(prompts, 2),
        "near_duplicates": near_duplicate_stats(prompts),
        "top_keywords": keyword_frequency(cases),
        "_prompt_words": p_words,
    }
