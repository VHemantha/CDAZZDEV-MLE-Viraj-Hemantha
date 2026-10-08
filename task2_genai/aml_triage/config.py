"""Task 2 configuration: paths, label spaces, generation and training constants."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Config for an AML alert-triage fine-tuning
# pipeline (teacher generation, QLoRA training, evaluation)', Date: 2026-10-07

from __future__ import annotations

import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TASK_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
OUTPUT_DIR = TASK_DIR / "outputs"
RAW_PATH = DATA_DIR / "raw_cases.jsonl"
SPLIT_PATHS = {s: DATA_DIR / f"{s}.jsonl" for s in ("train", "val", "test")}

# --------------------------------------------------------------------------- #
# Label spaces (the "ontology" both teacher and student must use)
# --------------------------------------------------------------------------- #
TYPOLOGIES = [
    "structuring",                    # splitting cash to stay under reporting thresholds
    "rapid_movement_of_funds",        # in-and-out pass-through within days
    "funnel_account",                 # many depositors in many places, one beneficiary
    "trade_based_money_laundering",   # over/under-invoicing, mismatched goods/payments
    "third_party_payments",           # payments for/from unrelated parties
    "high_risk_jurisdiction",         # flows to/from FATF-listed or secrecy jurisdictions
    "cash_intensive_business_anomaly",# cash volume inconsistent with business profile
    "account_takeover_fraud",         # compromised account draining
    "crypto_off_ramp",                # conversion of crypto proceeds to fiat
    "round_tripping",                 # funds leave and return disguised as new income
    "no_suspicious_activity",         # false positive with a legitimate explanation
]
RISK_LEVELS = ["low", "medium", "high"]
ACTIONS = ["close_no_action", "request_information", "enhanced_due_diligence", "escalate_sar"]

CUSTOMER_SEGMENTS = [
    "salaried retail customer", "university student", "retiree", "restaurant owner",
    "import/export SME", "money services business", "crypto trader", "registered charity",
    "real-estate agent", "freelance software developer", "politically exposed person",
    "used-car dealer", "medical practice", "e-commerce seller", "construction contractor",
]
JURISDICTIONS = [
    "United States", "United Kingdom", "Sri Lanka", "India", "United Arab Emirates", "Singapore",
    "Germany", "Nigeria", "Mexico", "Hong Kong", "Cayman Islands", "Panama", "Turkey",
    "Philippines", "Brazil", "Cyprus", "Myanmar", "Iran", "Canada", "Australia",
]
DIFFICULTY = ["clear-cut", "borderline", "misleading-surface"]  # surface signal may contradict truth

# --------------------------------------------------------------------------- #
# Dataset generation (teacher)
# --------------------------------------------------------------------------- #
# Teacher pool, tried in order. Free tiers have small DAILY token quotas, so when
# one is exhausted generation moves to the next. Every teacher differs from the
# student (Qwen2.5-1.5B), as the brief requires; the teacher is recorded per case.
TEACHER_POOL = [
    {"name": "groq/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1",
     "api_key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b", "cases_per_call": 3, "max_tokens": 8000},
    # Groq counts max_tokens against the 8K tokens/minute limit: keep prompt + max_tokens < 8K.
    {"name": "groq/qwen3.8-27b", "base_url": "https://api.groq.com/openai/v1",
     "api_key_env": "GROQ_API_KEY", "model": "qwen/qwen3.8-27b", "cases_per_call": 1, "max_tokens": 3000},
    # OpenRouter delisted the free gpt-oss-120b during generation (404); Nemotron-3 Super is free.
    {"name": "openrouter/nemotron-3-super-120b", "base_url": "https://openrouter.ai/api/v1",
     "api_key_env": "OPENROUTER_API_KEY", "model": "nvidia/nemotron-3-super-120b-a12b:free",
     "cases_per_call": 3, "max_tokens": 8000},
]
TEACHER_MODEL = TEACHER_POOL[0]["model"]  # primary teacher (documentation)
# "medium": low effort produced arithmetic errors in gold totals during the trial batch
TEACHER_REASONING_EFFORT = "medium"
TEACHER_TEMPERATURE = 0.9            # high: we want varied narratives, labels are pinned by seeds
TEACHER_MAX_TOKENS = 8000
CASES_PER_CALL = 3                   # amortises the system prompt; 4 caused JSON-mode failures
N_CASES = 150                        # -> 120 / 15 / 15 split
NO_SUSPICIOUS_SHARE = 0.22           # false positives are ~most real alerts; keep a sizeable share
RANDOM_SEED = 2026
MIN_TXNS, MAX_TXNS = 3, 10
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# --------------------------------------------------------------------------- #
# Student / fine-tuning
# --------------------------------------------------------------------------- #
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_REPO_ID = os.getenv("HF_REPO_ID", "YOUR_HF_USERNAME/qwen2.5-1.5b-aml-alert-triage")

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
GEN_MAX_NEW_TOKENS = 512
RAG_PERPLEXITY_THRESHOLD = float(os.getenv("RAG_PERPLEXITY_THRESHOLD", "1.35"))
RAG_TOP_K = 2
