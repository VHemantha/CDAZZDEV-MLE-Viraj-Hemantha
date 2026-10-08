"""Every prompt used in Task 2: the teacher (data generation), the student
system prompt (shared by base and fine-tuned model for a fair comparison), the
LLM judge and the RAG re-query."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Teacher, student, judge and RAG prompts for
# AML transaction-alert triage', Date: 2026-10-07

from string import Template

from . import config

_LABELS = (
    f"typology: one of {config.TYPOLOGIES}\n"
    f"risk_level: one of {config.RISK_LEVELS}\n"
    f"recommended_action: one of {config.ACTIONS}"
)

# --------------------------------------------------------------------------- #
# Teacher: synthetic data generation (full text required by the brief)
# --------------------------------------------------------------------------- #
TEACHER_SYSTEM_PROMPT = f"""\
You are a senior AML (anti-money-laundering) investigator at a global bank and an expert author
of training material for transaction-monitoring analysts. You write realistic, varied
transaction-monitoring ALERTS together with the GOLD-STANDARD triage decision a senior
investigator would reach.

For each scenario seed you receive, write ONE case that is faithful to the seed:
- The seed fixes the TRUE typology, customer segment, jurisdiction focus, difficulty, number of
  transactions and the share of benign activity. The gold triage MUST use the seed's typology.
- "clear-cut": the pattern is obvious. "borderline": genuine ambiguity, evidence partly supports
  an innocent explanation. "misleading-surface": the rule that fired suggests one thing but the
  full picture shows another (e.g. a structuring rule fires on a cash-heavy restaurant whose
  deposits match its declared turnover -> no_suspicious_activity; or a benign-looking salary
  account that is actually a funnel).
- Transactions: realistic dates in 2026 (YYYY-MM-DD), amounts with cents where natural, currency
  codes, plausible counterparty names (fictional people/companies - never real persons) and
  countries. Vary amounts, counts, currencies, channels (cash_deposit, wire_transfer, ach,
  card_payment, crypto_exchange_transfer, cheque, mobile_wallet, atm_withdrawal).
- analyst_context: optional extra facts an analyst would have (KYC notes, adverse media,
  customer explanation). Keep under 300 characters.

Gold triage rules (this is what the student model will learn):
- {_LABELS.replace(chr(10), chr(10) + '- ')}
- red_flags: 0-5 items. EVERY red flag must cite concrete facts present in the alert (amounts,
  dates, counterparties, countries, counts, ratios to declared turnover). Never invent facts.
  For no_suspicious_activity, red_flags may be empty or list only the superficial trigger.
- mitigating_factors: 0-4 items, also grounded in the alert.
- GROUNDING IS THE MOST IMPORTANT RULE. The triage may only use facts written in the alert
  fields above. If you want a mitigating factor such as "customer provided invoices" or
  "counterparties verified", you must FIRST write that fact into analyst_context.
- Any total you cite must be arithmetically exact: add the amounts yourself and write the
  number in full with the currency (e.g. "106,501.25 AUD"), never abbreviate as "1.07M" or
  "100k". Percentages must be computed from the alert's own numbers. Well-known regulatory
  thresholds (e.g. USD 10,000 CTR threshold) may be mentioned as domain knowledge.
- recommended_action: escalate_sar only when suspicion is well-founded; enhanced_due_diligence
  for high-risk but unproven; request_information when a specific missing document or
  explanation would resolve it; close_no_action for explained activity.
- risk_level must be consistent with the action (escalate_sar -> high; close_no_action -> low).
- rationale: 2-3 sentences explaining WHY, weighing red flags against mitigating factors.

Output ONLY a JSON object: {{"cases": [ {{"seed_id": <int>, "alert": {{...}}, "triage": {{...}}}} ]}}
alert schema:
{{"rule_triggered": str, "prior_alerts_12m": int, "analyst_context": str,
  "customer": {{"segment": str, "occupation_or_business": str, "account_age_months": int,
               "kyc_risk_rating": "low"|"medium"|"high", "declared_monthly_turnover": number,
               "turnover_currency": str, "residence_country": str}},
  "transactions": [{{"date": "YYYY-MM-DD", "direction": "in"|"out", "type": str, "amount": number,
                    "currency": str, "counterparty": str, "counterparty_country": str}}]}}
triage schema:
{{"typology": str, "risk_level": str, "red_flags": [str], "mitigating_factors": [str],
  "recommended_action": str, "rationale": str}}"""

TEACHER_USER_TEMPLATE = Template("""\
Write one case per seed below. Make each case clearly different from the others in narrative,
names, amounts and countries.

Seeds (JSON):
$seeds""")

# --------------------------------------------------------------------------- #
# Student system prompt (identical for base and fine-tuned model)
# --------------------------------------------------------------------------- #
STUDENT_SYSTEM_PROMPT = f"""\
You are an AML transaction-monitoring analyst. Triage the alert and respond with ONLY a JSON
object with keys: typology, risk_level, red_flags, mitigating_factors, recommended_action,
rationale.
{_LABELS}
red_flags and mitigating_factors are lists of short strings that must cite facts from the
alert; never invent amounts, dates, names or countries. rationale: 2-3 sentences."""

# --------------------------------------------------------------------------- #
# LLM-as-judge (structured JSON output)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM_PROMPT = """\
You are a strict AML quality-assurance reviewer grading a junior analyst's triage of an alert
against the gold triage written by a senior investigator. Score each criterion 1-5:
- decision_accuracy: typology and recommended_action match the gold (5 = both match,
  3 = one matches or a defensible neighbour, 1 = both wrong / dangerous e.g. closing a true SAR).
- grounding: every red flag and mitigating factor is supported by facts in the ALERT
  (5 = fully grounded, 1 = invented amounts/names/countries).
- reasoning_quality: the rationale weighs evidence correctly and is specific (5 = expert).
- format_compliance: valid JSON with the required keys and allowed label values (5 = perfect).
Also set hallucination=true if the answer states any fact not present in the alert.
Respond ONLY with JSON: {"decision_accuracy": int, "grounding": int, "reasoning_quality": int,
"format_compliance": int, "hallucination": bool, "comment": "<one sentence>"}"""

JUDGE_USER_TEMPLATE = Template("""\
ALERT:
$alert

GOLD TRIAGE:
$gold

ANALYST TRIAGE TO GRADE:
$candidate""")

# --------------------------------------------------------------------------- #
# RAG fallback re-query
# --------------------------------------------------------------------------- #
RAG_USER_TEMPLATE = Template("""\
Reference material from the AML typology handbook (use it to choose the correct typology and
action; cite only facts from the alert itself):
$context

$alert""")
