"""Offline tests for the Task 2 data pipeline (no GPU, no API)."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest for grounding check, stratified split,
# chat formatting, triage parsing and per-epoch loss aggregation', Date: 2026-10-07

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aml_triage import config  # noqa: E402
from aml_triage.dataset import near_duplicate_stats, stratified_split, to_chat  # noqa: E402
from aml_triage.evaluate import field_metrics, hallucination_rate, parse_triage, suggest_label  # noqa: E402
from aml_triage.generate import build_seeds, consistency_errors, ungrounded_figures  # noqa: E402
from aml_triage.schema import Alert, Case, Triage, render_triage  # noqa: E402
from aml_triage.train import HYPERPARAMETERS, epoch_losses  # noqa: E402

ALERT = {
    "alert_id": "TM-1", "rule_triggered": "Cash deposits below threshold", "prior_alerts_12m": 0,
    "customer": {"segment": "restaurant owner", "occupation_or_business": "Diner", "account_age_months": 30,
                 "kyc_risk_rating": "low", "declared_monthly_turnover": 40000, "turnover_currency": "USD",
                 "residence_country": "United States"},
    "transactions": [
        {"date": "2026-03-01", "direction": "in", "type": "cash_deposit", "amount": 9500, "currency": "USD",
         "counterparty": "Self", "counterparty_country": "United States"},
        {"date": "2026-03-02", "direction": "in", "type": "cash_deposit", "amount": 9400, "currency": "USD",
         "counterparty": "Self", "counterparty_country": "United States"},
        {"date": "2026-03-03", "direction": "in", "type": "cash_deposit", "amount": 9800.5, "currency": "USD",
         "counterparty": "Self", "counterparty_country": "United States"}]}


def _triage(**over):
    base = {"typology": "structuring", "risk_level": "high", "recommended_action": "escalate_sar",
            "red_flags": ["Three cash deposits totalling 28,700.50 USD just under 10,000"],
            "mitigating_factors": [], "rationale": "Deposits are clustered just below the reporting threshold."}
    return Triage.model_validate({**base, **over})


def test_grounding_accepts_real_figures_and_rejects_invented():
    a = Alert.model_validate(ALERT)
    assert ungrounded_figures("deposits of 9,500 and 9,800.50 totalling 28,700.50 USD", a) == []
    assert ungrounded_figures("below the 10,000 CTR threshold; turnover 40,000", a) == []
    assert ungrounded_figures("total 2.87M USD", a) == ["2.87M"]
    assert ungrounded_figures("wire of 55,000.00 to Panama", a) == ["55,000.00"]


def test_consistency_requires_seed_typology_and_coherent_action():
    a = Alert.model_validate(ALERT)
    seed = {"true_typology": "structuring"}
    assert consistency_errors(seed, Case(case_id="c", seed=seed, alert=a, triage=_triage())) == []
    errs = consistency_errors({"true_typology": "funnel_account"},
                              Case(case_id="c", seed=seed, alert=a, triage=_triage(risk_level="low")))
    assert any("seed" in e for e in errs) and any("high" in e for e in errs)


def test_seeds_are_stratified_and_reproducible():
    s1, s2 = build_seeds(150), build_seeds(150)
    assert s1 == s2
    benign = sum(s["true_typology"] == "no_suspicious_activity" for s in s1)
    assert benign == round(150 * config.NO_SUSPICIOUS_SHARE)
    assert {s["true_typology"] for s in s1} == set(config.TYPOLOGIES)


def test_split_is_80_10_10_and_disjoint():
    a = Alert.model_validate(ALERT)
    cases = [Case(case_id=f"c{i}", seed={}, alert=a,
                  triage=_triage(typology=config.TYPOLOGIES[i % len(config.TYPOLOGIES)])) for i in range(150)]
    sp = stratified_split(cases)
    assert [len(sp[k]) for k in ("train", "val", "test")] == [120, 15, 15]
    ids = [c.case_id for k in sp for c in sp[k]]
    assert len(ids) == len(set(ids)) == 150


def test_chat_record_roles_and_roundtrip():
    rec = to_chat(Case(case_id="c", seed={}, alert=Alert.model_validate(ALERT), triage=_triage()))
    assert [m["role"] for m in rec["prompt"]] == ["system", "user"]
    assert rec["completion"][0]["role"] == "assistant"
    assert parse_triage(rec["completion"][0]["content"]) == _triage()


def test_parse_and_field_metrics():
    gold = render_triage(_triage())
    wrong = render_triage(_triage(typology="funnel_account"))
    assert parse_triage("```json\n" + gold + "\n```") is not None
    assert parse_triage("I think it's structuring.") is None
    m = field_metrics([gold, wrong, "garbage"], [gold, gold, gold])
    assert m["valid_json_schema"] == pytest.approx(2 / 3) and m["typology_acc"] == pytest.approx(1 / 3)
    assert m["action_acc"] == pytest.approx(2 / 3)


def test_suggested_labels_and_rate():
    a = Alert.model_validate(ALERT)
    gold = render_triage(_triage())
    assert suggest_label(gold, gold, a, None)[0] == "correct"
    halluc = render_triage(_triage(red_flags=["Wire of 77,000.00 to Cyprus"]))
    assert suggest_label(halluc, gold, a, None)[0] == "hallucinated"
    assert suggest_label(render_triage(_triage(recommended_action="enhanced_due_diligence")), gold, a, None)[0] == "partially_correct"
    assert hallucination_rate(["correct", "hallucinated", "correct", "partially_correct"]) == 25.0


def test_near_duplicates_detects_templates():
    same = ["the customer deposited cash at the branch every day this week"] * 3
    assert near_duplicate_stats(same)["max_jaccard"] == 1.0


def test_epoch_losses_aggregation():
    hist = [{"epoch": 0.5, "loss": 2.0}, {"epoch": 1.0, "loss": 1.6}, {"epoch": 1.0, "eval_loss": 1.5},
            {"epoch": 1.5, "loss": 1.2}, {"epoch": 2.0, "loss": 1.0}, {"epoch": 2.0, "eval_loss": 1.1}]
    rows = epoch_losses(hist)
    assert rows[0] == {"epoch": 1, "train_loss": pytest.approx(1.8), "val_loss": 1.5}
    assert rows[1]["val_loss"] == 1.1


def test_every_required_hyperparameter_is_justified():
    required = ["lora_r", "lora_alpha", "target_modules", "learning_rate", "lr_scheduler_type",
                "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "max_seq_length"]
    for k in required:
        assert k in HYPERPARAMETERS and len(HYPERPARAMETERS[k][1]) > 40


def test_trainable_params_cast_to_fp32_frozen_untouched():
    torch = pytest.importorskip("torch")
    from aml_triage.train import cast_trainable_to_fp32

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(torch.bfloat16)
    model[0].requires_grad_(False)                 # frozen "base"
    before = cast_trainable_to_fp32(model)
    assert before == {"torch.bfloat16": 20}
    assert all(p.dtype == torch.float32 for p in model[1].parameters())
    assert all(p.dtype == torch.bfloat16 for p in model[0].parameters())
